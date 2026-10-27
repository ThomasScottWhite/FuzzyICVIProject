from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import psutil
import torch
from tqdm import tqdm
from torch import nn

from .clustering import FCMResult, fuzzy_c_means
from .metrics import exact_silhouette, simplified_silhouette
from .model import Autoencoder

_METRIC_NAMES = ("reference", "sampled", "simplified")


@dataclass
class DMSCConfig:
    n_clusters: int
    hidden_dim: int = 128
    latent_dim: int = 3
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    fuzzifier: float = 2.0
    silhouette_threshold: float = 0.5
    adjustment_gamma: float = 1e-3
    evaluation_interval: int = 10
    evaluation_epochs: tuple[int, ...] | None = None
    gate_metric: str = "reference"
    validation_metrics: tuple[str, ...] | None = None
    fcm_max_iter: int = 50
    fcm_tolerance: float = 1e-4
    evaluation_size: int | None = None
    fcm_block_size: int = 65_536
    silhouette_block_size: int = 512
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if self.hidden_dim < 1 or self.latent_dim < 1:
            raise ValueError("hidden_dim and latent_dim must be positive.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0 or self.adjustment_gamma < 0:
            raise ValueError("learning_rate must be positive and adjustment_gamma non-negative.")
        if self.fuzzifier <= 1 or not -1 <= self.silhouette_threshold <= 1:
            raise ValueError("fuzzifier must exceed 1 and threshold must be in [-1, 1].")
        if self.evaluation_interval < 1 or self.fcm_max_iter < 1:
            raise ValueError("evaluation_interval and fcm_max_iter must be positive.")
        if self.fcm_tolerance <= 0:
            raise ValueError("fcm_tolerance must be positive.")
        if self.evaluation_size is not None and self.evaluation_size < 2:
            raise ValueError("evaluation_size must be at least 2 when specified.")
        if self.fcm_block_size < 1 or self.silhouette_block_size < 1:
            raise ValueError("Distance block sizes must be positive.")
        if self.gate_metric not in _METRIC_NAMES:
            raise ValueError(f"gate_metric must be one of {_METRIC_NAMES}.")
        if self.validation_metrics is not None:
            if len(set(self.validation_metrics)) != len(self.validation_metrics):
                raise ValueError("validation_metrics cannot contain duplicates.")
            unknown = set(self.validation_metrics) - set(_METRIC_NAMES)
            if unknown:
                raise ValueError(f"Unknown validation metrics: {sorted(unknown)}.")
            if self.gate_metric not in self.validation_metrics:
                raise ValueError("validation_metrics must include gate_metric.")
        if self.evaluation_epochs is not None:
            epochs = tuple(self.evaluation_epochs)
            if not epochs or any(epoch < 1 or epoch > self.epochs for epoch in epochs):
                raise ValueError("evaluation_epochs must contain epochs in [1, epochs].")
            if tuple(sorted(set(epochs))) != epochs:
                raise ValueError("evaluation_epochs must be strictly increasing and unique.")
            self.evaluation_epochs = epochs

    @property
    def scheduled_evaluation_epochs(self) -> tuple[int, ...]:
        if self.evaluation_epochs is not None:
            return self.evaluation_epochs
        return tuple(range(self.evaluation_interval, self.epochs + 1, self.evaluation_interval))

    @property
    def active_validation_metrics(self) -> tuple[str, ...]:
        return self.validation_metrics or (self.gate_metric,)


@dataclass
class TrainingResult:
    model: Autoencoder
    history: list[dict[str, int | float | bool | str | None]]
    final_clustering: FCMResult
    final_silhouette: float | None
    final_gate_score: float | None
    final_scores: dict[str, float | None]
    evaluation_indices: np.ndarray
    sampled_indices: np.ndarray
    ae_seconds: float
    fcm_seconds: float
    validation_seconds: float
    metric_seconds: dict[str, float]
    peak_memory_bytes_by_metric: dict[str, int]
    gate_triggers: int
    adjustments_applied: int
    cumulative_guidance_strength: float
    device: str


@dataclass
class GuidanceDecision:
    """Guidance for the interval following a scheduled clustering check.

    strength and optional per-sample weights multiply adjustment_gamma.
    """

    strength: float
    sample_weights: torch.Tensor | None = None
    diagnostics: dict[str, float | int | None] = field(default_factory=dict)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_indices(
    indices: np.ndarray | None,
    *,
    n_samples: int,
    default_size: int,
    seed: int,
    name: str,
) -> np.ndarray:
    if indices is None:
        size = min(default_size, n_samples)
        if size == n_samples:
            return np.arange(n_samples, dtype=np.int64)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_samples, size=size, replace=False).astype(np.int64))

    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two indices.")
    if np.any(values < 0) or np.any(values >= n_samples):
        raise ValueError(f"{name} contains indices outside the feature matrix.")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} cannot contain duplicate indices.")
    return np.sort(values)


def _measure_metric(
    operation: Callable[[], float | None], device: torch.device
) -> tuple[float | None, float, int]:
    """Measure a validation call's elapsed time and peak process/device memory."""
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.01):
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        score = operation()
        _synchronize(device)
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join()
    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = int(peak_rss[0])
    return score, elapsed, peak_memory


class DMSCTrainer:
    """Train DMSC with configurable evaluation schedules and silhouette gates."""

    def __init__(self, config: DMSCConfig):
        self.config = config

    def _select_device(self) -> torch.device:
        requested = self.config.device
        if requested == "auto":
            if not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device("cuda", torch.cuda.current_device())
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch installation has no CUDA device.")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def _encode_all(self, model: Autoencoder, x: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model.encode(x)

    def _score_metrics(
        self,
        metrics: tuple[str, ...],
        embedding: torch.Tensor,
        clustering: FCMResult,
        evaluation_index_tensor: torch.Tensor,
        sampled_index_tensor: torch.Tensor,
        device: torch.device,
    ) -> tuple[dict[str, float | None], dict[str, float], dict[str, int]]:
        scores: dict[str, float | None] = {}
        elapsed_by_metric: dict[str, float] = {}
        memory_by_metric: dict[str, int] = {}
        for metric in metrics:
            if metric == "reference":
                operation = lambda: exact_silhouette(
                    embedding[evaluation_index_tensor],
                    clustering.labels[evaluation_index_tensor],
                    block_size=self.config.silhouette_block_size,
                )
            elif metric == "sampled":
                operation = lambda: exact_silhouette(
                    embedding[sampled_index_tensor],
                    clustering.labels[sampled_index_tensor],
                    block_size=self.config.silhouette_block_size,
                )
            else:
                operation = lambda: simplified_silhouette(
                    embedding,
                    clustering.labels,
                    clustering.centers,
                    block_size=self.config.fcm_block_size,
                )
            score, elapsed, peak = _measure_metric(operation, device)
            scores[metric] = score
            elapsed_by_metric[metric] = elapsed
            memory_by_metric[metric] = peak
        return scores, elapsed_by_metric, memory_by_metric

    def fit(
        self,
        features: np.ndarray | torch.Tensor,
        *,
        evaluation_indices: np.ndarray | None = None,
        sampled_indices: np.ndarray | None = None,
        epoch_observer: Callable[
            [int, torch.Tensor, FCMResult, dict[str, int | float | bool | str | None]], None
        ] | None = None,
        epoch_observer_epochs: tuple[int, ...] | None = None,
        guidance_decider: Callable[
            [int, torch.Tensor, FCMResult, float | None, np.ndarray], GuidanceDecision
        ] | None = None,
        progress_desc: str | None = None,
        progress_position: int = 0,
        report_device: bool = False,
    ) -> TrainingResult:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        device = self._select_device()

        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        if x.ndim != 2 or x.shape[0] < cfg.n_clusters:
            raise ValueError("features must be [n_samples, n_features] with n_samples >= n_clusters.")
        if not torch.isfinite(x).all():
            raise ValueError("features must contain only finite values.")
        if x.min().item() < -1e-6 or x.max().item() > 1.0 + 1e-6:
            raise ValueError("Features must be scaled to [0, 1] because the decoder ends in sigmoid.")

        n_samples, input_dim = x.shape
        default_eval_size = cfg.evaluation_size or n_samples
        evaluation_indices = _validate_indices(
            evaluation_indices,
            n_samples=n_samples,
            default_size=default_eval_size,
            seed=cfg.seed + 1,
            name="evaluation_indices",
        )
        requested_metrics = cfg.active_validation_metrics
        if sampled_indices is None:
            sample_size = min(2_000, evaluation_indices.size)
            if sample_size == evaluation_indices.size:
                sampled_indices = evaluation_indices.copy()
            else:
                sampled_indices = np.sort(
                    np.random.default_rng(cfg.seed + 3)
                    .choice(evaluation_indices, size=sample_size, replace=False)
                    .astype(np.int64)
                )
        else:
            sampled_indices = _validate_indices(
                sampled_indices,
                n_samples=n_samples,
                default_size=min(2_000, evaluation_indices.size),
                seed=cfg.seed + 3,
                name="sampled_indices",
            )
        eval_index_tensor = torch.as_tensor(evaluation_indices, dtype=torch.long, device=device)
        sampled_index_tensor = torch.as_tensor(sampled_indices, dtype=torch.long, device=device)

        model = Autoencoder(input_dim, cfg.hidden_dim, cfg.latent_dim).to(device)
        if report_device:
            if device.type == "cuda":
                device_name = torch.cuda.get_device_name(device)
                model_on_device = all(parameter.device == device for parameter in model.parameters())
                if not x.is_cuda or not model_on_device:
                    raise RuntimeError("CUDA was selected, but training tensors or model parameters are not on CUDA.")
                tqdm.write(
                    f"GPU selected: {device_name} ({device}); data and model are on CUDA. "
                    "A successful first epoch will confirm GPU execution."
                )
            else:
                cuda_available = torch.cuda.is_available()
                detail = (
                    "CUDA is available, but CPU was selected. Use --device cuda:0 to enable it."
                    if cuda_available
                    else "CUDA is unavailable in this PyTorch environment."
                )
                tqdm.write(f"GPU acceleration inactive: using {device}. {detail}")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        reconstruction_loss = nn.MSELoss(reduction="mean")
        shuffle_generator = torch.Generator(device=device)
        shuffle_generator.manual_seed(cfg.seed + 2)

        history: list[dict[str, int | float | bool | str | None]] = []
        ae_seconds = 0.0
        fcm_seconds = 0.0
        validation_seconds = 0.0
        metric_seconds = {name: 0.0 for name in requested_metrics}
        peak_memory_bytes_by_metric = {name: 0 for name in requested_metrics}
        gate_triggers = 0
        adjustments_applied = 0
        cumulative_guidance_strength = 0.0
        # The most recent scheduled gate decision controls every training epoch
        # until the next scheduled evaluation replaces it.
        active_adjustment: tuple[
            torch.Tensor, torch.Tensor, float, torch.Tensor | None
        ] | None = None
        final_clustering: FCMResult | None = None
        final_scores: dict[str, float | None] = {}
        scheduled_epochs = set(cfg.scheduled_evaluation_epochs)
        batches_per_epoch = (n_samples + cfg.batch_size - 1) // cfg.batch_size
        progress_bar = (
            tqdm(
                total=cfg.epochs * batches_per_epoch,
                desc=progress_desc,
                unit="batch",
                position=progress_position,
                leave=False,
                dynamic_ncols=True,
            )
            if progress_desc is not None
            else None
        )
        cuda_epoch_confirmed = False

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            embedding: torch.Tensor | None = None
            clustering: FCMResult | None = None
            order = torch.randperm(n_samples, generator=shuffle_generator, device=device)
            epoch_loss_total = 0.0
            epoch_sample_count = 0
            adjustment = active_adjustment
            if adjustment is not None:
                adjustments_applied += 1
                interval_strength = adjustment[2]
                if adjustment[3] is not None:
                    interval_strength *= float(adjustment[3].mean().item())
                cumulative_guidance_strength += interval_strength
            epoch_started = time.perf_counter()

            for start in range(0, n_samples, cfg.batch_size):
                indices = order[start : start + cfg.batch_size]
                batch = x[indices]
                optimizer.zero_grad(set_to_none=True)
                latent = model.encode(batch)
                if adjustment is not None:
                    centers, full_labels, strength, sample_weights = adjustment
                    assigned_centers = centers[full_labels[indices]]
                    scale = cfg.adjustment_gamma * strength
                    if sample_weights is not None:
                        latent = latent + scale * sample_weights[indices, None] * (assigned_centers - latent)
                    else:
                        latent = latent + scale * (assigned_centers - latent)
                reconstructed = model.decode(latent)
                loss = reconstruction_loss(reconstructed, batch)
                loss.backward()
                optimizer.step()
                batch_count = batch.shape[0]
                batch_loss = loss.detach().item()
                epoch_loss_total += batch_loss * batch_count
                epoch_sample_count += batch_count
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        epoch=f"{epoch}/{cfg.epochs}", loss=f"{batch_loss:.5f}", refresh=False
                    )

            _synchronize(device)
            if report_device and device.type == "cuda" and not cuda_epoch_confirmed:
                tqdm.write(
                    f"GPU acceleration confirmed: completed epoch {epoch} on "
                    f"{torch.cuda.get_device_name(device)} ({device}); CUDA kernels synchronized successfully."
                )
                cuda_epoch_confirmed = True
            epoch_seconds = time.perf_counter() - epoch_started
            ae_seconds += epoch_seconds
            record: dict[str, int | float | bool | str | None] = {
                "epoch": epoch,
                "reconstruction_loss": epoch_loss_total / max(epoch_sample_count, 1),
                "ae_seconds": epoch_seconds,
                "evaluated": False,
                "silhouette": None,
                "gate_metric": cfg.gate_metric,
                "gate_score": None,
                "gate_triggered": False,
                "adjustment_applied": adjustment is not None,
                "guidance_strength": (
                    adjustment[2] * (float(adjustment[3].mean().item()) if adjustment[3] is not None else 1.0)
                    if adjustment is not None else 0.0
                ),
                "decision_seconds": None,
                "fcm_iterations": None,
                "fcm_objective": None,
                "fcm_seconds": None,
                "validation_seconds": None,
            }
            for metric in _METRIC_NAMES:
                record[f"silhouette_{metric}"] = None
                record[f"gate_{metric}"] = None
                record[f"validation_seconds_{metric}"] = None
                record[f"peak_memory_bytes_{metric}"] = None

            if epoch in scheduled_epochs:
                _synchronize(device)
                validation_started = time.perf_counter()
                embedding = self._encode_all(model, x)
                _synchronize(device)
                encoding_seconds = time.perf_counter() - validation_started

                fcm_started = time.perf_counter()
                clustering = fuzzy_c_means(
                    embedding,
                    cfg.n_clusters,
                    fuzzifier=cfg.fuzzifier,
                    max_iter=cfg.fcm_max_iter,
                    tolerance=cfg.fcm_tolerance,
                    seed=cfg.seed + epoch,
                    block_size=cfg.fcm_block_size,
                )
                _synchronize(device)
                current_fcm_seconds = time.perf_counter() - fcm_started
                fcm_seconds += current_fcm_seconds

                scores, metric_elapsed, metric_memory = self._score_metrics(
                    requested_metrics,
                    embedding,
                    clustering,
                    eval_index_tensor,
                    sampled_index_tensor,
                    device,
                )
                score_for_gate = scores[cfg.gate_metric]
                decision_started = time.perf_counter()
                if guidance_decider is None:
                    triggered = score_for_gate is not None and score_for_gate >= cfg.silhouette_threshold
                    decision = GuidanceDecision(strength=float(triggered))
                else:
                    decision = guidance_decider(
                        epoch, embedding, clustering, score_for_gate, evaluation_indices
                    )
                    if not 0.0 <= decision.strength <= 1.0:
                        raise ValueError("Guidance strength must be in [0, 1].")
                    if decision.sample_weights is not None:
                        weights = decision.sample_weights
                        if weights.shape != (n_samples,) or weights.device != device:
                            raise ValueError("Guidance sample weights must match all training samples and device.")
                        if not torch.isfinite(weights).all() or weights.min() < 0 or weights.max() > 1:
                            raise ValueError("Guidance sample weights must be finite and in [0, 1].")
                    triggered = decision.strength > 0.0
                _synchronize(device)
                decision_seconds = time.perf_counter() - decision_started
                if triggered:
                    gate_triggers += 1
                if triggered and cfg.adjustment_gamma > 0.0:
                    active_adjustment = (
                        clustering.centers.detach(), clustering.labels.detach(),
                        decision.strength,
                        decision.sample_weights.detach() if decision.sample_weights is not None else None,
                    )
                else:
                    active_adjustment = None
                final_clustering = clustering
                final_scores = scores
                current_validation_seconds = encoding_seconds + sum(metric_elapsed.values())
                validation_seconds += current_validation_seconds
                for metric, elapsed in metric_elapsed.items():
                    metric_seconds[metric] += elapsed
                    peak_memory_bytes_by_metric[metric] = max(
                        peak_memory_bytes_by_metric[metric], metric_memory[metric]
                    )

                record.update(
                    {
                        "evaluated": True,
                        "silhouette": scores.get("reference", score_for_gate),
                        "gate_score": score_for_gate,
                        "gate_triggered": triggered,
                        "decision_seconds": decision_seconds,
                        "fcm_iterations": clustering.iterations,
                        "fcm_objective": clustering.objective,
                        "fcm_seconds": current_fcm_seconds,
                        "validation_seconds": current_validation_seconds,
                    }
                )
                for metric, score in scores.items():
                    record[f"silhouette_{metric}"] = score
                    record[f"gate_{metric}"] = score is not None and score >= cfg.silhouette_threshold
                    record[f"validation_seconds_{metric}"] = metric_elapsed[metric]
                    record[f"peak_memory_bytes_{metric}"] = metric_memory[metric]
                for name, value in decision.diagnostics.items():
                    record[f"decision_{name}"] = value

            if epoch_observer is not None and (
                epoch_observer_epochs is None or epoch in epoch_observer_epochs
            ):
                # Diagnostics run after the epoch's weight update. Reuse the
                # scheduled clustering result; extra evaluations never affect
                # the gate or the active training adjustment.
                if clustering is None:
                    embedding = self._encode_all(model, x)
                    clustering = fuzzy_c_means(
                        embedding,
                        cfg.n_clusters,
                        fuzzifier=cfg.fuzzifier,
                        max_iter=cfg.fcm_max_iter,
                        tolerance=cfg.fcm_tolerance,
                        seed=cfg.seed + epoch,
                        block_size=cfg.fcm_block_size,
                    )
                assert embedding is not None
                epoch_observer(epoch, embedding, clustering, record)

            history.append(record)
            if progress_bar is not None:
                postfix: dict[str, str] = {
                    "epoch": f"{epoch}/{cfg.epochs}",
                    "loss": f"{record['reconstruction_loss']:.5f}",
                }
                if record["evaluated"]:
                    score = record["gate_score"]
                    postfix["gate"] = "undefined" if score is None else f"{score:.3f}"
                    postfix["trigger"] = "yes" if record["gate_triggered"] else "no"
                progress_bar.set_postfix(postfix)

        if progress_bar is not None:
            progress_bar.close()

        if final_clustering is None or cfg.epochs not in scheduled_epochs:
            _synchronize(device)
            validation_started = time.perf_counter()
            embedding = self._encode_all(model, x)
            _synchronize(device)
            encoding_seconds = time.perf_counter() - validation_started

            fcm_started = time.perf_counter()
            final_clustering = fuzzy_c_means(
                embedding,
                cfg.n_clusters,
                fuzzifier=cfg.fuzzifier,
                max_iter=cfg.fcm_max_iter,
                tolerance=cfg.fcm_tolerance,
                seed=cfg.seed + cfg.epochs + 1,
                block_size=cfg.fcm_block_size,
            )
            _synchronize(device)
            fcm_seconds += time.perf_counter() - fcm_started
            final_scores, metric_elapsed, metric_memory = self._score_metrics(
                requested_metrics,
                embedding,
                final_clustering,
                eval_index_tensor,
                sampled_index_tensor,
                device,
            )
            validation_seconds += encoding_seconds + sum(metric_elapsed.values())
            for metric, elapsed in metric_elapsed.items():
                metric_seconds[metric] += elapsed
                peak_memory_bytes_by_metric[metric] = max(
                    peak_memory_bytes_by_metric[metric], metric_memory[metric]
                )

        _synchronize(device)
        return TrainingResult(
            model=model,
            history=history,
            final_clustering=final_clustering,
            final_silhouette=final_scores.get("reference"),
            final_gate_score=final_scores.get(cfg.gate_metric),
            final_scores=final_scores,
            evaluation_indices=evaluation_indices,
            sampled_indices=sampled_indices,
            ae_seconds=ae_seconds,
            fcm_seconds=fcm_seconds,
            validation_seconds=validation_seconds,
            metric_seconds=metric_seconds,
            peak_memory_bytes_by_metric=peak_memory_bytes_by_metric,
            gate_triggers=gate_triggers,
            adjustments_applied=adjustments_applied,
            cumulative_guidance_strength=cumulative_guidance_strength,
            device=str(device),
        )

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)
