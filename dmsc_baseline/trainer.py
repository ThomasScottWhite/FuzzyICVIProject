from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .clustering import FCMResult, fuzzy_c_means
from .metrics import exact_silhouette
from .model import Autoencoder


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


@dataclass
class TrainingResult:
    model: Autoencoder
    history: list[dict[str, int | float | bool | None]]
    final_clustering: FCMResult
    final_silhouette: float | None
    evaluation_indices: np.ndarray
    ae_seconds: float
    fcm_seconds: float
    validation_seconds: float
    gate_triggers: int
    adjustments_applied: int
    device: str


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class DMSCTrainer:
    """Train the reconstruction autoencoder with periodic DMSC FCM guidance."""

    def __init__(self, config: DMSCConfig):
        self.config = config

    def _select_device(self) -> torch.device:
        requested = self.config.device
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch installation has no CUDA device.")
        return device

    def _encode_all(self, model: Autoencoder, x: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model.encode(x)

    def fit(self, features: np.ndarray | torch.Tensor) -> TrainingResult:
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
        evaluation_size = cfg.evaluation_size or n_samples
        evaluation_size = min(evaluation_size, n_samples)
        eval_rng = np.random.default_rng(cfg.seed + 1)
        if evaluation_size == n_samples:
            evaluation_indices = np.arange(n_samples, dtype=np.int64)
        else:
            evaluation_indices = np.sort(
                eval_rng.choice(n_samples, size=evaluation_size, replace=False).astype(np.int64)
            )
        eval_index_tensor = torch.as_tensor(evaluation_indices, dtype=torch.long, device=device)

        model = Autoencoder(input_dim, cfg.hidden_dim, cfg.latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        reconstruction_loss = nn.MSELoss(reduction="mean")
        shuffle_generator = torch.Generator(device=device)
        shuffle_generator.manual_seed(cfg.seed + 2)

        history: list[dict[str, int | float | bool | None]] = []
        ae_seconds = 0.0
        fcm_seconds = 0.0
        validation_seconds = 0.0
        gate_triggers = 0
        adjustments_applied = 0
        pending_adjustment: tuple[torch.Tensor, torch.Tensor] | None = None
        final_clustering: FCMResult | None = None
        final_silhouette: float | None = None

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            order = torch.randperm(n_samples, generator=shuffle_generator, device=device)
            epoch_loss_total = 0.0
            epoch_sample_count = 0
            adjustment = pending_adjustment
            pending_adjustment = None
            if adjustment is not None:
                adjustments_applied += 1
            epoch_started = time.perf_counter()

            for start in range(0, n_samples, cfg.batch_size):
                indices = order[start : start + cfg.batch_size]
                batch = x[indices]
                optimizer.zero_grad(set_to_none=True)
                latent = model.encode(batch)
                if adjustment is not None:
                    centers, full_labels = adjustment
                    assigned_centers = centers[full_labels[indices]]
                    latent = latent + cfg.adjustment_gamma * (assigned_centers - latent)
                reconstructed = model.decode(latent)
                loss = reconstruction_loss(reconstructed, batch)
                loss.backward()
                optimizer.step()
                batch_count = batch.shape[0]
                epoch_loss_total += loss.detach().item() * batch_count
                epoch_sample_count += batch_count

            _synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_started
            ae_seconds += epoch_seconds
            record: dict[str, int | float | bool | None] = {
                "epoch": epoch,
                "reconstruction_loss": epoch_loss_total / max(epoch_sample_count, 1),
                "ae_seconds": epoch_seconds,
                "evaluated": False,
                "silhouette": None,
                "gate_triggered": False,
                "adjustment_applied": adjustment is not None,
                "fcm_iterations": None,
                "fcm_objective": None,
                "fcm_seconds": None,
                "validation_seconds": None,
            }

            if epoch % cfg.evaluation_interval == 0:
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

                silhouette_started = time.perf_counter()
                score = exact_silhouette(
                    embedding[eval_index_tensor],
                    clustering.labels[eval_index_tensor],
                    block_size=cfg.silhouette_block_size,
                )
                _synchronize(device)
                silhouette_seconds = time.perf_counter() - silhouette_started
                current_validation_seconds = encoding_seconds + silhouette_seconds
                validation_seconds += current_validation_seconds

                triggered = score is not None and score >= cfg.silhouette_threshold
                if triggered:
                    gate_triggers += 1
                    pending_adjustment = (clustering.centers.detach(), clustering.labels.detach())
                final_clustering = clustering
                final_silhouette = score
                record.update(
                    {
                        "evaluated": True,
                        "silhouette": score,
                        "gate_triggered": triggered,
                        "fcm_iterations": clustering.iterations,
                        "fcm_objective": clustering.objective,
                        "fcm_seconds": current_fcm_seconds,
                        "validation_seconds": current_validation_seconds,
                    }
                )

            history.append(record)

        if final_clustering is None or cfg.epochs % cfg.evaluation_interval != 0:
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
            current_fcm_seconds = time.perf_counter() - fcm_started
            fcm_seconds += current_fcm_seconds

            silhouette_started = time.perf_counter()
            final_silhouette = exact_silhouette(
                embedding[eval_index_tensor],
                final_clustering.labels[eval_index_tensor],
                block_size=cfg.silhouette_block_size,
            )
            _synchronize(device)
            validation_seconds += encoding_seconds + (time.perf_counter() - silhouette_started)

        _synchronize(device)
        return TrainingResult(
            model=model,
            history=history,
            final_clustering=final_clustering,
            final_silhouette=final_silhouette,
            evaluation_indices=evaluation_indices,
            ae_seconds=ae_seconds,
            fcm_seconds=fcm_seconds,
            validation_seconds=validation_seconds,
            gate_triggers=gate_triggers,
            adjustments_applied=adjustments_applied,
            device=str(device),
        )

    def config_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self.config)
