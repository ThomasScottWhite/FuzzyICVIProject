"""Prespecified guidance decisions for Experiment 2.

All policies see the same full-data FCM result and fixed reference sample at
each scheduled check. They never inspect ground-truth labels.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from dmsc_custom.clustering import FCMResult, fuzzy_c_means
from dmsc_custom.trainer import DMSCConfig, GuidanceDecision

CONDITIONS = (
    "no_guidance", "fixed", "relative", "null_calibrated", "stability",
    "confidence_weighted", "soft",
)


def _silhouette_from_distances(distances: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Exact mean silhouette from a reusable pairwise distance matrix."""
    n = labels.numel()
    unique, inverse, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    if unique.numel() < 2 or unique.numel() >= n:
        return None
    membership = F.one_hot(inverse, num_classes=unique.numel()).to(distances.dtype)
    sums = distances @ membership
    own_sum = sums.gather(1, inverse[:, None]).squeeze(1)
    own_count = counts[inverse]
    within = own_sum / (own_count - 1).clamp_min(1).to(distances.dtype)
    means = sums / counts.to(distances.dtype)[None, :]
    means.scatter_(1, inverse[:, None], torch.inf)
    nearest = means.min(dim=1).values
    denominator = torch.maximum(within, nearest)
    score = (nearest - within) / denominator.clamp_min(1e-12)
    score = torch.where((own_count > 1) & (denominator > 0), score, 0.0)
    return float(score.mean().item())


def _confidence_weights(embedding: torch.Tensor, clustering: FCMResult) -> torch.Tensor:
    """Local centroid silhouette times normalized maximum FCM membership."""
    labels = clustering.labels
    active = torch.unique(labels, sorted=True)
    if active.numel() < 2:
        return torch.zeros(len(labels), dtype=embedding.dtype, device=embedding.device)
    positions = torch.searchsorted(active, labels)
    distances = torch.cdist(embedding, clustering.centers[active])
    own = distances.gather(1, positions[:, None]).squeeze(1)
    distances.scatter_(1, positions[:, None], torch.inf)
    nearest = distances.min(dim=1).values
    local = ((nearest - own) / torch.maximum(nearest, own).clamp_min(1e-12)).clamp(0, 1)
    k = clustering.memberships.shape[1]
    confidence = ((clustering.memberships.max(dim=1).values - 1.0 / k) / (1.0 - 1.0 / k)).clamp(0, 1)
    return (local * confidence).detach()


class GuidancePolicy:
    def __init__(self, name: str, config: DMSCConfig):
        if name not in CONDITIONS:
            raise ValueError(f"Unknown guidance condition: {name}")
        self.name = name
        self.config = config
        self.early_scores: list[float] = []
        self.relative_threshold: float | None = None

    def __call__(
        self, epoch: int, embedding: torch.Tensor, clustering: FCMResult,
        score: float | None, reference: np.ndarray,
    ) -> GuidanceDecision:
        name = self.name
        if name == "no_guidance":
            return GuidanceDecision(0.0)
        if name == "fixed":
            return GuidanceDecision(float(score is not None and score >= 0.5),
                                    diagnostics={"threshold": 0.5})
        if name == "relative":
            if self.relative_threshold is None:
                if score is not None:
                    self.early_scores.append(score)
                if len(self.early_scores) == 5:
                    self.relative_threshold = max(0.0, float(np.quantile(self.early_scores, 0.75)))
                return GuidanceDecision(0.0, diagnostics={"threshold": self.relative_threshold})
            return GuidanceDecision(
                float(score is not None and score >= self.relative_threshold),
                diagnostics={"threshold": self.relative_threshold},
            )
        if name == "null_calibrated":
            if score is None:
                return GuidanceDecision(0.0)
            index = torch.as_tensor(reference, dtype=torch.long, device=embedding.device)
            with torch.no_grad():
                distances = torch.cdist(embedding[index], embedding[index])
                reference_labels = clustering.labels[index]
                generator = torch.Generator(device=embedding.device)
                generator.manual_seed(self.config.seed * 1_000_003 + epoch)
                null_scores = []
                for _ in range(19):
                    permuted = reference_labels[torch.randperm(len(index), generator=generator, device=embedding.device)]
                    value = _silhouette_from_distances(distances, permuted)
                    if value is not None:
                        null_scores.append(value)
            if len(null_scores) != 19:
                return GuidanceDecision(0.0, diagnostics={"null_p": None})
            p_value = (1 + sum(value >= score for value in null_scores)) / 20
            return GuidanceDecision(float(p_value <= 0.05), diagnostics={
                "null_p": p_value, "null_max": max(null_scores),
            })
        if name == "stability":
            index = torch.as_tensor(reference, dtype=torch.long, device=embedding.device)
            original = clustering.labels[index].detach().cpu().numpy()
            generator = torch.Generator(device=embedding.device)
            generator.manual_seed(self.config.seed * 1_000_003 + epoch)
            scale = embedding.std(dim=0).clamp_min(1e-6) * 0.02
            stability_scores = []
            with torch.no_grad():
                for repeat in range(2):
                    noise = torch.randn(embedding.shape, generator=generator,
                                        dtype=embedding.dtype, device=embedding.device)
                    perturbed = embedding + noise * scale
                    alternate = fuzzy_c_means(
                        perturbed, self.config.n_clusters,
                        fuzzifier=self.config.fuzzifier,
                        max_iter=self.config.fcm_max_iter,
                        tolerance=self.config.fcm_tolerance,
                        seed=self.config.seed + epoch + 10_000 + repeat,
                        block_size=self.config.fcm_block_size,
                    )
                    labels = alternate.labels[index].detach().cpu().numpy()
                    stability_scores.append(float(adjusted_rand_score(original, labels)))
            stability = float(np.mean(stability_scores))
            return GuidanceDecision(float(stability >= 0.9), diagnostics={"stability_ari": stability})
        if name == "confidence_weighted":
            with torch.no_grad():
                weights = _confidence_weights(embedding, clustering)
            mean_weight = float(weights.mean().item())
            return GuidanceDecision(float(mean_weight > 0.0), sample_weights=weights,
                                    diagnostics={"mean_weight": mean_weight})
        if name == "soft":
            if score is None:
                return GuidanceDecision(0.0)
            strength = 1.0 / (1.0 + math.exp(-(score - 0.5) / 0.05))
            return GuidanceDecision(strength, diagnostics={"soft_weight": strength})
        raise AssertionError(name)
