from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def exact_silhouette(
    embeddings: torch.Tensor, labels: torch.Tensor, block_size: int = 512
) -> float | None:
    """Compute the exact Euclidean silhouette while bounding pairwise memory by blocks."""
    n_samples = embeddings.shape[0]
    if embeddings.ndim != 2 or labels.ndim != 1 or labels.shape[0] != n_samples:
        raise ValueError("Expected embeddings [n, d] and labels [n].")
    if n_samples < 2:
        return None

    labels = labels.to(device=embeddings.device)
    unique, inverse, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    n_clusters = unique.numel()
    if n_clusters < 2 or n_clusters >= n_samples:
        return None

    masks = [labels == label for label in unique]
    scores = torch.empty(n_samples, dtype=embeddings.dtype, device=embeddings.device)
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        distances = torch.cdist(embeddings[start:end], embeddings)
        cluster_sums = torch.stack(
            [distances[:, mask].sum(dim=1) for mask in masks], dim=1
        )
        own_cluster = inverse[start:end]
        own_sum = cluster_sums.gather(1, own_cluster.unsqueeze(1)).squeeze(1)
        own_count = counts[own_cluster]
        own_mean = own_sum / (own_count - 1).clamp_min(1).to(embeddings.dtype)

        means = cluster_sums / counts.to(embeddings.dtype).unsqueeze(0)
        means.scatter_(1, own_cluster.unsqueeze(1), torch.inf)
        nearest_other_mean = means.min(dim=1).values
        denominator = torch.maximum(own_mean, nearest_other_mean)
        block_scores = (nearest_other_mean - own_mean) / denominator.clamp_min(1e-12)
        block_scores = torch.where(denominator > 0, block_scores, 0.0)
        block_scores = torch.where(own_count > 1, block_scores, 0.0)
        scores[start:end] = block_scores

    return float(scores.mean().item())


def simplified_silhouette(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    block_size: int = 65_536,
) -> float | None:
    """Compute the centroid-based silhouette in bounded sample blocks.

    Each point's within-cluster distance is its distance to its assigned center;
    its between-cluster distance is the distance to the nearest other active
    center. The score uses the usual (b - a) / max(a, b) formula.
    """
    n_samples = embeddings.shape[0]
    if embeddings.ndim != 2 or labels.ndim != 1 or labels.shape[0] != n_samples:
        raise ValueError("Expected embeddings [n, d] and labels [n].")
    if centers.ndim != 2 or centers.shape[1] != embeddings.shape[1]:
        raise ValueError("Expected centers [k, d] matching the embedding dimension.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if n_samples < 2:
        return None

    labels = labels.to(device=embeddings.device, dtype=torch.long)
    centers = centers.to(device=embeddings.device, dtype=embeddings.dtype)
    unique, inverse, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    n_clusters = unique.numel()
    if n_clusters < 2 or n_clusters >= n_samples:
        return None
    if unique[0].item() < 0 or unique[-1].item() >= centers.shape[0]:
        raise ValueError("Labels must index rows in centers.")

    active_centers = centers[unique]
    score_total = torch.zeros((), dtype=embeddings.dtype, device=embeddings.device)
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        cluster_positions = inverse[start:end]
        distances = torch.cdist(embeddings[start:end], active_centers)
        rows = torch.arange(end - start, device=embeddings.device)
        own_distance = distances[rows, cluster_positions]
        distances[rows, cluster_positions] = torch.inf
        nearest_other = distances.min(dim=1).values
        denominator = torch.maximum(own_distance, nearest_other)
        scores = (nearest_other - own_distance) / denominator.clamp_min(1e-12)
        scores = torch.where(denominator > 0, scores, 0.0)
        score_total += scores.sum()

    return float((score_total / n_samples).item())


def external_clustering_metrics(
    true_labels: np.ndarray, predicted_clusters: np.ndarray
) -> dict[str, float]:
    """Post-hoc ACC (Hungarian matched), NMI, and ARI."""
    true_labels = np.asarray(true_labels)
    predicted_clusters = np.asarray(predicted_clusters)
    if true_labels.ndim != 1 or predicted_clusters.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")
    if true_labels.shape[0] != predicted_clusters.shape[0]:
        raise ValueError("True and predicted labels must have the same length.")
    if true_labels.size == 0:
        raise ValueError("Cannot score an empty label vector.")

    true_values, true_encoded = np.unique(true_labels, return_inverse=True)
    cluster_values, cluster_encoded = np.unique(predicted_clusters, return_inverse=True)
    contingency = np.zeros((len(true_values), len(cluster_values)), dtype=np.int64)
    np.add.at(contingency, (true_encoded, cluster_encoded), 1)
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    accuracy = contingency[row_ind, col_ind].sum() / true_labels.size

    return {
        "ACC": float(accuracy),
        "NMI": float(normalized_mutual_info_score(true_labels, predicted_clusters)),
        "ARI": float(adjusted_rand_score(true_labels, predicted_clusters)),
    }
