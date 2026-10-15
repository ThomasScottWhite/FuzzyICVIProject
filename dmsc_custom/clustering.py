from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FCMResult:
    centers: torch.Tensor
    memberships: torch.Tensor
    labels: torch.Tensor
    iterations: int
    objective: float


def _squared_distances(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    # The clamp only removes tiny negative round-off from the algebraic form.
    x2 = (x * x).sum(dim=1, keepdim=True)
    c2 = (centers * centers).sum(dim=1).unsqueeze(0)
    return (x2 + c2 - 2.0 * (x @ centers.T)).clamp_min_(0.0)


def _distance_blocks(
    x: torch.Tensor, centers: torch.Tensor, block_size: int
):
    for start in range(0, x.shape[0], block_size):
        end = min(start + block_size, x.shape[0])
        yield start, end, _squared_distances(x[start:end], centers)


def two_stage_initial_centers(
    x: torch.Tensor, n_clusters: int, seed: int, block_size: int = 65_536
) -> torch.Tensor:
    """Initialize centers using the paper's farthest-from-center/midpoint rule."""
    n_samples = x.shape[0]
    if n_clusters < 2:
        raise ValueError("FCM requires at least two clusters.")
    if n_clusters > n_samples:
        raise ValueError("The number of clusters cannot exceed the number of samples.")

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    first = int(torch.randint(n_samples, (1,), generator=generator, device=x.device).item())
    selected = [first]
    centers = [x[first]]

    # The first additional center is the sample farthest from the first center.
    first_distances = _squared_distances(x, centers[0].unsqueeze(0)).flatten()
    first_distances[first] = -1.0
    second = int(torch.argmax(first_distances).item())
    selected.append(second)
    centers.append(x[second])

    # Subsequent centers are farthest from the midpoint (mean) of centers chosen so far.
    while len(centers) < n_clusters:
        midpoint = torch.stack(centers).mean(dim=0, keepdim=True)
        farthest_distance = torch.full(
            (n_samples,), -torch.inf, dtype=x.dtype, device=x.device
        )
        for start, end, distances in _distance_blocks(x, midpoint, block_size):
            farthest_distance[start:end] = distances[:, 0]
        farthest_distance[selected] = -torch.inf
        next_index = int(torch.argmax(farthest_distance).item())
        selected.append(next_index)
        centers.append(x[next_index])

    return torch.stack(centers)


def _initial_memberships(
    x: torch.Tensor, centers: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Use normalized exp(-Euclidean distance) weights as in the paper."""
    memberships = torch.empty(
        (x.shape[0], centers.shape[0]), dtype=x.dtype, device=x.device
    )
    for start, end, squared in _distance_blocks(x, centers, block_size):
        distances = squared.clamp_min(0.0).sqrt()
        memberships[start:end] = torch.softmax(-distances, dim=1)
    return memberships


def _memberships_from_squared_distances(
    squared: torch.Tensor, fuzzifier: float
) -> torch.Tensor:
    # For standard FCM, u_ij is proportional to (||x_i-c_j||^2)^(-1/(m-1)).
    zero = squared <= 1e-12
    has_zero = zero.any(dim=1, keepdim=True)
    weights = squared.clamp_min(1e-12).pow(-1.0 / (fuzzifier - 1.0))
    normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-20)
    zero_memberships = zero.to(squared.dtype) / zero.sum(dim=1, keepdim=True).clamp_min(1)
    return torch.where(has_zero, zero_memberships, normalized)


def fuzzy_c_means(
    x: torch.Tensor,
    n_clusters: int,
    *,
    fuzzifier: float = 2.0,
    max_iter: int = 50,
    tolerance: float = 1e-4,
    seed: int = 42,
    block_size: int = 65_536,
) -> FCMResult:
    """Run full-data FCM with the paper's distance-based two-stage initialization."""
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("FCM input must be a non-empty [n_samples, n_features] tensor.")
    if fuzzifier <= 1.0:
        raise ValueError("The FCM fuzzifier must be greater than 1.")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive.")

    centers = two_stage_initial_centers(x, n_clusters, seed, block_size)
    memberships = _initial_memberships(x, centers, block_size)
    iterations = 0

    for iteration in range(1, max_iter + 1):
        powered = memberships.pow(fuzzifier)
        centers = (powered.T @ x) / powered.sum(dim=0).clamp_min(1e-12).unsqueeze(1)

        updated = torch.empty_like(memberships)
        for start, end, squared in _distance_blocks(x, centers, block_size):
            updated[start:end] = _memberships_from_squared_distances(squared, fuzzifier)

        change = (updated - memberships).abs().max().item()
        memberships = updated
        iterations = iteration
        if change < tolerance:
            break

    # Align the returned centers with the final membership matrix.
    powered = memberships.pow(fuzzifier)
    centers = (powered.T @ x) / powered.sum(dim=0).clamp_min(1e-12).unsqueeze(1)
    objective_total = torch.zeros((), dtype=x.dtype, device=x.device)
    for start, end, squared in _distance_blocks(x, centers, block_size):
        objective_total += (memberships[start:end].pow(fuzzifier) * squared).sum()

    labels = memberships.argmax(dim=1)
    return FCMResult(
        centers=centers,
        memberships=memberships,
        labels=labels,
        iterations=iterations,
        objective=float(objective_total.item()),
    )
