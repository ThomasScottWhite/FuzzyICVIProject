"""From-scratch Fuzzy C-Means (FCM) clustering.

Why write our own instead of using a library?
    The DMSC study uses FCM with a specific configuration (fuzzifier m = 2, at most 50
    iterations, stop when the largest membership change drops below 1e-4, and a
    two-stage distance-based initialisation).  Writing the algorithm explicitly lets the
    baseline use *exactly* that configuration, and keeps it independent of the PyTorch
    implementation inside ``dmsc_baseline`` so the two can be cross-checked.

Algorithm (Bezdek, Ehrlich & Full, 1984)
    Given data ``X`` (n x d), ``k`` clusters and fuzzifier ``m > 1``, FCM minimises

        J = sum_i sum_j  u_ij^m * ||x_i - c_j||^2        subject to  sum_j u_ij = 1

    by alternating:

        centres:      c_j  = sum_i u_ij^m x_i / sum_i u_ij^m
        memberships:  u_ij = 1 / sum_l ( ||x_i - c_j||^2 / ||x_i - c_l||^2 )^(1 / (m - 1))

    Hard labels are ``argmax_j u_ij``.

Memory
    Covertype (581,012 x 54) and Fashion-MNIST (70,000 x 784) are processed in row
    chunks so no n x k x d tensor is ever materialised; only the n x k membership matrix
    is kept in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Floor for squared distances.  A point lying exactly on a centre has distance 0, which
# would make the membership formula divide by zero; the floor keeps it finite and the
# membership of that point effectively collapses onto the coincident centre.
_EPS = 1e-12


def _squared_distances(Xc: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between each row of ``Xc`` and each centre.

    Uses the expansion ``||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2`` (one matrix product)
    and clips tiny negative values caused by floating-point cancellation.

    Returns an array of shape ``(len(Xc), len(centers))`` in float64.
    """
    Xc = Xc.astype(np.float64, copy=False)
    d2 = (
        (Xc * Xc).sum(axis=1, keepdims=True)
        - 2.0 * Xc @ centers.T
        + (centers * centers).sum(axis=1)[None, :]
    )
    np.maximum(d2, 0.0, out=d2)
    return d2


def _memberships_from_d2(d2: np.ndarray, m: float) -> np.ndarray:
    """Convert squared distances to FCM memberships (rows sum to 1).

    With squared distances the exponent in the standard formula becomes ``1/(m-1)``:
    ``u_ij = (1 / d2_ij)^(1/(m-1)) / sum_l (1 / d2_il)^(1/(m-1))``.
    """
    inv = (1.0 / np.maximum(d2, _EPS)) ** (1.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


@dataclass
class FCMResult:
    """Outcome of one FCM fit.

    Attributes
    ----------
    centers : (k, d) float64   final cluster centres.
    memberships : (n, k) float64   final fuzzy membership matrix.
    labels : (n,) int64   hard assignment ``argmax`` of the memberships.
    n_iter : int   number of update iterations actually performed.
    converged : bool   True if the tolerance was met before ``max_iter``.
    objective : float   final value of the FCM objective ``J``.
    """

    centers: np.ndarray
    memberships: np.ndarray
    labels: np.ndarray
    n_iter: int
    converged: bool
    objective: float


class FuzzyCMeans:
    """Fuzzy C-Means with the configuration used by the DMSC reimplementation.

    Parameters
    ----------
    n_clusters : int
        Number of clusters ``k``.
    m : float, default 2.0
        Fuzzifier (``m > 1``).  The paper uses 2.
    max_iter : int, default 50
        Maximum number of centre/membership updates (paper: 50).
    tol : float, default 1e-4
        Stop when ``max_ij |u_new - u_old| < tol`` (tolerance is an implementation
        choice of the DMSC reimplementation).
    init : {"two_stage", "random"}, default "two_stage"
        Centre initialisation (see :meth:`_init_centers`).
    seed : int, default 0
        Seed for the initialisation RNG.  Fixing it makes runs reproducible.
    chunk_size : int, default 50_000
        Rows processed at a time; lower it if memory is tight.
    """

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        init: str = "two_stage",
        seed: int = 0,
        chunk_size: int = 50_000,
    ) -> None:
        if m <= 1.0:
            raise ValueError("Fuzzifier m must be > 1")
        if init not in ("two_stage", "random"):
            raise ValueError("init must be 'two_stage' or 'random'")
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.seed = seed
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------ helpers
    def _all_memberships(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute the full (n, k) membership matrix chunk by chunk."""
        n = X.shape[0]
        U = np.empty((n, self.n_clusters), dtype=np.float64)
        for s in range(0, n, self.chunk_size):
            e = min(s + self.chunk_size, n)
            U[s:e] = _memberships_from_d2(_squared_distances(X[s:e], centers), self.m)
        return U

    def _update_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Weighted-mean centre update ``c_j = sum u_ij^m x_i / sum u_ij^m`` (chunked)."""
        k, d = self.n_clusters, X.shape[1]
        num = np.zeros((k, d), dtype=np.float64)
        den = np.zeros(k, dtype=np.float64)
        for s in range(0, X.shape[0], self.chunk_size):
            e = min(s + self.chunk_size, X.shape[0])
            W = U[s:e] ** self.m                       # (chunk, k)
            num += W.T @ X[s:e].astype(np.float64)     # (k, d)
            den += W.sum(axis=0)
        return num / np.maximum(den, _EPS)[:, None]

    def _objective(self, X: np.ndarray, U: np.ndarray, centers: np.ndarray) -> float:
        """FCM objective ``J = sum_ij u_ij^m ||x_i - c_j||^2`` (chunked)."""
        total = 0.0
        for s in range(0, X.shape[0], self.chunk_size):
            e = min(s + self.chunk_size, X.shape[0])
            total += float(((U[s:e] ** self.m) * _squared_distances(X[s:e], centers)).sum())
        return total

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Choose starting centres.

        ``"random"``
            ``k`` distinct data points chosen uniformly at random.

        ``"two_stage"`` (default)
            A distance-based, two-stage scheme in the spirit of the DMSC paper:

            1. *Spread-out seeding* - pick the first centre uniformly at random, then pick
               each further centre with probability proportional to its squared distance
               to the nearest centre already chosen (k-means++ style).
            2. *Refinement* - assign every point to its nearest seed (squared Euclidean
               distance) and replace each seed by the mean of its points.

            .. warning::
               The DMSC paper describes its initialisation only briefly ("two-stage,
               distance-based").  This is **our interpretation**, not a verified copy of
               the paper's procedure.  Compare against the FCM routine in
               ``dmsc_baseline`` before claiming the two initialisations are identical;
               use ``--init random`` to remove the question entirely.
        """
        n, k = X.shape[0], self.n_clusters
        if self.init == "random":
            return X[rng.choice(n, size=k, replace=False)].astype(np.float64)

        # Stage 1: k-means++-style spread-out seeding.
        centers = np.empty((k, X.shape[1]), dtype=np.float64)
        centers[0] = X[rng.integers(n)]
        closest_d2 = _squared_distances(X, centers[:1])[:, 0]
        for j in range(1, k):
            total = closest_d2.sum()
            if total <= 0:  # all remaining points coincide with chosen seeds
                idx = rng.integers(n)
            else:
                idx = rng.choice(n, p=closest_d2 / total)
            centers[j] = X[idx]
            closest_d2 = np.minimum(closest_d2, _squared_distances(X, centers[j : j + 1])[:, 0])

        # Stage 2: one hard-assignment refinement (replace seeds by cluster means).
        assign = np.empty(n, dtype=np.int64)
        for s in range(0, n, self.chunk_size):
            e = min(s + self.chunk_size, n)
            assign[s:e] = _squared_distances(X[s:e], centers).argmin(axis=1)
        for j in range(k):
            members = assign == j
            if members.any():  # keep the seed if a cluster came out empty
                centers[j] = X[members].astype(np.float64).mean(axis=0)
        return centers

    # ---------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray) -> FCMResult:
        """Run FCM on ``X`` (shape ``(n, d)``) and return an :class:`FCMResult`."""
        rng = np.random.default_rng(self.seed)
        centers = self._init_centers(X, rng)
        U = self._all_memberships(X, centers)

        converged, n_iter = False, 0
        for n_iter in range(1, self.max_iter + 1):
            centers = self._update_centers(X, U)
            U_new = self._all_memberships(X, centers)
            delta = float(np.abs(U_new - U).max())  # largest membership change
            U = U_new
            if delta < self.tol:
                converged = True
                break

        return FCMResult(
            centers=centers,
            memberships=U,
            labels=U.argmax(axis=1).astype(np.int64),
            n_iter=n_iter,
            converged=converged,
            objective=self._objective(X, U, centers),
        )
