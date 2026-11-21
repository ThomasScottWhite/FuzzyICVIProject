"""Evaluation metrics, matching the DMSC study's reporting protocol.

* **ACC** - clustering accuracy after optimal one-to-one matching of predicted clusters
  to classes (Hungarian algorithm).
* **NMI** - normalised mutual information (scikit-learn default, arithmetic average).
* **ARI** - adjusted Rand index.
* **Silhouette** - exact silhouette coefficient with Euclidean distance on a fixed
  evaluation subset (all points for Iris/Wine; 5,000 fixed points for Covertype and
  Fashion-MNIST).  It is reported as a *diagnostic only* - it is computed in whichever
  space was clustered (raw features or 3-D PCA), so it is **not comparable** with the
  silhouette DMSC computes in its learned 3-D latent space.

ACC, NMI and ARI are always computed on the *full* dataset, as in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy under the best cluster-to-class assignment.

    Builds the contingency table between predicted clusters and true classes, finds the
    one-to-one matching with the most agreements (Hungarian algorithm), and returns the
    matched fraction.  Works when the number of clusters differs from the number of
    classes (unmatched clusters simply count as errors).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, t_idx = np.unique(y_true, return_inverse=True)
    clusters, p_idx = np.unique(y_pred, return_inverse=True)
    table = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    np.add.at(table, (p_idx, t_idx), 1)
    rows, cols = linear_sum_assignment(-table)  # negate: maximise matches
    return float(table[rows, cols].sum() / len(y_true))


def reference_silhouette(
    X: np.ndarray, labels: np.ndarray, subset_idx: np.ndarray
) -> float:
    """Exact Euclidean silhouette on a subset of points.

    As in the DMSC study, the labels come from clustering the *full* data, while the
    distances and cluster averages are computed *within the subset*.  Returns ``nan``
    when fewer than two clusters are represented in the subset (silhouette undefined).
    """
    sub_labels = labels[subset_idx]
    if len(np.unique(sub_labels)) < 2:
        return float("nan")
    return float(silhouette_score(X[subset_idx], sub_labels, metric="euclidean"))


def evaluate(
    X_clustered: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    subset_idx: np.ndarray,
) -> dict:
    """Compute ACC, NMI, ARI (full data) and the reference silhouette (subset)."""
    return {
        "acc": clustering_accuracy(y_true, labels),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "ari": float(adjusted_rand_score(y_true, labels)),
        "silhouette": reference_silhouette(X_clustered, labels, subset_idx),
    }
