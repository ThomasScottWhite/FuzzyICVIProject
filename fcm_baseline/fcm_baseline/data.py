"""Dataset loading and preprocessing.

The four datasets and their preprocessing follow Section VI-B of the project paper:

==============  =========  ========  =========  ==========================================
Dataset         Samples    Features  Classes    Preprocessing
==============  =========  ========  =========  ==========================================
Iris            150        4         3          min-max scale every feature to [0, 1]
Wine            178        13        3          min-max scale every feature to [0, 1]
Covertype       581,012    54        7          min-max scale the 10 continuous features;
                                                keep the 44 binary indicators unchanged
Fashion-MNIST   70,000     784       10         divide pixels by 255 (train + test combined)
==============  =========  ========  =========  ==========================================

Class labels are returned **only for post-hoc evaluation** (ACC / NMI / ARI).  They are
never used for fitting, initialisation or model selection.

Downloads
---------
Covertype and Fashion-MNIST are downloaded through scikit-learn on first use and cached
under ``data_dir``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml, load_iris, load_wine

#: Names accepted by :func:`load_dataset`.
DATASET_NAMES = ("iris", "wine", "covertype", "fashion_mnist")

#: Seed used (once) to draw the fixed evaluation subset for the silhouette on the two
#: large datasets.
EVAL_SUBSET_SEED = 2026

#: Size of the fixed evaluation subset for large datasets.
EVAL_SUBSET_SIZE = 5000

#: Covertype has 54 columns: the first 10 are continuous, the remaining 44 are binary
#: indicators (4 wilderness-area + 40 soil-type columns).
_COVERTYPE_N_CONTINUOUS = 10


@dataclass
class Dataset:
    """A preprocessed dataset ready for clustering.

    Attributes
    ----------
    name : str
        Canonical dataset name (one of :data:`DATASET_NAMES`).
    X : np.ndarray, shape (n_samples, n_features), float32
        Preprocessed feature matrix (see module docstring).
    y : np.ndarray, shape (n_samples,), int64
        Ground-truth class labels remapped to ``0..k-1``.  Evaluation only.
    n_clusters : int
        Number of clusters ``k`` to request.  Following the paper's benchmark convention
        this equals the number of ground-truth classes.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    n_clusters: int


def _minmax(X: np.ndarray) -> np.ndarray:
    """Min-max scale each column to [0, 1]; constant columns become all zeros."""
    X = X.astype(np.float64)
    lo = X.min(axis=0)
    span = X.max(axis=0) - lo
    span[span == 0] = 1.0  # avoid 0/0 for constant columns
    return (X - lo) / span


def _remap_labels(y: np.ndarray) -> np.ndarray:
    """Map arbitrary label values (ints or strings) onto 0..k-1."""
    _, inverse = np.unique(y, return_inverse=True)
    return inverse.astype(np.int64)


def load_dataset(name: str, data_dir: str = "data_cache") -> Dataset:
    """Load and preprocess one of the four study datasets.

    Parameters
    ----------
    name : str
        One of ``"iris"``, ``"wine"``, ``"covertype"``, ``"fashion_mnist"``.
    data_dir : str
        Directory used to cache downloaded data (only needed for the two large datasets).

    Returns
    -------
    Dataset
    """
    name = name.lower().replace("-", "_")
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    os.makedirs(data_dir, exist_ok=True)

    if name == "iris":
        bunch = load_iris()
        X, y = _minmax(bunch.data), bunch.target

    elif name == "wine":
        bunch = load_wine()
        X, y = _minmax(bunch.data), bunch.target

    elif name == "covertype":
        # shuffle=False keeps the canonical row order; labels are 1..7.
        bunch = fetch_covtype(data_home=data_dir, shuffle=False)
        X = bunch.data.astype(np.float64)
        # Scale only the continuous columns; binary indicators are already in {0, 1}.
        X[:, :_COVERTYPE_N_CONTINUOUS] = _minmax(X[:, :_COVERTYPE_N_CONTINUOUS])
        y = bunch.target

    else:  # fashion_mnist
        # OpenML "Fashion-MNIST" v1 is the full 70,000-image set (train + test).
        bunch = fetch_openml(
            "Fashion-MNIST", version=1, as_frame=False, parser="liac-arff", data_home=data_dir
        )
        X = bunch.data.astype(np.float64) / 255.0
        y = bunch.target

    y = _remap_labels(np.asarray(y))
    return Dataset(
        name=name,
        X=np.ascontiguousarray(X, dtype=np.float32),
        y=y,
        n_clusters=int(len(np.unique(y))),
    )


def evaluation_subset_indices(
    n_samples: int,
    size: int = EVAL_SUBSET_SIZE,
    seed: int = EVAL_SUBSET_SEED,
) -> np.ndarray:
    """Return the fixed point indices on which the reference silhouette is computed.

    Small datasets (``n_samples <= size``: Iris, Wine) use every point.  Large datasets
    use ``size`` points drawn once without replacement with a fixed seed, so the *same*
    subset is shared by every seed and method.
    """
    if n_samples <= size:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=size, replace=False))
