"""Independent clustering baselines for the DMSC silhouette-gate study.

This package clusters the *same four datasets* used in the DMSC reimplementation
(Iris, Wine, Covertype, Fashion-MNIST) with plain Fuzzy C-Means (FCM) and K-means,
using **no autoencoder, no silhouette gate, and no guidance of any kind**.  It is
deliberately independent of the ``dmsc_baseline`` / ``dmsc_custom`` packages so that
it can serve as an external reference point: if DMSC (with or without guidance)
does not beat these numbers, the learned representation itself is not buying
anything on that dataset.

Modules
-------
data        Dataset loading, caching and preprocessing (matches the paper's Section VI-B).
fcm         A from-scratch, memory-chunked Fuzzy C-Means implementation (NumPy).
metrics     ACC (Hungarian matching), NMI, ARI, and the 5,000-point reference silhouette.
run_baseline  Command-line runner: fits every (dataset, method, space, seed) combination.
aggregate   Collects per-dataset result files into summary tables.
prepare_data  Downloads/caches datasets (run once on a node with internet access).
"""

__version__ = "1.0.0"
