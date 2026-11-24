"""Command-line runner for the independent FCM / K-means baselines.

For every combination of  dataset x method x space x seed  this script

1. loads and preprocesses the dataset (see :mod:`fcm_baseline.data`),
2. optionally projects it to 3-D with PCA,
3. clusters it with FCM or K-means (no autoencoder, no gate, no guidance),
4. computes ACC / NMI / ARI on the full data and the reference silhouette on the fixed
   evaluation subset,
5. appends one row to ``<output-dir>/results_<dataset>.csv`` and saves the final hard
   assignments to ``<output-dir>/labels/``.

Methods
-------
``fcm``     Fuzzy C-Means (our implementation; m = 2, <= 50 iterations, tol 1e-4).
``kmeans``  scikit-learn K-means (k-means++ init, 10 restarts) - a hard-clustering anchor.

Spaces
------
``raw``   Cluster the preprocessed input features directly (the natural "independent"
          baseline: no learned representation at all).
``pca3``  Cluster a 3-D PCA projection.  DMSC clusters a 3-D autoencoder latent space, so
          this isolates the effect of *learning* the 3-D representation from the effect of
          merely having low dimension.

Each run writes its own row and labels file, and finished runs are skipped on restart
(``--no-resume`` to force recomputation), so jobs can be re-submitted safely after a
time-out or pre-emption.

Example
-------
    python -m fcm_baseline.run_baseline --datasets iris wine --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import platform
import time
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .data import DATASET_NAMES, evaluation_subset_indices, load_dataset
from .fcm import FuzzyCMeans
from .metrics import evaluate

log = logging.getLogger("fcm_baseline")

#: Column order of the per-dataset result CSV files.
CSV_FIELDS = [
    "dataset", "method", "space", "seed", "n_samples", "n_features_clustered",
    "n_clusters", "acc", "nmi", "ari", "silhouette", "n_iter", "converged",
    "objective", "fit_seconds", "init", "fuzzifier", "max_iter", "tol",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES), choices=DATASET_NAMES)
    p.add_argument("--methods", nargs="+", default=["fcm", "kmeans"], choices=["fcm", "kmeans"])
    p.add_argument("--spaces", nargs="+", default=["raw", "pca3"], choices=["raw", "pca3"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                   help="Seeds for the Experiment-1 comparison (paper: 42-46).")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--data-dir", default="data_cache", help="Dataset cache directory.")
    # FCM configuration (defaults = DMSC reimplementation, Section VI-A).
    p.add_argument("--fuzzifier", type=float, default=2.0)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--init", default="two_stage", choices=["two_stage", "random"])
    p.add_argument("--chunk-size", type=int, default=50_000)
    # Evaluation subset.
    p.add_argument("--subset-size", type=int, default=5000)
    p.add_argument("--subset-seed", type=int, default=2026)
    p.add_argument("--subset-indices", default=None,
                   help="Optional .npy file of subset indices (overrides --subset-size/seed). "
                        "Only valid when running a single large dataset.")
    p.add_argument("--no-resume", action="store_true", help="Recompute runs that already exist.")
    return p.parse_args(argv)


def project(X: np.ndarray, space: str, seed: int) -> np.ndarray:
    """Return the representation to be clustered: raw features or a 3-D PCA projection."""
    if space == "raw":
        return X
    # PCA is deterministic up to sign for the full solver; the randomized solver (used
    # automatically for large inputs) is seeded for reproducibility.
    return PCA(n_components=3, random_state=seed).fit_transform(X).astype(np.float32)


def cluster(X: np.ndarray, k: int, method: str, seed: int, args: argparse.Namespace) -> Dict:
    """Fit one clustering model; return labels plus fit diagnostics."""
    t0 = time.perf_counter()
    if method == "fcm":
        res = FuzzyCMeans(
            n_clusters=k, m=args.fuzzifier, max_iter=args.max_iter, tol=args.tol,
            init=args.init, seed=seed, chunk_size=args.chunk_size,
        ).fit(X)
        out = dict(labels=res.labels, n_iter=res.n_iter, converged=res.converged,
                   objective=res.objective, init=args.init)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        out = dict(labels=km.labels_.astype(np.int64), n_iter=int(km.n_iter_), converged=True,
                   objective=float(km.inertia_), init="k-means++")
    out["fit_seconds"] = time.perf_counter() - t0
    return out


def _completed(csv_path: str) -> set:
    """Keys (method, space, seed) already present in a result file."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {(r["method"], r["space"], int(r["seed"])) for r in csv.DictReader(f)}


def _append_row(csv_path: str, row: Dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(os.path.join(args.output_dir, "labels"), exist_ok=True)
    log.info("Host %s | Python %s | args %s", platform.node(), platform.python_version(), vars(args))

    for name in args.datasets:
        ds = load_dataset(name, args.data_dir)
        log.info("[%s] n=%d d=%d k=%d", name, ds.X.shape[0], ds.X.shape[1], ds.n_clusters)

        if args.subset_indices:
            subset_idx = np.load(args.subset_indices)
        else:
            subset_idx = evaluation_subset_indices(ds.X.shape[0], args.subset_size, args.subset_seed)

        csv_path = os.path.join(args.output_dir, f"results_{name}.csv")
        done = set() if args.no_resume else _completed(csv_path)

        for space in args.spaces:
            for seed in args.seeds:
                # Project once per (space, seed); reused by every method below.
                todo = [m for m in args.methods if (m, space, seed) not in done]
                if not todo:
                    continue
                Xc = project(ds.X, space, seed)
                for method in todo:
                    out = cluster(Xc, ds.n_clusters, method, seed, args)
                    scores = evaluate(Xc, ds.y, out["labels"], subset_idx)
                    np.save(os.path.join(args.output_dir, "labels",
                                         f"{name}_{method}_{space}_seed{seed}.npy"),
                            out["labels"].astype(np.int16))
                    _append_row(csv_path, dict(
                        dataset=name, method=method, space=space, seed=seed,
                        n_samples=Xc.shape[0], n_features_clustered=Xc.shape[1],
                        n_clusters=ds.n_clusters, n_iter=out["n_iter"],
                        converged=out["converged"], objective=out["objective"],
                        fit_seconds=round(out["fit_seconds"], 3), init=out["init"],
                        fuzzifier=args.fuzzifier if method == "fcm" else "",
                        max_iter=args.max_iter if method == "fcm" else "",
                        tol=args.tol if method == "fcm" else "", **scores,
                    ))
                    log.info("[%s] %-6s %-4s seed=%d  ACC=%.3f NMI=%.3f ARI=%.3f  (%d it, %.1fs)",
                             name, method, space, seed, scores["acc"], scores["nmi"],
                             scores["ari"], out["n_iter"], out["fit_seconds"])


if __name__ == "__main__":
    main()
