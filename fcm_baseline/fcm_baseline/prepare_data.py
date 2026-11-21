"""Download and cache all four datasets.

Run this ONCE on a machine with internet access (on most HPC systems: the login node)
before submitting batch jobs, because compute nodes are often offline.  Iris and Wine ship
with scikit-learn; Covertype (~75 MB) and Fashion-MNIST (~30 MB) are downloaded.

Usage:  python -m fcm_baseline.prepare_data --data-dir data_cache
"""

from __future__ import annotations

import argparse

from .data import DATASET_NAMES, load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data_cache")
    args = ap.parse_args()
    for name in DATASET_NAMES:
        ds = load_dataset(name, args.data_dir)
        print(f"{name:14s} X={ds.X.shape}  classes={ds.n_clusters}  "
              f"range=[{ds.X.min():.2f}, {ds.X.max():.2f}]")


if __name__ == "__main__":
    main()
