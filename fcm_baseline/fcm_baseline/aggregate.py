"""Aggregate per-dataset result CSVs into summary tables.

Reads ``results_<dataset>.csv`` files from a results directory and writes

* ``summary.csv`` - mean and standard deviation (over seeds) of ACC / NMI / ARI /
  silhouette for each dataset, method and space;
* ``summary.md``  - the same as a Markdown table, followed by the DMSC no-guidance
  reference numbers from Table I of the paper for side-by-side comparison.

Usage:  python -m fcm_baseline.aggregate --results-dir results
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

#: DMSC no-guidance results at epoch 200 (mean +- s.d. over seeds 42-46)
DMSC_NO_GUIDANCE = {
    "iris":          dict(acc="0.905 ± 0.039", nmi="0.769 ± 0.069", ari="0.759 ± 0.084"),
    "wine":          dict(acc="0.925 ± 0.015", nmi="0.763 ± 0.038", ari="0.780 ± 0.041"),
    "covertype":     dict(acc="0.310 ± 0.029", nmi="0.128 ± 0.029", ari="0.062 ± 0.016"),
    "fashion_mnist": dict(acc="0.499 ± 0.029", nmi="0.509 ± 0.027", ari="0.337 ± 0.025"),
}

METRICS = ["acc", "nmi", "ari", "silhouette"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "results_*.csv")))
    if not files:
        raise SystemExit(f"No results_*.csv files found in {args.results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    grp = df.groupby(["dataset", "method", "space"])
    summary = grp[METRICS + ["n_iter", "fit_seconds"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.insert(0, "n_seeds", grp.size())
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)

    # Markdown rendering with "mean ± std" cells.
    lines = ["# Independent baseline results (mean ± s.d. over seeds)\n",
             "| Dataset | Method | Space | Seeds | ACC | NMI | ARI |",
             "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        cells = [f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}" for m in ("acc", "nmi", "ari")]
        lines.append(f"| {r['dataset']} | {r['method']} | {r['space']} | {int(r['n_seeds'])} | "
                     + " | ".join(cells) + " |")
    lines += ["", "## DMSC no-guidance reference (paper Table I, epoch 200)\n",
              "| Dataset | ACC | NMI | ARI |", "|---|---|---|---|"]
    for ds, v in DMSC_NO_GUIDANCE.items():
        lines.append(f"| {ds} | {v['acc']} | {v['nmi']} | {v['ari']} |")
    lines += ["", "_Silhouette values in `summary.csv` are computed in the clustered space "
              "(raw or PCA-3) and are not comparable to DMSC's latent-space silhouette._"]
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
