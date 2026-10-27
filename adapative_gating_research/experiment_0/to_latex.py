"""Print a LaTeX timing table from Experiment 0 summary.json files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DEFAULT_RESULTS = Path(__file__).resolve().parent / "runs" / "bench_5000_e20_i10"
DATASET_NAMES = {
    "iris": "Iris",
    "wine": "Wine",
    "covertype": "Covertype",
    "fashion_mnist": "Fashion-MNIST",
}


def mean_sd(values: list[float], decimals: int) -> str:
    spread = stdev(values) if len(values) > 1 else 0.0
    return f"${mean(values):.{decimals}f} \\pm {spread:.{decimals}f}$"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="?", type=Path, default=DEFAULT_RESULTS,
        help="Directory containing dataset/seed-*/summary.json files",
    )
    args = parser.parse_args()

    summaries = sorted(args.results.glob("*/seed-*/summary.json"))
    if not summaries:
        parser.error(f"No summary.json files found under {args.results}")

    by_dataset: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for path in summaries:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        by_dataset[(result["dataset"], result["reference_size"])].append(result)

    print(r"\begin{tabular}{lrrrr}")
    print(r"\hline")
    print(r"Dataset & Silhouette points & Silhouette (s) & Fit (s) & Share of fit (\%) \\")
    print(r"\hline")
    dataset_order = {name: index for index, name in enumerate(DATASET_NAMES)}
    for (dataset, reference_size), runs in sorted(
        by_dataset.items(), key=lambda item: (dataset_order.get(item[0][0], len(dataset_order)), item[0])
    ):
        name = DATASET_NAMES.get(dataset, dataset.replace("_", r"\_"))
        silhouette = mean_sd([r["timing_seconds"]["silhouette"] for r in runs], 4)
        fit = mean_sd([r["timing_seconds"]["fit_wall"] for r in runs], 2)
        share = mean_sd([100 * r["silhouette_share_of_fit"] for r in runs], 2)
        print(f"{name} & {reference_size:,} & {silhouette} & {fit} & {share} " + r"\\")
    print(r"\hline")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
