"""Summarize paired Experiment 2 runs and plot final ARI and silhouette paths."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .policies import CONDITIONS

RUNS = Path(__file__).resolve().parent / "runs" / "paired_v1"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs: dict[str, dict[int, dict[str, tuple[dict, list[dict]]]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(args.input.glob("*/*/seed-*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        with (path.parent / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        runs[summary["dataset"]][int(summary["seed"])][summary["condition"]] = (summary, metrics)
    if not runs:
        parser.error(f"No completed runs under {args.input}")
    output = args.output or args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    paired: list[dict] = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    incomplete: list[dict] = []
    for dataset, by_seed in sorted(runs.items()):
        for seed, by_condition in sorted(by_seed.items()):
            missing = sorted(set(CONDITIONS) - set(by_condition))
            if missing:
                incomplete.append({"dataset": dataset, "seed": seed, "missing_conditions": ";".join(missing)})
            if "fixed" not in by_condition or "no_guidance" not in by_condition:
                continue
            fixed = by_condition["fixed"][0]["final_metrics"]
            control = by_condition["no_guidance"][0]["final_metrics"]
            for condition, (summary, _) in sorted(by_condition.items()):
                final = summary["final_metrics"]
                row = {
                    "dataset": dataset, "seed": seed, "condition": condition,
                    "ARI": final["ARI"], "NMI": final["NMI"], "ACC": final["ACC"],
                    "delta_ARI_vs_fixed": final["ARI"] - fixed["ARI"],
                    "delta_ARI_vs_no_guidance": final["ARI"] - control["ARI"],
                    "delta_NMI_vs_fixed": final["NMI"] - fixed["NMI"],
                    "gate_openings": summary["gate_openings"],
                    "guided_epochs": summary["guided_epochs"],
                    "first_guided_epoch": summary["first_guided_epoch"],
                    "cumulative_effective_gamma": summary["cumulative_effective_gamma"],
                    "wall_seconds": summary["wall_seconds"],
                }
                paired.append(row)
                grouped[(dataset, condition)].append(row)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "incomplete.csv", incomplete)

    aggregate: list[dict] = []
    for (dataset, condition), rows in sorted(grouped.items()):
        deltas = [row["delta_ARI_vs_fixed"] for row in rows]
        aggregate.append({
            "dataset": dataset, "condition": condition, "paired_seeds": len(rows),
            "mean_ARI": mean(row["ARI"] for row in rows),
            "mean_NMI": mean(row["NMI"] for row in rows),
            "mean_delta_ARI_vs_fixed": mean(deltas),
            "sd_delta_ARI_vs_fixed": stdev(deltas) if len(deltas) > 1 else 0.0,
            "mean_delta_ARI_vs_no_guidance": mean(row["delta_ARI_vs_no_guidance"] for row in rows),
            "mean_guided_epochs": mean(row["guided_epochs"] for row in rows),
            "mean_cumulative_effective_gamma": mean(row["cumulative_effective_gamma"] for row in rows),
        })
    write_csv(output / "summary.csv", aggregate)

    for dataset, by_seed in sorted(runs.items()):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for condition in CONDITIONS:
            condition_runs = [by_condition[condition] for by_condition in by_seed.values()
                              if condition in by_condition]
            if not condition_runs:
                continue
            final_ari = [summary["final_metrics"]["ARI"] for summary, _ in condition_runs]
            axes[0].scatter([condition] * len(final_ari), final_ari, alpha=0.5, s=24)
            axes[0].plot([condition], [mean(final_ari)], marker="D", markersize=7)
            paths = defaultdict(list)
            for _, metrics in condition_runs:
                for row in metrics:
                    paths[int(row["epoch"])].append(float(row["silhouette"]))
            epochs = sorted(paths)
            axes[1].plot(epochs, [mean(paths[epoch]) for epoch in epochs],
                         label=f"{condition} (n={len(condition_runs)})")
        axes[0].set(ylabel="Final ARI", title=f"{dataset}: final ARI by method")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].set(xlabel="Epoch", ylabel="Reference silhouette")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(output / f"{dataset}.png", dpi=160)
        plt.close(fig)
    print(f"Analyzed {len(paired)} paired condition runs; results in {output}")


if __name__ == "__main__":
    main()
