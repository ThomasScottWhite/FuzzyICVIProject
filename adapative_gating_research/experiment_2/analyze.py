"""Compare the Experiment 2 gamma sweep at epochs 200 and 1000."""

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

RUNS = Path(__file__).resolve().parent / "runs" / "gamma_sweep_1000"
GUIDED = tuple(condition for condition in CONDITIONS if condition != "no_guidance")
GAMMAS = (0.001, 0.01, 0.05, 0.1)
CHECKPOINTS = (200, 1000)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(root: Path) -> dict:
    runs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in sorted(root.glob("*/*/gamma-*/seed-*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        with (path.parent / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        dataset, seed, condition = summary["dataset"], int(summary["seed"]), summary["condition"]
        gamma = float(summary["base_gamma"])
        runs[dataset][seed][condition][gamma] = (summary, metrics)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = load_runs(args.input)
    if not runs:
        parser.error(f"No completed gamma-sweep runs under {args.input}")
    output = args.output or args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    paired: list[dict] = []
    incomplete: list[dict] = []
    for dataset, by_seed in sorted(runs.items()):
        for seed, by_condition in sorted(by_seed.items()):
            control = by_condition.get("no_guidance", {}).get(0.0)
            for gamma in GAMMAS:
                fixed = by_condition.get("fixed", {}).get(gamma)
                for epoch in CHECKPOINTS:
                    missing = [condition for condition in GUIDED if gamma not in by_condition.get(condition, {})]
                    if control is None:
                        missing.append("no_guidance")
                    if missing:
                        incomplete.append({"dataset": dataset, "seed": seed, "gamma": gamma,
                                           "epoch": epoch, "missing_conditions": ";".join(missing)})
                    if control is None or fixed is None:
                        continue
                    fixed_checkpoint = fixed[0]["checkpoints"].get(str(epoch))
                    control_checkpoint = control[0]["checkpoints"].get(str(epoch))
                    if fixed_checkpoint is None or control_checkpoint is None:
                        continue
                    fixed_ari = fixed_checkpoint["external_metrics"]["ARI"]
                    control_ari = control_checkpoint["external_metrics"]["ARI"]
                    for condition in CONDITIONS:
                        item = control if condition == "no_guidance" else by_condition.get(condition, {}).get(gamma)
                        if item is None:
                            continue
                        checkpoint = item[0]["checkpoints"].get(str(epoch))
                        if checkpoint is None:
                            continue
                        scores = checkpoint["external_metrics"]
                        paired.append({
                            "dataset": dataset, "seed": seed, "gamma": gamma, "epoch": epoch,
                            "condition": condition, "ARI": scores["ARI"], "NMI": scores["NMI"],
                            "ACC": scores["ACC"], "silhouette": checkpoint["silhouette"],
                            "delta_ARI_vs_fixed": scores["ARI"] - fixed_ari,
                            "delta_ARI_vs_no_guidance": scores["ARI"] - control_ari,
                            "gate_openings": checkpoint["gate_openings"],
                            "guided_epochs": checkpoint["guided_epochs"],
                            "first_guided_epoch": checkpoint.get("first_guided_epoch"),
                            "cumulative_effective_gamma": checkpoint["cumulative_effective_gamma"],
                        })
    write_csv(output / "paired.csv", paired)
    write_csv(output / "incomplete.csv", incomplete)

    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["dataset"], row["gamma"], row["epoch"], row["condition"])].append(row)
    aggregate: list[dict] = []
    for (dataset, gamma, epoch, condition), rows in sorted(grouped.items()):
        delta = [row["delta_ARI_vs_fixed"] for row in rows]
        aggregate.append({
            "dataset": dataset, "gamma": gamma, "epoch": epoch, "condition": condition,
            "paired_seeds": len(rows), "mean_ARI": mean(row["ARI"] for row in rows),
            "mean_NMI": mean(row["NMI"] for row in rows),
            "mean_delta_ARI_vs_fixed": mean(delta),
            "sd_delta_ARI_vs_fixed": stdev(delta) if len(delta) > 1 else 0.0,
            "mean_delta_ARI_vs_no_guidance": mean(row["delta_ARI_vs_no_guidance"] for row in rows),
            "mean_guided_epochs": mean(row["guided_epochs"] for row in rows),
            "mean_cumulative_effective_gamma": mean(row["cumulative_effective_gamma"] for row in rows),
        })
    write_csv(output / "summary.csv", aggregate)

    for dataset in sorted(runs):
        for epoch in CHECKPOINTS:
            fig, ax = plt.subplots(figsize=(9, 5))
            plotted = False
            for condition in GUIDED:
                points = sorted((row["gamma"], row["mean_delta_ARI_vs_fixed"])
                                for row in aggregate if row["dataset"] == dataset
                                and row["epoch"] == epoch and row["condition"] == condition)
                if points:
                    ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=condition)
                    plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set(xlabel="Base guidance gamma", ylabel="Mean paired ARI minus fixed gate",
                   title=f"{dataset}: epoch {epoch}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(output / f"{dataset}_epoch-{epoch}_gamma.png", dpi=160)
            plt.close(fig)

        for gamma in GAMMAS:
            fig, ax = plt.subplots(figsize=(10, 5))
            plotted = False
            for condition in CONDITIONS:
                paths = defaultdict(list)
                n_runs = 0
                for by_condition in runs[dataset].values():
                    item = by_condition.get(condition, {}).get(0.0 if condition == "no_guidance" else gamma)
                    if item is None:
                        continue
                    n_runs += 1
                    for metric in item[1]:
                        if metric["silhouette"]:
                            paths[int(metric["epoch"])].append(float(metric["silhouette"]))
                if paths:
                    epochs = sorted(paths)
                    ax.plot(epochs, [mean(paths[epoch]) for epoch in epochs],
                            label=f"{condition} (n={n_runs})")
                    plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.axvline(200, color="gray", linestyle="--", linewidth=1)
            ax.set(xlabel="Epoch", ylabel="Mean reference silhouette",
                   title=f"{dataset}: gamma {gamma:g}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(output / f"{dataset}_gamma-{gamma:g}_silhouette.png", dpi=160)
            plt.close(fig)

    print(f"Analyzed {len(paired)} paired rows; results in {output}")


if __name__ == "__main__":
    main()
