"""Graph Experiment 1 quality, gate behavior, and the gamma sweep.

Run from the repository root with::

    python -m adapative_gating_research.experiment_1.graphs

Only completed directories containing summary.json are read. Outputs go to
``runs/gamma_sweep_200/graphs``
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "runs" / "gamma_sweep_200"
METRICS = ("silhouette", "ARI", "NMI", "ACC")
TITLES = {
    "silhouette": "Reference silhouette",
    "ARI": "Adjusted Rand index",
    "NMI": "Normalized mutual information",
    "ACC": "Clustering accuracy",
}
COLORS = {"fixed": "#1675a6", "no_guidance": "#e18739"}
DATASET_ORDER = ("iris", "wine", "covertype", "fashion_mnist")


def is_true(value: str) -> bool:
    return value.lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class Run:
    gamma: float
    dataset: str
    seed: int
    summary: dict
    fixed: list[dict[str, str]]
    control: list[dict[str, str]]
    history: list[dict[str, str]]

    def final_delta(self, metric: str) -> float:
        results = self.summary["results"]
        if metric == "silhouette":
            return float(results["fixed"]["final_silhouette"]) - float(
                results["no_guidance"]["final_silhouette"]
            )
        return float(results["fixed"]["external_metrics"][metric]) - float(
            results["no_guidance"]["external_metrics"][metric]
        )

    def metric_epochs(self) -> list[int]:
        return [int(row["epoch"]) for row in self.fixed]

    def series(self, mode: str, metric: str) -> list[float]:
        rows = self.fixed if mode == "fixed" else self.control
        return [float(row[metric]) for row in rows]

    def delta_series(self, metric: str) -> list[float]:
        return [a - b for a, b in zip(self.series("fixed", metric),
                                      self.series("no_guidance", metric))]

    def gate_checks(self) -> list[dict[str, str]]:
        return [row for row in self.fixed if is_true(row["scheduled_evaluation"])]

    def openings(self) -> int:
        return sum(is_true(row["gate_opened"]) for row in self.history)

    def applied_epochs(self) -> int:
        return sum(is_true(row["guidance_applied"]) for row in self.history)


def load_runs(root: Path) -> list[Run]:
    paths = sorted(root.glob("gamma-*/*/seed-*/summary.json"))
    if not paths:
        raise ValueError(f"No completed runs found under {root}")
    runs = []
    seen = set()
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        gamma = float(summary["gamma"])
        dataset = summary["dataset"]
        seed = int(summary["seed"])
        if (gamma, dataset, seed) in seen:
            raise ValueError(f"Duplicate gamma/dataset/seed: {path}")
        seen.add((gamma, dataset, seed))
        if not np.isclose(gamma, float(path.parents[2].name.removeprefix("gamma-"))):
            raise ValueError(f"Gamma in summary differs from folder: {path}")
        fixed = read_csv(path.parent / "fixed_epoch_metrics.csv")
        control = read_csv(path.parent / "no_guidance_epoch_metrics.csv")
        history = read_csv(path.parent / "fixed_history.csv")
        if not fixed or [r["epoch"] for r in fixed] != [r["epoch"] for r in control]:
            raise ValueError(f"Metric epochs missing or unmatched: {path.parent}")
        if not history:
            raise ValueError(f"Empty fixed history: {path.parent}")
        runs.append(Run(gamma, dataset, seed, summary, fixed, control, history))
    return runs


def ordered_datasets(runs: list[Run]) -> list[str]:
    names = {run.dataset for run in runs}
    return sorted(names, key=lambda name: (
        DATASET_ORDER.index(name) if name in DATASET_ORDER else len(DATASET_ORDER), name
    ))


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}"


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_quality(runs: list[Run], path: Path) -> None:
    """One dataset/gamma: metric trajectories for both conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for mode in COLORS:
            values: dict[int, list[float]] = defaultdict(list)
            for run in runs:
                epochs = run.metric_epochs()
                scores = run.series(mode, metric)
                ax.plot(epochs, scores, color=COLORS[mode], alpha=0.16, lw=1)
                for epoch, score in zip(epochs, scores):
                    values[epoch].append(score)
            epochs = sorted(values)
            ax.plot(epochs, [mean(values[epoch]) for epoch in epochs],
                    color=COLORS[mode], lw=2.4,
                    label="Fixed gate" if mode == "fixed" else "No guidance")
        if metric == "silhouette":
            ax.axhline(float(runs[0].summary["threshold"]), color="gray",
                       ls=":", lw=1, label="Gate threshold")
        ax.set(title=TITLES[metric], xlabel="Epoch", ylabel="Score")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{runs[0].dataset} · gamma={gamma_label(runs[0].gamma)} · "
                 f"{len(runs)} seeds\nScores at gate checks and following epochs; "
                 "faint lines: seeds; bold lines: means")
    if not any(run.openings() for run in runs):
        axes[0, 0].text(0.98, 0.03, "No gates opened; curves overlap",
                        transform=axes[0, 0].transAxes, ha="right", va="bottom",
                        fontsize=9, color="#444444")
    fig.tight_layout()
    save(fig, path)


def plot_gate_response(runs: list[Run], path: Path, *, include_control: bool = False) -> None:
    """One dataset/gamma: checks, paired response, and guidance activity."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1.2, 1.2, 0.8]})
    check_scores: dict[int, list[float]] = defaultdict(list)
    control_scores: dict[int, list[float]] = defaultdict(list)
    differences = {metric: defaultdict(list) for metric in ("ARI", "silhouette")}
    applied: dict[int, list[float]] = defaultdict(list)
    opened_x, opened_y = [], []
    for run in runs:
        checks = run.gate_checks()
        axes[0].plot([int(row["epoch"]) for row in checks],
                     [float(row["silhouette"]) for row in checks],
                     color="#1675a6", alpha=0.2, lw=1,
                     label="Fixed-gate seeds" if run is runs[0] else None)
        if include_control:
            control_checks = [row for row in run.control
                              if is_true(row["scheduled_evaluation"])]
            axes[0].plot([int(row["epoch"]) for row in control_checks],
                         [float(row["silhouette"]) for row in control_checks],
                         color="#e18739", alpha=0.2, lw=1,
                         label="No-guidance seeds" if run is runs[0] else None)
            for row in control_checks:
                control_scores[int(row["epoch"])].append(float(row["silhouette"]))
        for row in checks:
            epoch = int(row["epoch"])
            score = float(row["silhouette"])
            check_scores[epoch].append(score)
            if is_true(row["gate_opened"]):
                opened_x.append(epoch)
                opened_y.append(score)
        epochs = run.metric_epochs()
        for index, metric in enumerate(("ARI", "silhouette"), start=1):
            values = run.delta_series(metric)
            axes[index].plot(epochs, values, color="#1675a6", alpha=0.16, lw=1)
            for epoch, value in zip(epochs, values):
                differences[metric][epoch].append(value)
        for row in run.history:
            applied[int(row["epoch"])].append(float(is_true(row["guidance_applied"])))

    check_epochs = sorted(check_scores)
    axes[0].plot(check_epochs, [mean(check_scores[e]) for e in check_epochs],
                 color="#1675a6", marker="o", ms=3, lw=2.4,
                 label="Fixed-gate mean", zorder=3)
    if include_control:
        control_epochs = sorted(control_scores)
        axes[0].plot(control_epochs, [mean(control_scores[e]) for e in control_epochs],
                     color="#e18739", ls="--", marker="o", ms=3, lw=2.4,
                     label="No-guidance mean", zorder=3.5)
    if opened_x:
        axes[0].scatter(opened_x, opened_y, color="#bd3f53", marker="*", s=50,
                        alpha=0.75, label="Gate opened", zorder=5)
    axes[0].axhline(float(runs[0].summary["threshold"]), color="gray", ls="--",
                    lw=1, label="Threshold")
    axes[0].set_ylabel("Reference silhouette")
    axes[0].legend(frameon=False, fontsize=8)

    for index, metric in enumerate(("ARI", "silhouette"), start=1):
        epochs = sorted(differences[metric])
        axes[index].plot(epochs, [mean(differences[metric][e]) for e in epochs],
                         color="#1675a6", lw=2.4, label="Mean paired difference")
        axes[index].axhline(0, color="gray", ls="--", lw=1)
        axes[index].set_ylabel(f"Δ {metric}")

    epochs = sorted(applied)
    axes[3].plot(epochs, [mean(applied[e]) for e in epochs], color="#79529d",
                 marker="o", ms=2, lw=1.5)
    axes[3].set(ylabel="Fraction guided", xlabel="Epoch", ylim=(-0.05, 1.05))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"{runs[0].dataset} · gamma={gamma_label(runs[0].gamma)}\n"
                 "Each gate check controls subsequent epochs; Δ = fixed − no guidance")
    fig.tight_layout()
    save(fig, path)


def plot_final_differences(runs: list[Run], path: Path) -> None:
    """One gamma: final paired outcomes across datasets."""
    datasets = ordered_datasets(runs)
    fig, axes = plt.subplots(2, 2, figsize=(12, max(7, 1.2 * len(datasets) + 3)))
    for ax, metric in zip(axes.flat, METRICS):
        for index, dataset in enumerate(datasets):
            group = sorted((r for r in runs if r.dataset == dataset), key=lambda r: r.seed)
            deltas = [r.final_delta(metric) for r in group]
            offsets = np.linspace(-0.14, 0.14, len(group)) if len(group) > 1 else [0]
            ax.scatter(deltas, [index + offset for offset in offsets],
                       color="#1675a6", alpha=0.75, s=35, zorder=2)
            ax.scatter(mean(deltas), index, marker="D", color="#222222", s=45, zorder=3)
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_yticks(range(len(datasets)), datasets)
        ax.invert_yaxis()
        ax.set(xlabel="Fixed − no guidance", title=TITLES[metric])
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"Final paired differences · gamma={gamma_label(runs[0].gamma)}\n"
                 "Blue points: seeds; black diamonds: means")
    fig.tight_layout()
    save(fig, path)


def plot_gamma_response(runs: list[Run], metrics: tuple[str, str], path: Path) -> None:
    """One dataset: connect each seed's paired outcome across gamma values."""
    gammas = sorted({r.gamma for r in runs})
    seeds = sorted({r.seed for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, metrics):
        for seed in seeds:
            selected = sorted((r for r in runs if r.seed == seed), key=lambda r: r.gamma)
            ax.plot([r.gamma for r in selected], [r.final_delta(metric) for r in selected],
                    color="#1675a6", marker="o", alpha=0.35, lw=1.2, ms=4)
        by_gamma = {gamma: [r.final_delta(metric) for r in runs if r.gamma == gamma]
                    for gamma in gammas}
        ax.plot(gammas, [mean(by_gamma[g]) for g in gammas], color="#222222",
                marker="D", lw=2.2, ms=6, label="Mean")
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set(xscale="log", xticks=gammas, xlabel="Gamma (log scale)",
               ylabel=f"Change vs. no guidance\n({metric} score points)",
               title=TITLES[metric])
        ax.set_xticklabels([gamma_label(g) for g in gammas])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(f"{runs[0].dataset} · final paired outcome across gamma\n"
                 "Positive change favors the fixed gate; blue lines: matched seeds; "
                 "black line: mean")
    fig.tight_layout()
    save(fig, path)


def plot_heatmaps(runs: list[Run], datasets: list[str], gammas: list[float],
                  kind: str, path: Path) -> None:
    """Across gamma: positive-outcome counts or guidance activity."""
    if kind == "positive":
        labels = ("ARI", "silhouette")
        title = "Seeds improved over no guidance"
    else:
        labels = ("Gate openings", "Guided epochs")
        title = "Mean guidance activity per run"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, label in zip(axes, labels):
        matrix = np.full((len(datasets), len(gammas)), np.nan)
        annotations = {}
        for i, dataset in enumerate(datasets):
            for j, gamma in enumerate(gammas):
                group = [r for r in runs if r.dataset == dataset and r.gamma == gamma]
                if not group:
                    continue
                if kind == "positive":
                    count = sum(r.final_delta(label) > 0 for r in group)
                    matrix[i, j] = count / len(group)
                    annotations[i, j] = f"{count}/{len(group)}"
                else:
                    values = ([r.openings() for r in group] if label == "Gate openings"
                              else [r.applied_epochs() for r in group])
                    matrix[i, j] = mean(values)
                    annotations[i, j] = f"{mean(values):.1f}"
        vmax = 1 if kind == "positive" else max(1, float(np.nanmax(matrix)))
        image = ax.imshow(matrix, vmin=0, vmax=vmax, cmap="YlGnBu", aspect="auto")
        ax.set(xticks=range(len(gammas)), yticks=range(len(datasets)), title=label)
        ax.set_xticklabels([gamma_label(g) for g in gammas])
        ax.set_yticklabels(datasets)
        ax.set_xlabel("Gamma")
        for (i, j), value in annotations.items():
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if matrix[i, j] > vmax * 0.55 else "black")
        fig.colorbar(image, ax=ax, shrink=0.8,
                     label="Fraction of seeds" if kind == "positive" else "Mean count")
    fig.suptitle(title)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT,
                        help="Directory containing gamma-<value>/<dataset>/seed-<n>/summary.json")
    parser.add_argument("--output", type=Path,
                        help="Graph directory (default: <input>/graphs)")
    args = parser.parse_args()
    try:
        runs = load_runs(args.input)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    output = args.output or args.input / "graphs"
    gammas = sorted({r.gamma for r in runs})
    datasets = ordered_datasets(runs)
    count = 0
    for gamma in gammas:
        gamma_runs = [r for r in runs if r.gamma == gamma]
        folder = output / "by_gamma" / f"gamma-{gamma_label(gamma)}"
        plot_final_differences(gamma_runs, folder / "final_differences.png")
        count += 1
        for dataset in ordered_datasets(gamma_runs):
            group = [r for r in gamma_runs if r.dataset == dataset]
            plot_quality(group, folder / f"{dataset}_quality.png")
            plot_gate_response(group, folder / f"{dataset}_gate_response.png")
            plot_gate_response(group, folder / f"{dataset}_gate_response_with_fixed.png",
                               include_control=True)
            count += 3
    comparison = output / "across_gamma"
    for dataset in datasets:
        group = [r for r in runs if r.dataset == dataset]
        plot_gamma_response(group, ("ARI", "silhouette"),
                            comparison / f"{dataset}_gamma_response.png")
        plot_gamma_response(group, ("NMI", "ACC"),
                            comparison / f"{dataset}_other_metrics.png")
        count += 2
    plot_heatmaps(runs, datasets, gammas, "positive", comparison / "positive_seeds.png")
    plot_heatmaps(runs, datasets, gammas, "activity", comparison / "gate_activity.png")
    count += 2
    print(f"Graphed {len(runs)} completed runs into {count} figures under {output}")


if __name__ == "__main__":
    main()
