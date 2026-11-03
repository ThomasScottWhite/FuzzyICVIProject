"""Graph completed fixed-gate versus no-guidance experiment runs.

Run from the project root with::

    python -m adapative_gating_research.experiment_1.graph

Only directories containing summary.json are included. Plots are written to
``runs/graphs`` by default; no training data or results are modified.
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


RUNS = Path(__file__).resolve().parent / "runs"
COLORS = {"fixed": "#1976a3", "no_guidance": "#e28743"}
LABELS = {"fixed": "Fixed gate", "no_guidance": "No guidance"}


@dataclass
class Run:
    dataset: str
    seed: int
    summary: dict
    history: dict[str, list[dict]]

    def final(self, mode: str, metric: str) -> float:
        result = self.summary["results"][mode]
        value = (result["final_silhouette"] if metric == "silhouette"
                 else result["external_metrics"][metric])
        return float(value)


def read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_runs(input_dir: Path) -> dict[str, list[Run]]:
    grouped: dict[str, list[Run]] = defaultdict(list)
    paths = sorted(input_dir.glob("*/seed-*/summary.json"))
    if not paths:
        raise ValueError(f"No completed runs (summary.json) found under {input_dir}")
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        histories = {}
        for mode in COLORS:
            history_path = path.parent / f"{mode}_history.csv"
            if not history_path.is_file():
                raise ValueError(f"Missing history for completed run: {history_path}")
            histories[mode] = read_history(history_path)
        fixed_epochs = [int(row["epoch"]) for row in histories["fixed"]
                        if row["evaluated"].lower() == "true"]
        control_epochs = [int(row["epoch"]) for row in histories["no_guidance"]
                          if row["evaluated"].lower() == "true"]
        if not fixed_epochs or fixed_epochs != control_epochs:
            raise ValueError(f"Evaluation epochs are missing or differ in {path.parent}")
        dataset = path.parent.parent.name
        grouped[dataset].append(Run(dataset, int(summary["seed"]), summary, histories))
    for runs in grouped.values():
        runs.sort(key=lambda run: run.seed)
    return dict(sorted(grouped.items()))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(grouped: dict[str, list[Run]], output: Path) -> None:
    """Show each seed's paired final scores, without hiding seed variation."""
    fig, axes = plt.subplots(len(grouped), 2, figsize=(10, max(3.4, 2.8 * len(grouped))),
                             squeeze=False, sharex="col")
    for row_index, (dataset, runs) in enumerate(grouped.items()):
        for col_index, (metric, title) in enumerate((("silhouette", "Reference silhouette"),
                                                      ("ARI", "Adjusted Rand index"))):
            ax = axes[row_index][col_index]
            for run in runs:
                control = run.final("no_guidance", metric)
                fixed = run.final("fixed", metric)
                ax.plot([0, 1], [control, fixed], color="#888888", alpha=0.65, lw=1.4)
                ax.scatter([0, 1], [control, fixed],
                           color=[COLORS["no_guidance"], COLORS["fixed"]], s=32, zorder=3)
            ax.set_xticks([0, 1], ["No guidance", "Fixed gate"])
            ax.set_xlim(-0.25, 1.25)
            ax.grid(axis="y", alpha=0.25)
            ax.set_ylabel(f"{dataset} (n={len(runs)})\n{title}")
            if row_index == 0:
                ax.set_title(title)
    fig.suptitle("Final paired outcomes · each line is one seed", fontsize=13)
    fig.tight_layout()
    save(fig, output / "final_outcomes.png")


def plot_differences(grouped: dict[str, list[Run]], output: Path) -> None:
    """Expose effect direction and consistency across datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3.8, 0.65 * len(grouped) + 2)))
    for ax, metric, title in zip(axes, ("silhouette", "ARI"),
                                 ("Reference silhouette", "Adjusted Rand index")):
        for index, (dataset, runs) in enumerate(grouped.items()):
            deltas = [run.final("fixed", metric) - run.final("no_guidance", metric)
                      for run in runs]
            offsets = ([0.0] if len(deltas) == 1 else
                       [0.24 * (i / (len(deltas) - 1) - 0.5) for i in range(len(deltas))])
            ax.scatter(deltas, [index + offset for offset in offsets],
                       color=COLORS["fixed"], alpha=0.7, s=35, label="Individual seed" if index == 0 else None)
            ax.scatter(mean(deltas), index, marker="D", color="#1d1d1d", s=52,
                       label="Mean" if index == 0 else None, zorder=4)
        ax.axvline(0, color="#777777", linestyle="--", lw=1)
        ax.set_yticks(range(len(grouped)), list(grouped))
        ax.invert_yaxis()
        ax.set_xlabel("Fixed gate − no guidance")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Final paired differences · right of zero favors fixed gate", fontsize=13)
    fig.tight_layout()
    save(fig, output / "final_differences.png")


def plot_dataset(dataset: str, runs: list[Run], output: Path) -> None:
    """Plot trajectories, within-seed differences, and gate activity."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.4]})
    scores: dict[str, dict[int, list[float]]] = {
        mode: defaultdict(list) for mode in COLORS
    }
    differences: dict[int, list[float]] = defaultdict(list)
    opened: dict[int, list[bool]] = defaultdict(list)

    for run in runs:
        fixed = [row for row in run.history["fixed"]
                 if row["evaluated"].lower() == "true"]
        control = [row for row in run.history["no_guidance"]
                   if row["evaluated"].lower() == "true"]
        epochs = [int(row["epoch"]) for row in fixed]
        fixed_scores = [float(row["silhouette"]) for row in fixed]
        control_scores = [float(row["silhouette"]) for row in control]
        deltas = [a - b for a, b in zip(fixed_scores, control_scores)]
        axes[0].plot(epochs, fixed_scores, color=COLORS["fixed"], alpha=0.18, lw=1)
        axes[0].plot(epochs, control_scores, color=COLORS["no_guidance"], alpha=0.18, lw=1)
        axes[1].plot(epochs, deltas, color=COLORS["fixed"], alpha=0.25, lw=1)
        for epoch, a, b, delta, row in zip(epochs, fixed_scores, control_scores, deltas, fixed):
            scores["fixed"][epoch].append(a)
            scores["no_guidance"][epoch].append(b)
            differences[epoch].append(delta)
            opened[epoch].append(row["gate_opened"].lower() == "true")

    for mode in COLORS:
        epochs = sorted(scores[mode])
        axes[0].plot(epochs, [mean(scores[mode][epoch]) for epoch in epochs],
                     color=COLORS[mode], lw=2.5, marker="o", ms=3, label=LABELS[mode])
    epochs = sorted(differences)
    axes[1].plot(epochs, [mean(differences[epoch]) for epoch in epochs],
                 color=COLORS["fixed"], lw=2.5, marker="o", ms=3, label="Mean paired difference")
    axes[1].axhline(0, color="#777777", linestyle="--", lw=1)
    axes[2].plot(epochs, [mean(opened[epoch]) for epoch in epochs],
                 color="#b24a78", lw=2, marker="o", ms=3, label="Gate opened")
    axes[2].set_ylim(-0.05, 1.05)
    axes[0].axhline(float(runs[0].summary["threshold"]), color="#888888",
                    linestyle=":", lw=1, label="Gate threshold")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(loc="best", frameon=False)
    axes[0].set_ylabel("Reference silhouette")
    axes[1].set_ylabel("Fixed − no guidance")
    axes[2].set_ylabel("Fraction of runs")
    axes[2].set_xlabel("Training epoch")
    fig.suptitle(f"{dataset} · {len(runs)} completed seed(s)\n"
                 "Faint lines are seeds; bold lines are means", fontsize=13)
    fig.tight_layout()
    save(fig, output / f"{dataset}_trajectories.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RUNS,
                        help="Run directory containing <dataset>/seed-<n>/summary.json")
    parser.add_argument("--output", type=Path,
                        help="Plot directory (default: <input>/graphs)")
    args = parser.parse_args()
    try:
        grouped = load_runs(args.input)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    output = args.output or args.input / "graphs"
    output.mkdir(parents=True, exist_ok=True)
    plot_outcomes(grouped, output)
    plot_differences(grouped, output)
    for dataset, runs in grouped.items():
        plot_dataset(dataset, runs, output)
    print(f"Graphed {sum(map(len, grouped.values()))} completed runs across "
          f"{len(grouped)} datasets in {output}")


if __name__ == "__main__":
    main()
