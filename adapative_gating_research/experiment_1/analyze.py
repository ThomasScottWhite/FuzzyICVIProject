"""Summarize gate frequency and paired silhouette trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parent / "runs"


def read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def score(row: dict) -> float:
    return float(row["silhouette"]) if row["silhouette"] else float("nan")


def analyze_run(path: Path, plot_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    fixed = read_history(path.parent / "fixed_history.csv")
    control = read_history(path.parent / "no_guidance_history.csv")
    if len(fixed) != len(control) or any(a["epoch"] != b["epoch"] for a, b in zip(fixed, control)):
        raise ValueError(f"Histories do not match: {path.parent}")

    evaluations = []
    for a, b in zip(fixed, control):
        if a["evaluated"] == "True":
            evaluations.append((a, b))
    eligible = [a for a, _ in evaluations if int(a["epoch"]) < len(fixed)]
    openings = [a for a, _ in evaluations if a["gate_opened"] == "True"]
    eligible_openings = [a for a in eligible if a["gate_opened"] == "True"]
    applied = [int(row["epoch"]) for row in fixed if row["guidance_applied"] == "True"]
    first_open = int(openings[0]["epoch"]) if openings else None
    first_applied = applied[0] if applied else None
    open_fraction = len(openings) / len(evaluations) if evaluations else float("nan")
    early_cutoff = int(evaluations[(len(evaluations) - 1) // 3][0]["epoch"]) if evaluations else None

    trajectory = []
    for a, b in evaluations:
        fixed_score, control_score = score(a), score(b)
        trajectory.append({
            "dataset": summary["dataset"],
            "seed": summary["seed"],
            "epoch": int(a["epoch"]),
            "fixed_silhouette": fixed_score,
            "no_guidance_silhouette": control_score,
            "paired_difference": fixed_score - control_score,
            "gate_opened": a["gate_opened"] == "True",
            "guidance_applied_by_this_epoch": any(e <= int(a["epoch"]) for e in applied),
        })

    x = [row["epoch"] for row in trajectory]
    y_fixed = [row["fixed_silhouette"] for row in trajectory]
    y_control = [row["no_guidance_silhouette"] for row in trajectory]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y_fixed, marker="o", label="Fixed gate (rho=0.5)")
    ax.plot(x, y_control, marker="o", label="No guidance")
    opened = [row for row in trajectory if row["gate_opened"]]
    ax.scatter([row["epoch"] for row in opened],
               [row["fixed_silhouette"] for row in opened],
               marker="*", s=120, color="crimson", label="Gate opened", zorder=3)
    ax.axhline(summary["threshold"], color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Training epoch", ylabel="Mean reference silhouette",
           title=f"{summary['dataset']} · seed {summary['seed']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{summary['dataset']}_seed-{summary['seed']}.png", dpi=160)
    plt.close(fig)

    last = trajectory[-1] if trajectory else None
    row = {
        "dataset": summary["dataset"],
        "seed": summary["seed"],
        "reference_size": summary["reference_size"],
        "evaluations": len(evaluations),
        "eligible_evaluations": len(eligible),
        "open_count": len(openings),
        "eligible_open_count": len(eligible_openings),
        "open_fraction": open_fraction,
        "never_opened": len(openings) == 0,
        "opened_early": first_open is not None and first_open <= early_cutoff,
        "opened_nearly_every_evaluation": bool(evaluations) and open_fraction >= 0.9,
        "first_open_epoch": first_open,
        "first_applied_epoch": first_applied,
        "adjustments_applied": len(applied),
        "final_fixed_silhouette": summary["results"]["fixed"]["final_silhouette"],
        "final_no_guidance_silhouette": summary["results"]["no_guidance"]["final_silhouette"],
        "final_paired_silhouette_difference": (
            last["paired_difference"] if last is not None else float("nan")
        ),
        "final_fixed_ARI": summary["results"]["fixed"]["external_metrics"]["ARI"],
        "final_no_guidance_ARI": summary["results"]["no_guidance"]["external_metrics"]["ARI"],
    }
    return row, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*/seed-*/summary.json"))
    if not paths:
        parser.error(f"No completed runs found under {args.input}")
    output = args.output or args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows, trajectories = [], []
    for path in paths:
        row, series = analyze_run(path, output)
        rows.append(row)
        trajectories.extend(series)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "trajectories.csv", trajectories)
    dataset_rows = []
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        dataset_rows.append({
            "dataset": dataset,
            "seeds_completed": len(group),
            "mean_open_fraction": mean(row["open_fraction"] for row in group),
            "never_opened_runs": sum(row["never_opened"] for row in group),
            "opened_early_runs": sum(row["opened_early"] for row in group),
            "opened_nearly_every_evaluation_runs": sum(
                row["opened_nearly_every_evaluation"] for row in group
            ),
            "mean_final_paired_silhouette_difference": mean(
                row["final_paired_silhouette_difference"] for row in group
            ),
            "mean_final_paired_ARI_difference": mean(
                row["final_fixed_ARI"] - row["final_no_guidance_ARI"] for row in group
            ),
        })
    write_csv(output / "datasets.csv", dataset_rows)
    print(f"Analyzed {len(rows)} paired runs; results in {output}")


if __name__ == "__main__":
    main()
