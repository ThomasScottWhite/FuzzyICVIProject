from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dmsc_custom.trainer import DMSCConfig
from experiments.common import (
    DEFAULT_DATA_HOME,
    DEFAULT_SEEDS,
    PROJECT_ROOT,
    describe,
    paired_bootstrap_interval,
    run_one,
    write_aggregate_json,
    write_csv,
)

DATASETS = ("covertype", "fashion_mnist")
CLUSTERS = {"covertype": 7, "fashion_mnist": 10}
BASELINE_SCHEDULE = "every_10"


@dataclass(frozen=True)
class Schedule:
    name: str
    epochs: tuple[int, ...]
    comparisons: tuple[str, ...]


def schedules() -> tuple[Schedule, ...]:
    every = lambda interval: tuple(range(interval, 201, interval))
    front_loaded = (
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        38, 56, 74, 92, 110, 128, 146, 164, 182, 200,
    )
    back_loaded = (
        18, 36, 54, 72, 90, 108, 126, 144, 162, 180,
        182, 184, 186, 188, 190, 192, 194, 196, 198, 200,
    )
    return (
        Schedule("every_1", every(1), ("fixed_length",)),
        Schedule("every_5", every(5), ("fixed_length",)),
        Schedule("every_10", every(10), ("fixed_length", "fixed_budget")),
        Schedule("every_20", every(20), ("fixed_length",)),
        Schedule("front_loaded", front_loaded, ("fixed_budget",)),
        Schedule("back_loaded", back_loaded, ("fixed_budget",)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-length and fixed-budget DMSC schedule comparisons."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda:0.")
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA_HOME)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "experiment_1")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed runs with a matching configuration (default: resume).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="List the configurations without downloading data or training models.",
    )
    return parser


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        baseline_by_seed = {
            int(row["seed"]): row
            for row in dataset_rows
            if row["schedule"] == BASELINE_SCHEDULE
        }
        for schedule in schedules():
            selected = [row for row in dataset_rows if row["schedule"] == schedule.name]
            if not selected:
                continue
            item: dict[str, Any] = {
                "dataset": dataset,
                "schedule": schedule.name,
                "comparisons": ",".join(schedule.comparisons),
                "n_runs": len(selected),
            }
            for metric in ("ACC", "NMI", "ARI"):
                values = [float(row["external_metrics"][metric]) for row in selected]
                summary = describe(values)
                item[f"{metric}_mean"] = summary["mean"]
                item[f"{metric}_std"] = summary["std"]
                paired = [
                    float(row["external_metrics"][metric])
                    - float(baseline_by_seed[int(row["seed"])]["external_metrics"][metric])
                    for row in selected
                    if int(row["seed"]) in baseline_by_seed
                ]
                ci_low, ci_high = paired_bootstrap_interval(paired)
                item[f"delta_{metric}_mean"] = float(np.mean(paired)) if paired else None
                item[f"delta_{metric}_ci95_low"] = ci_low
                item[f"delta_{metric}_ci95_high"] = ci_high
            for timing_name in (
                "autoencoder_training", "fcm", "validation", "io", "compute", "wall"
            ):
                values = [float(row["timing_seconds"][timing_name]) for row in selected]
                summary = describe(values)
                item[f"{timing_name}_median_seconds"] = summary["median"]
                item[f"{timing_name}_iqr_seconds"] = summary["iqr"]
                paired = [
                    float(row["timing_seconds"][timing_name])
                    - float(baseline_by_seed[int(row["seed"])]["timing_seconds"][timing_name])
                    for row in selected
                    if int(row["seed"]) in baseline_by_seed
                ]
                ci_low, ci_high = paired_bootstrap_interval(paired)
                item[f"delta_{timing_name}_mean_seconds"] = float(np.mean(paired)) if paired else None
                item[f"delta_{timing_name}_ci95_low_seconds"] = ci_low
                item[f"delta_{timing_name}_ci95_high_seconds"] = ci_high
            aggregate.append(item)
    return aggregate


def main() -> None:
    args = _parser().parse_args()
    selected_schedules = schedules()
    total = len(args.datasets) * len(selected_schedules) * len(args.seeds)
    if args.plan_only:
        print(f"Experiment 1: {total} runs ({len(args.datasets)} datasets × 6 schedules × {len(args.seeds)} seeds)")
        for schedule in selected_schedules:
            print(f"{schedule.name}: {len(schedule.epochs)} evaluations; {','.join(schedule.comparisons)}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, Any]] = []
    completed_count = 0
    for dataset in args.datasets:
        for schedule in selected_schedules:
            for seed in args.seeds:
                config = DMSCConfig(
                    n_clusters=CLUSTERS[dataset],
                    epochs=200,
                    evaluation_interval=10,
                    evaluation_epochs=schedule.epochs,
                    gate_metric="reference",
                    validation_metrics=("reference",),
                    evaluation_size=20_000,
                    seed=seed,
                    device=args.device,
                )
                run_key = f"{dataset}/{schedule.name}/seed-{seed}"
                run_dir = args.output / dataset / schedule.name / f"seed-{seed}"
                summary, skipped = run_one(
                    dataset_name=dataset,
                    data_home=args.data_home,
                    run_dir=run_dir,
                    run_key=run_key,
                    config=config,
                    resume=args.resume,
                    metadata={
                        "schedule": schedule.name,
                        "comparison_groups": list(schedule.comparisons),
                    },
                )
                summary["schedule"] = schedule.name
                run_summaries.append(summary)
                completed_count += 1
                status = "already complete" if skipped else "completed"
                print(f"[{completed_count}/{total}] {run_key}: {status}")

    run_rows = [
        {
            "dataset": summary["dataset"],
            "schedule": summary["schedule"],
            "seed": summary["seed"],
            "n_samples": summary["n_samples"],
            "external_metrics": summary["external_metrics"],
            "gate_triggers": summary["gate_triggers"],
            "adjustments_applied": summary["adjustments_applied"],
            "timing_seconds": summary["timing_seconds"],
            "final_silhouette": summary["final_silhouette"],
            "run_key": summary["run_key"],
        }
        for summary in run_summaries
    ]
    aggregate = _aggregate(run_summaries)
    write_aggregate_json(args.output / "runs.json", run_rows)
    write_csv(args.output / "runs.csv", run_rows)
    write_aggregate_json(args.output / "aggregate.json", aggregate)
    write_csv(args.output / "aggregate.csv", aggregate)
    print(f"Experiment 1 complete: {len(run_summaries)} runs; results in {args.output}")


if __name__ == "__main__":
    main()
