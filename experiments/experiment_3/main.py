from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from dmsc_custom.trainer import DMSCConfig
from experiments.common import (
    DEFAULT_DATA_HOME,
    DEFAULT_SEEDS,
    PROJECT_ROOT,
    describe,
    load_history,
    paired_bootstrap_interval,
    run_one,
    write_aggregate_json,
    write_csv,
)

DATASETS = ("covertype", "fashion_mnist")
CLUSTERS = {"covertype": 7, "fashion_mnist": 10}
MODES = (
    ("reference_shadow", "reference", ("reference", "sampled", "simplified")),
    ("sampled_controller", "sampled", ("sampled",)),
    ("simplified_controller", "simplified", ("simplified",)),
)
PRIMARY_THRESHOLD = 0.5
DIAGNOSTIC_THRESHOLDS = tuple(round(value, 1) for value in np.linspace(-1.0, 1.0, 21))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare reference, sampled, and simplified silhouette validation in DMSC."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda:0.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm training progress bars.")
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA_HOME)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "experiment_3")
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


def _number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _decision_rows(
    summaries: list[dict[str, Any]], output_root: Path
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, float], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "tn": 0, "false_trigger": 0, "missed_trigger": 0, "undefined": 0}
    )
    for summary in summaries:
        if summary.get("mode") != "reference_shadow":
            continue
        run_dir = output_root / summary["dataset"] / summary["mode"] / f"seed-{summary['seed']}"
        history = load_history(run_dir)
        for event in history:
            if event.get("evaluated") != "True":
                continue
            reference = _number(event.get("silhouette_reference"))
            for alternative in ("sampled", "simplified"):
                score = _number(event.get(f"silhouette_{alternative}"))
                for threshold in DIAGNOSTIC_THRESHOLDS:
                    row_counts = counts[(summary["dataset"], alternative, threshold)]
                    if reference is None or score is None:
                        row_counts["undefined"] += 1
                        continue
                    reference_trigger = reference >= threshold
                    alternative_trigger = score >= threshold
                    if reference_trigger and alternative_trigger:
                        row_counts["tp"] += 1
                    elif not reference_trigger and not alternative_trigger:
                        row_counts["tn"] += 1
                    elif alternative_trigger:
                        row_counts["false_trigger"] += 1
                    else:
                        row_counts["missed_trigger"] += 1

    rows: list[dict[str, Any]] = []
    for (dataset, alternative, threshold), values in sorted(counts.items()):
        comparable = values["tp"] + values["tn"] + values["false_trigger"] + values["missed_trigger"]
        precision_denominator = values["tp"] + values["false_trigger"]
        recall_denominator = values["tp"] + values["missed_trigger"]
        rows.append(
            {
                "dataset": dataset,
                "alternative": alternative,
                "threshold": threshold,
                "primary_threshold": threshold == PRIMARY_THRESHOLD,
                "evaluations": comparable,
                "undefined_evaluations": values["undefined"],
                "agreement": (values["tp"] + values["tn"]) / comparable if comparable else None,
                "true_positive": values["tp"],
                "true_negative": values["tn"],
                "false_trigger": values["false_trigger"],
                "missed_trigger": values["missed_trigger"],
                "precision": values["tp"] / precision_denominator if precision_denominator else None,
                "recall": values["tp"] / recall_denominator if recall_denominator else None,
            }
        )
    return rows


def _aggregate_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        selected_dataset = [row for row in summaries if row["dataset"] == dataset]
        reference_by_seed = {
            int(row["seed"]): row
            for row in selected_dataset
            if row["mode"] == "reference_shadow"
        }
        for mode, _, _ in MODES:
            selected = [row for row in selected_dataset if row["mode"] == mode]
            if not selected:
                continue
            item: dict[str, Any] = {"dataset": dataset, "mode": mode, "n_runs": len(selected)}
            for metric in ("ACC", "NMI", "ARI"):
                values = [float(row["external_metrics"][metric]) for row in selected]
                stats = describe(values)
                item[f"{metric}_mean"] = stats["mean"]
                item[f"{metric}_std"] = stats["std"]
                paired = [
                    float(row["external_metrics"][metric])
                    - float(reference_by_seed[int(row["seed"])]["external_metrics"][metric])
                    for row in selected
                    if mode != "reference_shadow" and int(row["seed"]) in reference_by_seed
                ]
                ci_low, ci_high = paired_bootstrap_interval(paired)
                item[f"delta_{metric}_mean_vs_reference"] = float(np.mean(paired)) if paired else None
                item[f"delta_{metric}_ci95_low_vs_reference"] = ci_low
                item[f"delta_{metric}_ci95_high_vs_reference"] = ci_high

            for timing_name in ("autoencoder_training", "fcm", "validation", "io", "compute", "wall"):
                values = [float(row["timing_seconds"][timing_name]) for row in selected]
                stats = describe(values)
                item[f"{timing_name}_median_seconds"] = stats["median"]
                item[f"{timing_name}_iqr_seconds"] = stats["iqr"]
            for metric_name in ("reference", "sampled", "simplified"):
                selected_metrics = [
                    float(row["validation_metric_seconds"][metric_name])
                    for row in selected
                    if metric_name in row["validation_metric_seconds"]
                ]
                memory_values = [
                    float(row["peak_memory_bytes_by_metric"][metric_name])
                    for row in selected
                    if metric_name in row["peak_memory_bytes_by_metric"]
                ]
                item[f"{metric_name}_validation_median_seconds"] = describe(selected_metrics)["median"]
                item[f"{metric_name}_peak_memory_median_bytes"] = describe(memory_values)["median"]
            rows.append(item)
    return rows


def main() -> None:
    args = _parser().parse_args()
    total = len(args.datasets) * len(MODES) * len(args.seeds)
    if args.plan_only:
        print(f"Experiment 3: {total} runs ({len(args.datasets)} datasets × 3 modes × {len(args.seeds)} seeds)")
        for name, gate_metric, metrics in MODES:
            print(f"{name}: gate={gate_metric}; validation metrics={','.join(metrics)}")
        print(f"Primary threshold: {PRIMARY_THRESHOLD}; diagnostic thresholds: {len(DIAGNOSTIC_THRESHOLDS)}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, Any]] = []
    completed_count = 0
    device_reported = False
    for dataset in args.datasets:
        for mode, gate_metric, validation_metrics in MODES:
            for seed in args.seeds:
                config = DMSCConfig(
                    n_clusters=CLUSTERS[dataset],
                    epochs=200,
                    evaluation_interval=10,
                    gate_metric=gate_metric,
                    validation_metrics=validation_metrics,
                    evaluation_size=20_000,
                    silhouette_threshold=PRIMARY_THRESHOLD,
                    seed=seed,
                    device=args.device,
                )
                run_key = f"{dataset}/{mode}/seed-{seed}"
                run_dir = args.output / dataset / mode / f"seed-{seed}"
                summary, skipped = run_one(
                    dataset_name=dataset,
                    data_home=args.data_home,
                    run_dir=run_dir,
                    run_key=run_key,
                    config=config,
                    resume=args.resume,
                    progress_desc=(
                        None if args.no_progress else f"{dataset}/{mode}/seed-{seed}"
                    ),
                    report_device=not device_reported,
                    metadata={"mode": mode, "controller": gate_metric},
                )
                summary["mode"] = mode
                summary["controller"] = gate_metric
                run_summaries.append(summary)
                completed_count += 1
                status = "already complete" if skipped else "completed"
                if not skipped:
                    device_reported = True
                tqdm.write(f"[{completed_count}/{total}] {run_key}: {status}")

    run_rows = [
        {
            "dataset": summary["dataset"],
            "mode": summary["mode"],
            "controller": summary["controller"],
            "seed": summary["seed"],
            "n_samples": summary["n_samples"],
            "external_metrics": summary["external_metrics"],
            "gate_triggers": summary["gate_triggers"],
            "adjustments_applied": summary["adjustments_applied"],
            "final_validation_scores": summary["final_validation_scores"],
            "validation_metric_seconds": summary["validation_metric_seconds"],
            "peak_memory_bytes_by_metric": summary["peak_memory_bytes_by_metric"],
            "timing_seconds": summary["timing_seconds"],
            "run_key": summary["run_key"],
        }
        for summary in run_summaries
    ]
    aggregate = _aggregate_rows(run_summaries)
    decisions = _decision_rows(run_summaries, args.output)
    write_aggregate_json(args.output / "runs.json", run_rows)
    write_csv(args.output / "runs.csv", run_rows)
    write_aggregate_json(args.output / "aggregate.json", aggregate)
    write_csv(args.output / "aggregate.csv", aggregate)
    write_aggregate_json(args.output / "decision_summary.json", decisions)
    write_csv(args.output / "decision_summary.csv", decisions)
    print(f"Experiment 3 complete: {len(run_summaries)} runs; results in {args.output}")


if __name__ == "__main__":
    main()
