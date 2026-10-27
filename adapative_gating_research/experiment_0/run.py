"""Time exact silhouette scoring against the rest of one DMSC training run.

Run from the repository root with ``python -m adapative_gating_research.experiment_0.run``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dmsc_custom.data import load_dataset
from dmsc_custom.trainer import DMSCConfig, DMSCTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("iris", "wine", "covertype", "fashion_mnist"), default="iris")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--reference-size", type=int, default=2000,
                        help="Maximum number of points used for each exact silhouette score.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-home", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, default=Path("adapative_gating_research/experiment_0/runs"))
    args = parser.parse_args()
    if args.epochs < 1 or args.evaluation_interval < 1 or args.reference_size < 2:
        parser.error("--epochs, --evaluation-interval, and --reference-size must be positive (size >= 2)")

    load_started = time.perf_counter()
    dataset = load_dataset(args.dataset, data_home=args.data_home)
    load_seconds = time.perf_counter() - load_started
    n_samples = len(dataset.features)
    n_clusters = int(np.unique(dataset.labels).size)
    reference_size = min(args.reference_size, n_samples)
    if reference_size <= n_clusters:
        parser.error("--reference-size must exceed the number of clusters")

    output = args.output / dataset.name / f"seed-{args.seed}-n-{reference_size}-e-{args.epochs}"
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    reference = np.sort(np.random.default_rng(args.seed + 1).choice(
        n_samples, size=reference_size, replace=False,
    ))
    config = DMSCConfig(
        n_clusters=n_clusters,
        epochs=args.epochs,
        evaluation_interval=args.evaluation_interval,
        evaluation_size=reference_size,
        gate_metric="reference",
        validation_metrics=("reference",),
        adjustment_gamma=0.0,
        seed=args.seed,
        device=args.device,
    )
    fit_started = time.perf_counter()
    result = DMSCTrainer(config).fit(dataset.features, evaluation_indices=reference)
    fit_seconds = time.perf_counter() - fit_started

    silhouette_seconds = result.metric_seconds["reference"]
    other_seconds = fit_seconds - silhouette_seconds
    encoding_seconds = result.validation_seconds - silhouette_seconds
    accounted_seconds = result.ae_seconds + result.fcm_seconds + result.validation_seconds
    other_measured_stages = result.ae_seconds + encoding_seconds + result.fcm_seconds
    evaluations = [row for row in result.history if row["evaluated"]]
    # The trainer performs a final evaluation outside history when the last epoch
    # is not on the evaluation schedule.
    final_extra = int(args.epochs % args.evaluation_interval != 0)
    n_scores = len(evaluations) + final_extra
    summary = {
        "dataset": dataset.name,
        "n_samples": n_samples,
        "n_features": int(dataset.features.shape[1]),
        "reference_size": reference_size,
        "silhouette_evaluations": n_scores,
        "config": asdict(config),
        "device": result.device,
        "timing_seconds": {
            "data_loading": load_seconds,
            "fit_wall": fit_seconds,
            "silhouette": silhouette_seconds,
            "rest_of_fit": other_seconds,
            "autoencoder": result.ae_seconds,
            "embedding_encoding": encoding_seconds,
            "fcm": result.fcm_seconds,
            "other_fit_overhead": fit_seconds - accounted_seconds,
            "other_measured_stages": other_measured_stages,
            "silhouette_per_evaluation": silhouette_seconds / n_scores,
        },
        "silhouette_share_of_fit": silhouette_seconds / fit_seconds,
        "silhouette_share_of_measured_stages": silhouette_seconds / accounted_seconds,
        "final_silhouette": result.final_silhouette,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "epoch", "ae_seconds", "fcm_seconds", "silhouette_seconds", "silhouette_score",
        ))
        writer.writeheader()
        for row in evaluations:
            writer.writerow({
                "epoch": row["epoch"],
                "ae_seconds": row["ae_seconds"],
                "fcm_seconds": row["fcm_seconds"],
                "silhouette_seconds": row["validation_seconds_reference"],
                "silhouette_score": row["silhouette_reference"],
            })

    print(f"{dataset.name}: {n_samples} samples, {reference_size} silhouette points, "
          f"{n_scores} evaluations on {result.device}")
    print(f"Silhouette: {silhouette_seconds:.3f}s ({summary['silhouette_share_of_fit']:.1%} of fit)")
    print(f"Rest of fit: {other_seconds:.3f}s; total fit: {fit_seconds:.3f}s")
    print(f"Other measured stages: {other_measured_stages:.3f}s; "
          f"silhouette is {summary['silhouette_share_of_measured_stages']:.1%} of measured stages")
    print(f"Saved {output / 'summary.json'} and {output / 'evaluations.csv'}")


if __name__ == "__main__":
    main()
