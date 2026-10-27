"""Run one paired fixed-gate/no-guidance comparison for a Slurm array task."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dmsc_custom.data import load_dataset
from dmsc_custom.clustering import FCMResult
from dmsc_custom.metrics import exact_silhouette, external_clustering_metrics
from dmsc_custom.trainer import DMSCConfig, DMSCTrainer

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"
DATASETS = ("iris", "wine", "covertype", "fashion_mnist")
THRESHOLD = 0.5
GAMMA = 0.001


def save_history(path: Path, history: list[dict], *, guidance_enabled: bool) -> None:
    """Keep only fields needed to understand when feedback changed training."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "epoch", "reconstruction_loss", "evaluated", "silhouette",
            "would_open_at_0_5", "gate_opened", "guidance_applied",
        ))
        writer.writeheader()
        for item in history:
            writer.writerow({
                "epoch": item["epoch"],
                "reconstruction_loss": item["reconstruction_loss"],
                "evaluated": item["evaluated"],
                "silhouette": item["silhouette"],
                "would_open_at_0_5": item["gate_triggered"],
                "gate_opened": guidance_enabled and item["gate_triggered"],
                "guidance_applied": guidance_enabled and item["adjustment_applied"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=RUNS)
    parser.add_argument("--data-home", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--reference-size", type=int, default=5000,
                        help="Maximum fixed subset size for gate silhouette (default: 5000)")
    parser.add_argument("--metrics-every-epoch", action="store_true",
                        help="Run diagnostic FCM and metrics at every epoch (expensive on large datasets)")
    args = parser.parse_args()

    if args.reference_size < 2:
        parser.error("--reference-size must be at least 2")
    if args.evaluation_interval < 1 or args.epochs < 2 * args.evaluation_interval:
        parser.error("Use at least two scheduled evaluations")
    if args.epochs % args.evaluation_interval:
        parser.error("--epochs must be divisible by --evaluation-interval")
    run_dir = args.output / args.dataset / f"seed-{args.seed}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        raise SystemExit(f"Run already exists: {summary_path}. Choose a new output directory.")

    data = load_dataset(args.dataset, data_home=args.data_home)
    n_samples = len(data.features)
    n_clusters = int(np.unique(data.labels).size)
    reference_count = min(args.reference_size, n_samples)
    if reference_count <= n_clusters:
        parser.error("--reference-size must exceed the number of clusters")
    # Keep one unlabeled subset fixed across epochs, conditions, and training seeds.
    reference = (
        np.arange(n_samples, dtype=np.int64)
        if reference_count == n_samples
        else np.sort(np.random.default_rng(2026).choice(
            n_samples, size=reference_count, replace=False,
        ))
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "reference_indices.npy", reference)
    reference_tensor = torch.as_tensor(reference, dtype=torch.long)
    results = {}
    for mode, gamma in (("fixed", GAMMA), ("no_guidance", 0.0)):
        config = DMSCConfig(
            n_clusters=n_clusters,
            epochs=args.epochs,
            evaluation_interval=args.evaluation_interval,
            silhouette_threshold=THRESHOLD,
            adjustment_gamma=gamma,
            gate_metric="reference",
            validation_metrics=("reference",),
            evaluation_size=reference_count,
            seed=args.seed,
            device=args.device,
        )
        checks = set(config.scheduled_evaluation_epochs)
        metric_epochs = (
            None if args.metrics_every_epoch else
            tuple(sorted(checks | {epoch + 1 for epoch in checks if epoch < config.epochs}))
        )
        metrics_path = run_dir / f"{mode}_epoch_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "epoch", "silhouette", "ACC", "NMI", "ARI",
                "scheduled_evaluation", "gate_opened", "guidance_applied",
            ))
            writer.writeheader()

            def log_epoch(
                epoch: int, embedding: torch.Tensor, clustering: FCMResult,
                record: dict[str, int | float | bool | str | None],
            ) -> None:
                reference_on_device = reference_tensor.to(embedding.device)
                silhouette = (record["silhouette"] if record["evaluated"] else
                              exact_silhouette(
                                  embedding[reference_on_device],
                                  clustering.labels[reference_on_device],
                                  block_size=config.silhouette_block_size,
                              ))
                scores = external_clustering_metrics(
                    data.labels, clustering.labels.detach().cpu().numpy()
                )
                writer.writerow({
                    "epoch": epoch,
                    "silhouette": silhouette,
                    "ACC": scores["ACC"],
                    "NMI": scores["NMI"],
                    "ARI": scores["ARI"],
                    "scheduled_evaluation": record["evaluated"],
                    "gate_opened": mode == "fixed" and record["gate_triggered"],
                    "guidance_applied": mode == "fixed" and record["adjustment_applied"],
                })
                handle.flush()

            started = perf_counter()
            result = DMSCTrainer(config).fit(
                data.features, evaluation_indices=reference,
                epoch_observer=log_epoch, epoch_observer_epochs=metric_epochs,
            )
            wall_seconds = perf_counter() - started
        save_history(run_dir / f"{mode}_history.csv", result.history,
                     guidance_enabled=(mode == "fixed"))
        labels = result.final_clustering.labels.detach().cpu().numpy()
        metrics = external_clustering_metrics(data.labels, labels)
        results[mode] = {
            "config": asdict(config),
            "final_silhouette": result.final_silhouette,
            "external_metrics": metrics,
            "gate_triggers": result.gate_triggers if mode == "fixed" else 0,
            "adjustments_applied": result.adjustments_applied if mode == "fixed" else 0,
            "training_seconds": result.ae_seconds + result.fcm_seconds + result.validation_seconds,
            "wall_seconds": wall_seconds,
        }
        print(f"{args.dataset} seed={args.seed} {mode}: "
              f"silhouette={result.final_silhouette}, ARI={metrics['ARI']:.4f}", flush=True)

    summary = {
        "dataset": data.name,
        "seed": args.seed,
        "n_samples": n_samples,
        "reference_size": reference_count,
        "threshold": THRESHOLD,
        "gamma": GAMMA,
        "guidance_schedule": "each gate check controls all epochs through the next check",
        "metric_schedule": "every_epoch" if args.metrics_every_epoch else "gate_checks_and_next_epoch",
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
