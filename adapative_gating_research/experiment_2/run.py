"""Run one dataset, seed, and guidance condition for Experiment 2."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dmsc_custom.data import load_dataset
from dmsc_custom.metrics import external_clustering_metrics
from dmsc_custom.trainer import DMSCConfig, DMSCTrainer

from .policies import CONDITIONS, GuidancePolicy

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"
DATASETS = ("iris", "wine", "covertype", "fashion_mnist")


def _write_history(path: Path, history: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in history for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--reference-size", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-home", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output", type=Path, default=RUNS)
    args = parser.parse_args()
    if args.epochs < 1 or args.evaluation_interval < 1 or args.reference_size < 2:
        parser.error("epochs and interval must be positive; reference size must be at least 2")
    if args.epochs % args.evaluation_interval:
        parser.error("--epochs must be divisible by --evaluation-interval")
    if not np.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        parser.error("--gamma must be finite and in [0, 1]")
    if args.condition != "no_guidance" and args.gamma == 0:
        parser.error("Guided conditions require --gamma > 0")

    gamma = 0.0 if args.condition == "no_guidance" else args.gamma
    run_dir = args.output / args.dataset / args.condition / f"gamma-{gamma:g}" / f"seed-{args.seed}"
    if (run_dir / "summary.json").exists():
        parser.error(f"Completed run already exists: {run_dir}")
    data = load_dataset(args.dataset, data_home=args.data_home)
    n_samples = len(data.features)
    n_clusters = int(np.unique(data.labels).size)
    reference_size = min(args.reference_size, n_samples)
    if reference_size <= n_clusters:
        parser.error("Reference size must exceed the number of clusters")
    reference = (
        np.arange(n_samples, dtype=np.int64) if reference_size == n_samples
        else np.sort(np.random.default_rng(2026).choice(
            n_samples, size=reference_size, replace=False,
        ))
    )

    config = DMSCConfig(
        n_clusters=n_clusters,
        epochs=args.epochs,
        evaluation_interval=args.evaluation_interval,
        evaluation_size=reference_size,
        silhouette_threshold=0.5,
        adjustment_gamma=gamma,
        gate_metric="reference",
        validation_metrics=("reference",),
        seed=args.seed,
        device=args.device,
    )
    policy = GuidancePolicy(args.condition, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "reference_indices.npy", reference)
    metric_rows: list[dict] = []

    def observe(epoch, embedding, clustering, record) -> None:
        external = external_clustering_metrics(data.labels, clustering.labels.detach().cpu().numpy())
        metric_rows.append({
            "epoch": epoch,
            "silhouette": record["silhouette"],
            "ACC": external["ACC"],
            "NMI": external["NMI"],
            "ARI": external["ARI"],
            "decision_active": record["gate_triggered"],
            "guidance_applied_this_epoch": record["adjustment_applied"],
            "effective_strength_this_epoch": record["guidance_strength"],
            "decision_seconds": record["decision_seconds"],
            "relative_threshold": record.get("decision_threshold"),
            "null_p": record.get("decision_null_p"),
            "null_max": record.get("decision_null_max"),
            "stability_ari": record.get("decision_stability_ari"),
            "mean_weight": record.get("decision_mean_weight"),
            "soft_weight": record.get("decision_soft_weight"),
        })
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)

    started = time.perf_counter()
    result = DMSCTrainer(config).fit(
        data.features,
        evaluation_indices=reference,
        guidance_decider=policy,
        epoch_observer=observe,
        epoch_observer_epochs=config.scheduled_evaluation_epochs,
    )
    wall_seconds = time.perf_counter() - started
    _write_history(run_dir / "history.csv", result.history)
    final_external = external_clustering_metrics(
        data.labels, result.final_clustering.labels.detach().cpu().numpy()
    )
    opened = [int(row["epoch"]) for row in result.history if row["gate_triggered"]]
    guided = [int(row["epoch"]) for row in result.history if row["adjustment_applied"]]
    checkpoints = {}
    for checkpoint_epoch in (200, args.epochs):
        if checkpoint_epoch > args.epochs or str(checkpoint_epoch) in checkpoints:
            continue
        metric = next((row for row in metric_rows if row["epoch"] == checkpoint_epoch), None)
        if metric is None:
            continue
        history_to_checkpoint = [row for row in result.history if row["epoch"] <= checkpoint_epoch]
        checkpoints[str(checkpoint_epoch)] = {
            "silhouette": metric["silhouette"],
            "external_metrics": {name: metric[name] for name in ("ACC", "NMI", "ARI")},
            "gate_openings": sum(bool(row["gate_triggered"]) for row in history_to_checkpoint)
            if args.condition not in {"confidence_weighted", "soft"} else None,
            "guided_epochs": sum(bool(row["adjustment_applied"]) for row in history_to_checkpoint),
            "first_guided_epoch": next(
                (int(row["epoch"]) for row in history_to_checkpoint if row["adjustment_applied"]), None
            ),
            "cumulative_effective_gamma": gamma * sum(
                float(row["guidance_strength"]) for row in history_to_checkpoint
            ),
        }
    summary = {
        "dataset": data.name,
        "condition": args.condition,
        "seed": args.seed,
        "n_samples": n_samples,
        "reference_size": reference_size,
        "base_gamma": gamma,
        "config": asdict(config),
        "policy": {
            "relative": "first 5 valid unguided scores; 75th percentile; floor 0",
            "null_calibrated": "19 label permutations; empirical p <= 0.05",
            "stability": "two 2%-SD latent perturbations; mean assignment ARI >= 0.9",
            "confidence_weighted": "positive local centroid silhouette times normalized maximum FCM membership",
            "soft": "sigmoid((silhouette - 0.5) / 0.05)",
        }.get(args.condition, args.condition),
        "guidance_schedule": "decision at each check applies from the next epoch through the next check",
        "gate_openings": len(opened) if args.condition not in {"confidence_weighted", "soft"} else None,
        "active_decisions": len(opened),
        "first_open_epoch": (
            opened[0] if opened and args.condition not in {"confidence_weighted", "soft"} else None
        ),
        "first_active_decision_epoch": opened[0] if opened else None,
        "guided_epochs": result.adjustments_applied,
        "first_guided_epoch": guided[0] if guided else None,
        "cumulative_guidance_strength": result.cumulative_guidance_strength,
        "cumulative_effective_gamma": config.adjustment_gamma * result.cumulative_guidance_strength,
        "final_silhouette": result.final_silhouette,
        "final_metrics": final_external,
        "checkpoints": checkpoints,
        "wall_seconds": wall_seconds,
        "decision_seconds": sum(float(row["decision_seconds"] or 0) for row in result.history),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{data.name} seed={args.seed} {args.condition}: ARI={final_external['ARI']:.4f}, "
          f"NMI={final_external['NMI']:.4f}, guided_epochs={result.adjustments_applied}; "
          f"saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
