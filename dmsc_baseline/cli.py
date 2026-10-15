from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import load_dataset
from .metrics import external_clustering_metrics
from .trainer import DMSCConfig, DMSCTrainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the DMSC baseline and save its validation trajectory and final clusters."
    )
    parser.add_argument("--dataset", choices=("iris", "wine", "csv"), default="iris")
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--label-column", help="Optional CSV column used only for post-hoc metrics.")
    parser.add_argument("--n-clusters", type=int, help="Defaults to the label count for Iris/Wine or labeled CSVs.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--adjustment-gamma", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fuzzifier", type=float, default=2.0)
    parser.add_argument("--fcm-max-iter", type=int, default=50)
    parser.add_argument("--fcm-tolerance", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=int, help="Fixed random subset size; omit to score all samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda:0.")
    parser.add_argument("--output", type=Path, help="New output directory; defaults under runs/.")
    return parser


def _write_history(path: Path, history: list[dict]) -> None:
    fields = [
        "epoch",
        "reconstruction_loss",
        "ae_seconds",
        "evaluated",
        "silhouette",
        "gate_triggered",
        "adjustment_applied",
        "fcm_iterations",
        "fcm_objective",
        "fcm_seconds",
        "validation_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def _write_predictions(
    path: Path, labels: np.ndarray | None, clusters: np.ndarray, evaluation_indices: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["row_index", "cluster"]
        if labels is not None:
            fields.append("true_label")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, cluster in enumerate(clusters):
            row = {"row_index": index, "cluster": int(cluster)}
            if labels is not None:
                row["true_label"] = int(labels[index])
            writer.writerow(row)

    if evaluation_indices.size < len(clusters):
        with (path.parent / "evaluation_indices.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_index"])
            writer.writerows([[int(index)] for index in evaluation_indices])


def main() -> None:
    args = _parser().parse_args()
    load_started = time.perf_counter()
    dataset = load_dataset(args.dataset, args.csv_path, args.label_column)
    load_seconds = time.perf_counter() - load_started

    n_clusters = args.n_clusters
    if n_clusters is None and dataset.labels is not None:
        n_clusters = int(np.unique(dataset.labels).size)
    if n_clusters is None:
        raise SystemExit("--n-clusters is required for an unlabeled CSV.")

    config = DMSCConfig(
        n_clusters=n_clusters,
        epochs=args.epochs,
        evaluation_interval=args.evaluation_interval,
        silhouette_threshold=args.threshold,
        adjustment_gamma=args.adjustment_gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        fuzzifier=args.fuzzifier,
        fcm_max_iter=args.fcm_max_iter,
        fcm_tolerance=args.fcm_tolerance,
        evaluation_size=args.validation_size,
        seed=args.seed,
        device=args.device,
    )
    output = args.output or Path("runs") / f"dmsc-baseline-{dataset.name}-seed-{args.seed}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output path is not a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    trainer = DMSCTrainer(config)
    result = trainer.fit(dataset.features)

    predicted = result.final_clustering.labels.detach().cpu().numpy().astype(np.int64)
    final_metrics = (
        external_clustering_metrics(dataset.labels, predicted)
        if dataset.labels is not None
        else None
    )
    config_record = asdict(config)
    config_record["device"] = result.device
    summary = {
        "dataset": dataset.name,
        "n_samples": int(dataset.features.shape[0]),
        "n_features": int(dataset.features.shape[1]),
        "n_clusters": int(n_clusters),
        "seed": int(args.seed),
        "device": result.device,
        "label_encoding": dataset.label_values,
        "feature_names": dataset.feature_names,
        "config": config_record,
        "final_silhouette": result.final_silhouette,
        "final_fcm_iterations": result.final_clustering.iterations,
        "final_fcm_objective": result.final_clustering.objective,
        "gate_triggers": result.gate_triggers,
        "adjustments_applied": result.adjustments_applied,
        "external_metrics": final_metrics,
        "timing_seconds": {
            "data_loading": load_seconds,
            "autoencoder_training": result.ae_seconds,
            "fcm": result.fcm_seconds,
            "validation": result.validation_seconds,
            "compute": result.ae_seconds + result.fcm_seconds + result.validation_seconds,
        },
    }

    _write_history(output / "history.csv", result.history)
    _write_predictions(output / "predictions.csv", dataset.labels, predicted, result.evaluation_indices)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": config_record,
            "feature_min": dataset.minimum,
            "feature_max": dataset.maximum,
            "feature_names": dataset.feature_names,
        },
        output / "model.pt",
    )

    print(f"Completed {config.epochs} epochs on {dataset.name}; device={result.device}")
    print(f"Final silhouette: {result.final_silhouette}")
    print(f"FCM iterations: {result.final_clustering.iterations}")
    print(f"Silhouette gate triggers: {result.gate_triggers}; adjustments applied: {result.adjustments_applied}")
    if final_metrics is not None:
        print("External metrics: " + ", ".join(f"{key}={value:.4f}" for key, value in final_metrics.items()))
    print(f"Saved run artifacts to {output}")


if __name__ == "__main__":
    main()
