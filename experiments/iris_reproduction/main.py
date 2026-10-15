"""Run three paired DMSC Iris trials and save their validation-score histories.

From the repository root, run:

    python -m experiments.iris_reproduction.main
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dmsc_custom.data import load_dataset
from dmsc_custom.metrics import external_clustering_metrics
from dmsc_custom.trainer import DMSCConfig, DMSCTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "runs" / "iris_reproduction"
SEED = 42
EPOCHS = 200
EVALUATION_INTERVAL = 10
GATE_THRESHOLD = 0.5
SAMPLE_SIZE = 75
VALIDATION_METHODS = ("reference", "sampled", "simplified")


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields = list(history[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("iris")

    # Iris is small enough to use every row for the reference score. Keep a
    # fixed half-sized subset for the sampled score, shared by all three runs.
    indices_path = OUTPUT / "validation_indices.npz"
    if indices_path.exists():
        with np.load(indices_path) as saved:
            reference_indices = saved["reference"].astype(np.int64, copy=True)
            sampled_indices = saved["sampled"].astype(np.int64, copy=True)
    else:
        reference_indices = np.arange(dataset.features.shape[0], dtype=np.int64)
        sampled_indices = np.sort(
            np.random.default_rng(SEED).choice(
                reference_indices, size=min(SAMPLE_SIZE, reference_indices.size), replace=False
            )
        )
        np.savez_compressed(
            indices_path, reference=reference_indices, sampled=sampled_indices
        )

    if sampled_indices.size >= reference_indices.size:
        raise ValueError("The sampled validation set must be smaller than the reference set.")
    if not set(sampled_indices.tolist()).issubset(set(reference_indices.tolist())):
        raise ValueError("The sampled validation indices must be inside the reference set.")

    # Labels are not passed into the trainer; they set k and score the final
    # clustering only after training has finished.
    n_clusters = int(np.unique(dataset.labels).size)
    summaries: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    device_reported = False

    for gate_metric in VALIDATION_METHODS:
        run_dir = OUTPUT / gate_metric
        run_dir.mkdir(parents=True, exist_ok=True)
        config = DMSCConfig(
            n_clusters=n_clusters,
            epochs=EPOCHS,
            evaluation_interval=EVALUATION_INTERVAL,
            gate_metric=gate_metric,
            # Record all three scores in every run so their trajectories can
            # be compared while each selected score controls its own gate.
            validation_metrics=VALIDATION_METHODS,
            silhouette_threshold=GATE_THRESHOLD,
            seed=SEED,
            device="auto",
        )

        started = time.perf_counter()
        result = DMSCTrainer(config).fit(
            dataset.features,
            evaluation_indices=reference_indices,
            sampled_indices=sampled_indices,
            progress_desc=f"Iris / {gate_metric} gate",
            report_device=not device_reported,
        )
        device_reported = True

        predicted = result.final_clustering.labels.detach().cpu().numpy()
        summary = {
            "dataset": "iris",
            "gate_metric": gate_metric,
            "seed": SEED,
            "n_samples": int(dataset.features.shape[0]),
            "n_features": int(dataset.features.shape[1]),
            "n_clusters": n_clusters,
            "reference_size": int(reference_indices.size),
            "sampled_size": int(sampled_indices.size),
            "threshold": GATE_THRESHOLD,
            "config": asdict(config),
            "device": result.device,
            "gate_triggers": result.gate_triggers,
            "adjustments_applied": result.adjustments_applied,
            "final_validation_scores": result.final_scores,
            "external_metrics": external_clustering_metrics(dataset.labels, predicted),
            "validation_metric_seconds": result.metric_seconds,
            "peak_memory_bytes_by_metric": result.peak_memory_bytes_by_metric,
            "timing_seconds": {
                "autoencoder_training": result.ae_seconds,
                "fcm": result.fcm_seconds,
                "validation": result.validation_seconds,
                "wall": time.perf_counter() - started,
            },
        }

        history = [dict(row) for row in result.history]
        histories[gate_metric] = history
        _write_history(run_dir / "history.csv", history)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        torch.save(
            {"state_dict": result.model.state_dict(), "config": asdict(config)},
            run_dir / "model.pt",
        )
        summaries.append(summary)
        tqdm.write(
            f"{gate_metric} gate: triggers={result.gate_triggers}, "
            f"adjustments={result.adjustments_applied}, "
            f"final scores={result.final_scores}"
        )

    # Plot all three validation scores for each gate controller. Red markers
    # identify evaluations where that panel's controller activated the gate.
    colors = {"reference": "#2864a5", "sampled": "#e07a16", "simplified": "#29946b"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, sharey=True, layout="constrained")
    for axis, gate_metric in zip(axes, VALIDATION_METHODS, strict=True):
        evaluations = [row for row in histories[gate_metric] if row["evaluated"]]
        epochs = np.asarray([int(row["epoch"]) for row in evaluations])
        for metric in VALIDATION_METHODS:
            scores = np.asarray(
                [np.nan if row[f"silhouette_{metric}"] is None else row[f"silhouette_{metric}"]
                 for row in evaluations],
                dtype=np.float64,
            )
            axis.plot(epochs, scores, marker="o", markersize=3, linewidth=1.4,
                      color=colors[metric], label=f"{metric.title()} score")
            if metric == gate_metric:
                triggered = np.asarray([bool(row["gate_triggered"]) for row in evaluations])
                axis.scatter(epochs[triggered], scores[triggered], marker="D", s=34,
                             color="#c83232", edgecolor="white", linewidth=0.5,
                             label="Gate activated", zorder=4)
        axis.axhline(GATE_THRESHOLD, color="#333333", linestyle="--", linewidth=1,
                     label=f"Gate threshold ({GATE_THRESHOLD:g})")
        axis.set_title(f"{gate_metric.title()} score controls the gate")
        axis.set_ylabel("Silhouette")
        axis.set_ylim(-1.05, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, ncol=2, fontsize=8, loc="best")
    axes[-1].set_xlabel("Training epoch")
    fig.suptitle("DMSC on Iris: reference, sampled, and simplified validation")
    fig.savefig(OUTPUT / "silhouette_comparison.png", dpi=300)
    plt.close(fig)

    (OUTPUT / "results.json").write_text(
        json.dumps({"runs": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "gate_metric", "seed", "device", "gate_triggers", "adjustments_applied",
            "reference_final", "sampled_final", "simplified_final", "ACC", "NMI", "ARI",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            scores = summary["final_validation_scores"]
            metrics = summary["external_metrics"]
            writer.writerow({
                "gate_metric": summary["gate_metric"],
                "seed": summary["seed"],
                "device": summary["device"],
                "gate_triggers": summary["gate_triggers"],
                "adjustments_applied": summary["adjustments_applied"],
                "reference_final": scores.get("reference"),
                "sampled_final": scores.get("sampled"),
                "simplified_final": scores.get("simplified"),
                **metrics,
            })

    print(f"Iris reproduction complete. Results and graph saved in: {OUTPUT}")


if __name__ == "__main__":
    main()
