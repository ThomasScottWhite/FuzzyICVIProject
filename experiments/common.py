from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dmsc_custom.data import Dataset, load_dataset
from dmsc_custom.metrics import external_clustering_metrics
from dmsc_custom.trainer import DMSCConfig, DMSCTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_HOME = PROJECT_ROOT / "data" / "cache"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
REFERENCE_SIZE = 20_000
SAMPLED_SIZE = 2_000
VALIDATION_SELECTION_SEED = 42
RUN_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def fixed_validation_indices(
    dataset_name: str,
    n_samples: int,
    experiment_root: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load or create the one reference/sample selection shared by every run."""
    path = experiment_root / dataset_name / "validation_indices.npz"
    started = time.perf_counter()
    if path.exists():
        with np.load(path) as saved:
            reference = saved["reference"].astype(np.int64, copy=True)
            sampled = saved["sampled"].astype(np.int64, copy=True)
        if reference.size != min(REFERENCE_SIZE, n_samples):
            raise ValueError(f"Saved reference index count does not match dataset size: {path}")
        if sampled.size != min(SAMPLED_SIZE, reference.size):
            raise ValueError(f"Saved sampled index count does not match dataset size: {path}")
        if np.any(reference < 0) or np.any(reference >= n_samples):
            raise ValueError(f"Saved reference indices do not match the current data: {path}")
        if np.any(sampled < 0) or np.any(sampled >= n_samples):
            raise ValueError(f"Saved sampled indices do not match the current data: {path}")
        if not set(sampled.tolist()).issubset(set(reference.tolist())):
            raise ValueError(f"Saved sampled indices must be a subset of reference indices: {path}")
        return reference, sampled, time.perf_counter() - started

    reference_count = min(REFERENCE_SIZE, n_samples)
    sample_count = min(SAMPLED_SIZE, reference_count)
    if reference_count < 2:
        raise ValueError("Datasets must contain at least two observations.")
    rng = np.random.default_rng(VALIDATION_SELECTION_SEED)
    reference = np.sort(rng.choice(n_samples, reference_count, replace=False).astype(np.int64))
    sampled = np.sort(rng.choice(reference, sample_count, replace=False).astype(np.int64))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, reference=reference, sampled=sampled)
    os.replace(temporary, path)
    return reference, sampled, time.perf_counter() - started


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in history:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def _write_json_atomic(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(content), handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run_fingerprint(run_key: str, config: DMSCConfig) -> str:
    payload = {
        "schema": RUN_SCHEMA_VERSION,
        "run_key": run_key,
        "config": _jsonable(asdict(config)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_complete(run_dir: Path, fingerprint: str) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not (summary_path.is_file() and (run_dir / "history.csv").is_file() and (run_dir / "model.pt").is_file()):
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("status") == "complete" and summary.get("run_fingerprint") == fingerprint:
        return summary
    return None


def run_one(
    *,
    dataset_name: str,
    data_home: Path,
    run_dir: Path,
    run_key: str,
    config: DMSCConfig,
    resume: bool,
    metadata: dict[str, Any] | None = None,
    progress_desc: str | None = None,
    report_device: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Train one configuration, atomically mark it complete, and save its checkpoint."""
    fingerprint = run_fingerprint(run_key, config)
    if resume:
        completed = _is_complete(run_dir, fingerprint)
        if completed is not None:
            return completed, True

    load_started = time.perf_counter()
    dataset: Dataset = load_dataset(dataset_name, data_home=data_home)
    data_loading_seconds = time.perf_counter() - load_started
    n_clusters = int(np.unique(dataset.labels).size) if dataset.labels is not None else config.n_clusters
    if n_clusters != config.n_clusters:
        raise ValueError(
            f"Configured {config.n_clusters} clusters for {dataset.name}, "
            f"but the benchmark has {n_clusters} labels."
        )

    reference_indices, sampled_indices, index_io_seconds = fixed_validation_indices(
        dataset.name,
        int(dataset.features.shape[0]),
        run_dir.parent.parent.parent,
    )
    trainer = DMSCTrainer(config)
    result = trainer.fit(
        dataset.features,
        evaluation_indices=reference_indices,
        sampled_indices=sampled_indices,
        progress_desc=progress_desc,
        report_device=report_device,
    )
    predicted = result.final_clustering.labels.detach().cpu().numpy().astype(np.int64)
    final_metrics = (
        external_clustering_metrics(dataset.labels, predicted)
        if dataset.labels is not None
        else None
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"
    checkpoint_temporary = run_dir / "model.pt.tmp"
    checkpoint_started = time.perf_counter()
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": _jsonable(asdict(config)),
            "feature_min": dataset.minimum,
            "feature_max": dataset.maximum,
            "feature_names": dataset.feature_names,
        },
        checkpoint_temporary,
    )
    os.replace(checkpoint_temporary, checkpoint_path)
    checkpoint_seconds = time.perf_counter() - checkpoint_started

    timing = {
        "data_loading": data_loading_seconds,
        "validation_index_io": index_io_seconds,
        "checkpoint_saving": checkpoint_seconds,
        "io": data_loading_seconds + index_io_seconds + checkpoint_seconds,
        "autoencoder_training": result.ae_seconds,
        "fcm": result.fcm_seconds,
        "validation": result.validation_seconds,
        "compute": result.ae_seconds + result.fcm_seconds + result.validation_seconds,
        "wall": (
            result.ae_seconds
            + result.fcm_seconds
            + result.validation_seconds
            + data_loading_seconds
            + index_io_seconds
            + checkpoint_seconds
        ),
    }
    summary: dict[str, Any] = {
        "status": "complete",
        "schema_version": RUN_SCHEMA_VERSION,
        "run_key": run_key,
        "run_fingerprint": fingerprint,
        "experiment_metadata": metadata or {},
        "dataset": dataset.name,
        "n_samples": int(dataset.features.shape[0]),
        "n_features": int(dataset.features.shape[1]),
        "n_clusters": n_clusters,
        "seed": int(config.seed),
        "device": result.device,
        "label_encoding": dataset.label_values,
        "config": _jsonable(asdict(config)),
        "final_silhouette": result.final_silhouette,
        "final_gate_score": result.final_gate_score,
        "final_validation_scores": result.final_scores,
        "final_fcm_iterations": result.final_clustering.iterations,
        "final_fcm_objective": result.final_clustering.objective,
        "gate_triggers": result.gate_triggers,
        "adjustments_applied": result.adjustments_applied,
        "external_metrics": final_metrics,
        "validation_metric_seconds": result.metric_seconds,
        "peak_memory_bytes_by_metric": result.peak_memory_bytes_by_metric,
        "memory_measurement": "CUDA peak allocated bytes on CUDA; process peak RSS on CPU.",
        "timing_seconds": timing,
        "reference_indices_file": str(
            run_dir.parent.parent.parent / dataset.name / "validation_indices.npz"
        ),
        "history_file": "history.csv",
        "checkpoint_file": "model.pt",
    }
    _write_history(run_dir / "history.csv", result.history)
    _write_json_atomic(run_dir / "summary.json", summary)
    return summary, False


def load_history(run_dir: Path) -> list[dict[str, str | None]]:
    with (run_dir / "history.csv").open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(_jsonable(value), sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def write_aggregate_json(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_json_atomic(path, {"runs": rows, "run_count": len(rows)})


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "median": None, "iqr": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    }


def paired_bootstrap_interval(
    deltas: list[float], *, seed: int = 42, samples: int = 10_000
) -> tuple[float | None, float | None]:
    if not deltas:
        return None, None
    array = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
