from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris, load_wine


@dataclass
class Dataset:
    name: str
    features: np.ndarray
    labels: np.ndarray | None
    label_values: list[str] | None
    feature_names: list[str]
    minimum: np.ndarray
    maximum: np.ndarray


def _scale_to_unit_interval(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Features must be a non-empty [n_samples, n_features] matrix.")
    if not np.isfinite(features).all():
        raise ValueError("Features must contain only finite numeric values.")
    minimum = features.min(axis=0)
    maximum = features.max(axis=0)
    span = maximum - minimum
    safe_span = np.where(span == 0, 1.0, span)
    scaled = ((features - minimum) / safe_span).astype(np.float32, copy=False)
    return scaled, minimum.astype(np.float32), maximum.astype(np.float32)


def load_dataset(
    name: str, csv_path: str | Path | None = None, label_column: str | None = None
) -> Dataset:
    """Load Iris, Wine, or a headered CSV with numeric feature columns."""
    normalized_name = name.lower()
    if normalized_name == "iris":
        raw = load_iris()
        raw_features = raw.data
        raw_labels = raw.target
        label_values = [str(item) for item in raw.target_names]
        feature_names = list(raw.feature_names)
    elif normalized_name == "wine":
        raw = load_wine()
        raw_features = raw.data
        raw_labels = raw.target
        label_values = [str(item) for item in raw.target_names]
        feature_names = list(raw.feature_names)
    elif normalized_name == "csv":
        if csv_path is None:
            raise ValueError("--csv-path is required when --dataset csv is selected.")
        path = Path(csv_path)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV file is empty: {path}") from exc
            if not header:
                raise ValueError("CSV must have a header row.")
            if label_column is not None and label_column not in header:
                raise ValueError(f"Label column {label_column!r} is not in the CSV header.")
            label_index = header.index(label_column) if label_column is not None else None
            feature_indices = [i for i in range(len(header)) if i != label_index]
            feature_names = [header[i] for i in feature_indices]
            feature_rows: list[list[float]] = []
            raw_label_values: list[str] = []
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(f"CSV row {line_number} has {len(row)} fields; expected {len(header)}.")
                try:
                    feature_rows.append([float(row[i]) for i in feature_indices])
                except ValueError as exc:
                    raise ValueError(
                        f"CSV row {line_number} has a non-numeric feature; encode categorical features first."
                    ) from exc
                if label_index is not None:
                    raw_label_values.append(row[label_index])
        if not feature_rows:
            raise ValueError("CSV has a header but no data rows.")
        raw_features = np.asarray(feature_rows, dtype=np.float32)
        if label_index is None:
            raw_labels = None
            label_values = None
        else:
            label_values, raw_labels = np.unique(
                np.asarray(raw_label_values, dtype=str), return_inverse=True
            )
            label_values = [str(value) for value in label_values.tolist()]
    else:
        raise ValueError("Dataset must be one of: iris, wine, csv.")

    scaled, minimum, maximum = _scale_to_unit_interval(raw_features)
    labels = None if raw_labels is None else np.asarray(raw_labels, dtype=np.int64)
    return Dataset(
        name=normalized_name,
        features=scaled,
        labels=labels,
        label_values=label_values,
        feature_names=feature_names,
        minimum=minimum,
        maximum=maximum,
    )
