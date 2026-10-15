from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml, load_iris, load_wine


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
    name: str,
    csv_path: str | Path | None = None,
    label_column: str | None = None,
    *,
    data_home: str | Path | None = None,
) -> Dataset:
    """Load a built-in benchmark or a headered CSV and scale its features."""
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
    elif normalized_name in {"covertype", "forest_cover_type", "forest-covertype"}:
        raw = fetch_covtype(data_home=None if data_home is None else str(data_home))
        raw_features = np.asarray(raw.data, dtype=np.float32)
        raw_labels = np.asarray(raw.target)
        label_values = [str(item) for item in np.unique(raw_labels).tolist()]
        feature_names = list(
            getattr(raw, "feature_names", [f"feature_{index}" for index in range(raw_features.shape[1])])
        )
        if raw_features.shape[1] != 54:
            raise ValueError(f"Expected 54 Covertype features; got {raw_features.shape[1]}.")
        scaled, minimum, maximum = _scale_to_unit_interval(raw_features[:, :10])
        binary = raw_features[:, 10:]
        if not np.isin(binary, (0.0, 1.0)).all():
            raise ValueError("Covertype indicator features must be binary.")
        scaled_features = np.concatenate((scaled, binary.astype(np.float32, copy=False)), axis=1)
        all_minimum = np.concatenate((minimum, binary.min(axis=0).astype(np.float32)))
        all_maximum = np.concatenate((maximum, binary.max(axis=0).astype(np.float32)))
        labels = np.asarray(raw_labels, dtype=np.int64)
        return Dataset(
            name="covertype",
            features=scaled_features,
            labels=labels,
            label_values=label_values,
            feature_names=feature_names,
            minimum=all_minimum,
            maximum=all_maximum,
        )
    elif normalized_name in {"fashion_mnist", "fashion-mnist", "fashionmnist"}:
        raw = fetch_openml(
            name="Fashion-MNIST",
            version=1,
            as_frame=False,
            data_home=None if data_home is None else str(data_home),
        )
        raw_features = np.asarray(raw.data, dtype=np.float32)
        raw_labels = np.asarray(raw.target, dtype=str)
        labels, label_values = np.unique(raw_labels, return_inverse=True)
        if raw_features.ndim != 2 or raw_features.shape[1] != 784:
            raise ValueError(f"Expected flattened 784-pixel Fashion-MNIST rows; got {raw_features.shape}.")
        minimum = np.zeros(raw_features.shape[1], dtype=np.float32)
        maximum = np.full(raw_features.shape[1], 255.0, dtype=np.float32)
        scaled_features = raw_features / 255.0
        if not np.isfinite(scaled_features).all() or scaled_features.min() < 0 or scaled_features.max() > 1:
            raise ValueError("Fashion-MNIST pixels must be in [0, 255].")
        feature_names = [f"pixel_{index}" for index in range(raw_features.shape[1])]
        return Dataset(
            name="fashion_mnist",
            features=scaled_features.astype(np.float32, copy=False),
            labels=label_values.astype(np.int64, copy=False),
            label_values=[str(value) for value in labels.tolist()],
            feature_names=feature_names,
            minimum=minimum,
            maximum=maximum,
        )
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
        raise ValueError("Dataset must be one of: iris, wine, covertype, fashion_mnist, csv.")

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
