"""Reproducible, bounded-memory reduction of the local headerless HIGGS CSV."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SAMPLE_SIZE = 1_000_000
SAMPLE_SEED = 2026
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "data" / "HIGGS.csv"


def load_higgs(
    source: Path, cache_home: Path, *, sample_size: int = SAMPLE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw sampled rows and provenance indices; serialize concurrent builders.

    Two streaming passes count rows, then parse only selected rows. The original
    file is never loaded into memory or modified. Sampling is independent of the
    model seed, without replacement, and preserves source order.
    """
    source = source.resolve()
    stat = source.stat()
    metadata = json.dumps({
        "version": 1, "source": str(source), "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sample_size": sample_size,
        "sample_seed": SAMPLE_SEED,
    }, sort_keys=True)
    cache_home.mkdir(parents=True, exist_ok=True)
    cache = cache_home / f"higgs-{sample_size}-seed-{SAMPLE_SEED}.npz"
    # Slurm array tasks share this lock and see only atomically completed caches.
    with cache.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["metadata"].item()) == metadata:
                    return saved["rows"], saved["source_indices"]
        print(f"Counting HIGGS rows in {source}", flush=True)
        with source.open("rb") as handle:
            n_rows = sum(1 for _ in handle)
        if n_rows < sample_size:
            raise ValueError(f"HIGGS needs {sample_size:,} rows; found {n_rows:,}.")
        indices = np.sort(np.random.default_rng(SAMPLE_SEED).choice(
            n_rows, size=sample_size, replace=False,
        ))

        def selected_rows():
            selected = 0
            with source.open("rb") as handle:
                for index, line in enumerate(handle):
                    if index == indices[selected]:
                        yield line
                        selected += 1
                        if selected == sample_size:
                            break

        print(f"Reading {sample_size:,} sampled HIGGS rows", flush=True)
        rows = np.loadtxt(selected_rows(), delimiter=",", dtype=np.float32, ndmin=2)
        if rows.shape != (sample_size, 29):
            raise ValueError(f"Expected HIGGS shape {(sample_size, 29)}; got {rows.shape}.")
        if not np.isfinite(rows).all() or not np.isin(rows[:, 0], (0, 1)).all():
            raise ValueError("HIGGS must have finite values and binary labels in column 0.")
        current = source.stat()
        if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("HIGGS source changed during subset preparation; retry.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_home, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, rows=rows, source_indices=indices, metadata=metadata)
            os.replace(temporary, cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Saved HIGGS subset and source indices to {cache}", flush=True)
        return rows, indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_SOURCE.parent / "cache")
    args = parser.parse_args()
    rows, _ = load_higgs(args.csv_path, args.data_home)
    print(f"HIGGS ready: {len(rows):,} rows, {rows.shape[1] - 1} features", flush=True)


if __name__ == "__main__":
    main()
