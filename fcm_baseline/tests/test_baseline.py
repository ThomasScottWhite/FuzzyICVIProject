"""Unit tests for the baseline package.  Run with:  pytest -q

These tests use only Iris/Wine and synthetic data, so they need no internet access.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from fcm_baseline.data import evaluation_subset_indices, load_dataset
from fcm_baseline.fcm import FuzzyCMeans
from fcm_baseline.metrics import clustering_accuracy, evaluate
from fcm_baseline import run_baseline


def test_accuracy_is_permutation_invariant():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert clustering_accuracy(y, np.array([2, 2, 0, 0, 1, 1])) == 1.0
    assert clustering_accuracy(y, np.array([0, 0, 1, 1, 2, 1])) == pytest.approx(5 / 6)


@pytest.mark.parametrize("init", ["two_stage", "random"])
def test_fcm_recovers_separated_blobs(init):
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.4, random_state=0)
    res = FuzzyCMeans(3, init=init, seed=1).fit(X.astype(np.float32))
    assert clustering_accuracy(y, res.labels) > 0.99
    assert res.converged


def test_memberships_are_valid_distributions():
    X, _ = make_blobs(n_samples=200, centers=4, random_state=1)
    res = FuzzyCMeans(4, seed=0).fit(X.astype(np.float32))
    assert res.memberships.shape == (200, 4)
    np.testing.assert_allclose(res.memberships.sum(axis=1), 1.0, atol=1e-9)
    assert res.memberships.min() >= 0.0


def test_fcm_is_deterministic_and_chunk_invariant():
    X, _ = make_blobs(n_samples=500, centers=3, random_state=2)
    X = X.astype(np.float32)
    a = FuzzyCMeans(3, seed=5, chunk_size=64).fit(X)
    b = FuzzyCMeans(3, seed=5, chunk_size=10_000).fit(X)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.centers, b.centers, atol=1e-8)


def test_fcm_handles_point_on_centre():
    # Duplicate points make a centre coincide with data; must not produce NaN/inf.
    X = np.vstack([np.zeros((10, 2)), np.ones((10, 2))]).astype(np.float32)
    res = FuzzyCMeans(2, seed=0).fit(X)
    assert np.isfinite(res.memberships).all()


@pytest.mark.parametrize("name,shape,k", [("iris", (150, 4), 3), ("wine", (178, 13), 3)])
def test_small_datasets(name, shape, k, tmp_path):
    ds = load_dataset(name, str(tmp_path))
    assert ds.X.shape == shape and ds.n_clusters == k
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0


def test_subset_indices():
    assert len(evaluation_subset_indices(150)) == 150          # small: use everything
    a = evaluation_subset_indices(100_000)
    assert len(a) == 5000 and len(set(a)) == 5000
    np.testing.assert_array_equal(a, evaluation_subset_indices(100_000))  # fixed seed


def test_iris_reasonable_quality(tmp_path):
    ds = load_dataset("iris", str(tmp_path))
    res = FuzzyCMeans(3, seed=42).fit(ds.X)
    scores = evaluate(ds.X, ds.y, res.labels, np.arange(150))
    assert scores["acc"] > 0.8 and scores["ari"] > 0.6


def test_runner_end_to_end_and_resume(tmp_path):
    argv = ["--datasets", "iris", "--seeds", "42", "43", "--output-dir", str(tmp_path / "out"),
            "--data-dir", str(tmp_path / "d")]
    run_baseline.main(argv)
    csv_path = tmp_path / "out" / "results_iris.csv"
    n_rows = len(csv_path.read_text().strip().splitlines()) - 1
    assert n_rows == 2 * 2 * 2 * 1  # seeds x methods x spaces
    assert len(list((tmp_path / "out" / "labels").glob("*.npy"))) == n_rows
    run_baseline.main(argv)  # resume: nothing recomputed, no duplicate rows
    assert len(csv_path.read_text().strip().splitlines()) - 1 == n_rows
