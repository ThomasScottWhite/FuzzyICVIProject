"""HIGGS sampling, cache provenance, and feature/label separation checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dmsc_custom.data import load_dataset
from dmsc_custom.higgs import load_higgs


class HiggsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "HIGGS.csv"
        self.cache = self.root / "cache"
        self.rows = np.arange(100 * 29, dtype=np.float32).reshape(100, 29)
        self.rows[:, 0] = np.arange(100) % 2
        np.savetxt(self.source, self.rows, delimiter=",")

    def test_reproducible_sample_and_source_indices(self):
        rows, indices = load_higgs(self.source, self.cache, sample_size=30)
        other, other_indices = load_higgs(self.source, self.root / "other", sample_size=30)
        self.assertEqual(len(np.unique(indices)), 30)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertFalse(np.array_equal(indices, np.arange(30)))
        np.testing.assert_array_equal(rows, self.rows[indices])
        np.testing.assert_array_equal(rows, other)
        np.testing.assert_array_equal(indices, other_indices)
        # A cache hit must avoid parsing the large source again.
        with patch("dmsc_custom.higgs.np.loadtxt", side_effect=AssertionError("cache miss")):
            cached, cached_indices = load_higgs(self.source, self.cache, sample_size=30)
        np.testing.assert_array_equal(cached, rows)
        np.testing.assert_array_equal(cached_indices, indices)

    def test_changed_source_invalidates_cache(self):
        rows, indices = load_higgs(self.source, self.cache, sample_size=30)
        self.rows[:, 1:] += 10000
        np.savetxt(self.source, self.rows, delimiter=",")
        changed, changed_indices = load_higgs(self.source, self.cache, sample_size=30)
        np.testing.assert_array_equal(changed_indices, indices)
        np.testing.assert_array_equal(changed[:, 1:], rows[:, 1:] + 10000)

    def test_short_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "needs 101 rows; found 100"):
            load_higgs(self.source, self.cache, sample_size=101)

    def test_invalid_labels_and_width_are_rejected(self):
        self.rows[:, 0] = 2
        np.savetxt(self.source, self.rows, delimiter=",")
        with self.assertRaisesRegex(ValueError, "binary labels"):
            load_higgs(self.source, self.cache, sample_size=30)
        np.savetxt(self.source, self.rows[:, :-1], delimiter=",")
        with self.assertRaisesRegex(ValueError, "Expected HIGGS shape"):
            load_higgs(self.source, self.cache, sample_size=30)

    def test_loader_excludes_labels_and_scales_features(self):
        def small_subset(source, cache):
            return load_higgs(source, cache, sample_size=30)

        with patch("dmsc_custom.higgs.load_higgs", side_effect=small_subset):
            data = load_dataset("higgs", csv_path=self.source, data_home=self.cache)
        self.assertEqual(data.name, "higgs")
        self.assertEqual(data.features.shape, (30, 28))
        self.assertEqual(data.features.dtype, np.float32)
        self.assertEqual(len(data.feature_names), 28)
        np.testing.assert_array_equal(np.unique(data.labels), [0, 1])
        np.testing.assert_array_equal(data.features.min(axis=0), np.zeros(28))
        np.testing.assert_array_equal(data.features.max(axis=0), np.ones(28))


if __name__ == "__main__":
    unittest.main()
