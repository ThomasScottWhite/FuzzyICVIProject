from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dmsc_custom.data import load_dataset
from dmsc_custom.metrics import exact_silhouette, simplified_silhouette
from dmsc_custom.trainer import DMSCConfig, DMSCTrainer
from experiments.common import fixed_validation_indices, run_one
from experiments.experiment_1.main import schedules
from experiments.experiment_3.main import _decision_rows


class SilhouetteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = torch.tensor(
            [[-3.0, 0.0], [-2.8, 0.0], [3.0, 0.0], [3.2, 0.0]], dtype=torch.float32
        )
        self.labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        self.centers = torch.tensor([[-2.9, 0.0], [3.1, 0.0]], dtype=torch.float32)

    def test_exact_and_simplified_silhouettes_are_high_for_separated_groups(self) -> None:
        exact = exact_silhouette(self.embeddings, self.labels, block_size=2)
        simplified = simplified_silhouette(self.embeddings, self.labels, self.centers, block_size=2)
        self.assertIsNotNone(exact)
        self.assertIsNotNone(simplified)
        self.assertGreater(exact, 0.9)
        self.assertGreater(simplified, 0.9)

    def test_silhouettes_are_undefined_for_one_or_n_singleton_clusters(self) -> None:
        one_cluster = torch.zeros(4, dtype=torch.long)
        singleton_clusters = torch.arange(4)
        self.assertIsNone(exact_silhouette(self.embeddings, one_cluster))
        self.assertIsNone(simplified_silhouette(self.embeddings, one_cluster, self.centers))
        self.assertIsNone(exact_silhouette(self.embeddings, singleton_clusters))
        self.assertIsNone(simplified_silhouette(self.embeddings, singleton_clusters, self.centers))


class DatasetLoaderTests(unittest.TestCase):
    @patch("dmsc_custom.data.fetch_covtype")
    def test_covertype_preserves_binary_indicators(self, fetch) -> None:
        continuous = np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 10))
        binary = np.zeros((4, 44), dtype=np.float32)
        binary[:, 0] = [0, 0, 1, 1]
        binary[:, 1] = 1
        features = np.concatenate((continuous, binary), axis=1)
        fetch.return_value = SimpleNamespace(
            data=features,
            target=np.array([1, 1, 2, 2]),
            feature_names=[f"feature_{index}" for index in range(54)],
        )

        dataset = load_dataset("covertype")

        np.testing.assert_array_equal(dataset.features[:, 10:], binary)
        np.testing.assert_allclose(dataset.features[:, 0], [0.0, 1 / 3, 2 / 3, 1.0])
        self.assertEqual(dataset.name, "covertype")
        self.assertEqual(dataset.labels.tolist(), [1, 1, 2, 2])

    @patch("dmsc_custom.data.fetch_openml")
    def test_fashion_mnist_scales_pixels_and_encodes_labels(self, fetch) -> None:
        features = np.zeros((4, 784), dtype=np.uint8)
        features[1, :] = 255
        features[2, :] = 128
        fetch.return_value = SimpleNamespace(data=features, target=np.array(["2", "0", "1", "0"]))

        dataset = load_dataset("fashion_mnist")

        self.assertEqual(dataset.features.shape, (4, 784))
        np.testing.assert_allclose(dataset.features[:, 0], [0.0, 1.0, 128 / 255, 0.0])
        self.assertEqual(dataset.labels.tolist(), [2, 0, 1, 0])
        self.assertEqual(dataset.label_values, ["0", "1", "2"])


class ScheduleAndTrainerTests(unittest.TestCase):
    def test_experiment_1_schedules_have_expected_evaluations(self) -> None:
        by_name = {schedule.name: schedule for schedule in schedules()}
        self.assertEqual(len(by_name["every_1"].epochs), 200)
        self.assertEqual(len(by_name["every_5"].epochs), 40)
        self.assertEqual(len(by_name["every_10"].epochs), 20)
        self.assertEqual(len(by_name["every_20"].epochs), 10)
        self.assertEqual(len(by_name["front_loaded"].epochs), 20)
        self.assertEqual(len(by_name["back_loaded"].epochs), 20)
        self.assertEqual(by_name["front_loaded"].epochs[-1], 200)
        self.assertEqual(by_name["back_loaded"].epochs[-1], 200)

    def test_shadow_metrics_do_not_change_reference_controlled_training(self) -> None:
        rng = np.random.default_rng(4)
        groups = [
            rng.normal(0.12, 0.025, size=(12, 4)),
            rng.normal(0.50, 0.025, size=(12, 4)),
            rng.normal(0.88, 0.025, size=(12, 4)),
        ]
        features = np.clip(np.concatenate(groups).astype(np.float32), 0.0, 1.0)
        evaluation_indices = np.arange(36, dtype=np.int64)
        sampled_indices = np.arange(0, 36, 2, dtype=np.int64)
        common = {
            "n_clusters": 3,
            "epochs": 2,
            "batch_size": 12,
            "evaluation_epochs": (1, 2),
            "fcm_max_iter": 5,
            "silhouette_threshold": 0.5,
            "seed": 71,
            "device": "cpu",
        }
        reference_result = DMSCTrainer(
            DMSCConfig(**common, gate_metric="reference", validation_metrics=("reference",))
        ).fit(features, evaluation_indices=evaluation_indices, sampled_indices=sampled_indices)
        shadow_result = DMSCTrainer(
            DMSCConfig(
                **common,
                gate_metric="reference",
                validation_metrics=("reference", "sampled", "simplified"),
            )
        ).fit(features, evaluation_indices=evaluation_indices, sampled_indices=sampled_indices)

        self.assertEqual(
            [row["gate_triggered"] for row in reference_result.history],
            [row["gate_triggered"] for row in shadow_result.history],
        )
        self.assertEqual(
            [row["adjustment_applied"] for row in reference_result.history],
            [row["adjustment_applied"] for row in shadow_result.history],
        )
        for name, tensor in reference_result.model.state_dict().items():
            torch.testing.assert_close(tensor, shadow_result.model.state_dict()[name])
        self.assertIn("sampled", shadow_result.metric_seconds)
        self.assertIn("simplified", shadow_result.peak_memory_bytes_by_metric)

    def test_sampled_controller_drives_gate_and_next_epoch_adjustment(self) -> None:
        rng = np.random.default_rng(19)
        features = np.clip(
            np.concatenate(
                [rng.normal(center, 0.02, size=(10, 3)) for center in (0.1, 0.5, 0.9)]
            ).astype(np.float32),
            0.0,
            1.0,
        )
        result = DMSCTrainer(
            DMSCConfig(
                n_clusters=3,
                epochs=2,
                batch_size=10,
                evaluation_epochs=(1, 2),
                gate_metric="sampled",
                validation_metrics=("sampled",),
                fcm_max_iter=5,
                silhouette_threshold=-1.0,
                seed=9,
                device="cpu",
            )
        ).fit(
            features,
            evaluation_indices=np.arange(30),
            sampled_indices=np.arange(0, 30, 2),
        )
        first, second = result.history
        self.assertEqual(first["gate_metric"], "sampled")
        self.assertEqual(first["gate_triggered"], first["gate_sampled"])
        self.assertEqual(second["adjustment_applied"], first["gate_triggered"])
        self.assertIsNone(result.final_silhouette)
        self.assertEqual(result.final_gate_score, result.final_scores["sampled"])

    def test_fixed_validation_indices_are_reused_and_nested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_reference, first_sampled, _ = fixed_validation_indices("sample", 20_100, root)
            second_reference, second_sampled, _ = fixed_validation_indices("sample", 20_100, root)
        np.testing.assert_array_equal(first_reference, second_reference)
        np.testing.assert_array_equal(first_sampled, second_sampled)
        self.assertEqual(first_reference.size, 20_000)
        self.assertEqual(first_sampled.size, 2_000)
        self.assertTrue(set(first_sampled).issubset(set(first_reference)))

    @patch("experiments.common.load_dataset")
    def test_run_output_is_complete_and_resumable(self, load_dataset_mock) -> None:
        rng = np.random.default_rng(91)
        features = np.clip(
            np.concatenate(
                [rng.normal(center, 0.02, size=(6, 3)) for center in (0.1, 0.5, 0.9)]
            ).astype(np.float32),
            0.0,
            1.0,
        )
        dataset = SimpleNamespace(
            name="synthetic",
            features=features,
            labels=np.repeat(np.arange(3), 6),
            label_values=["0", "1", "2"],
            feature_names=["x", "y", "z"],
            minimum=np.zeros(3, dtype=np.float32),
            maximum=np.ones(3, dtype=np.float32),
        )
        load_dataset_mock.return_value = dataset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            run_dir = root / "synthetic" / "reference_shadow" / "seed-3"
            config = DMSCConfig(
                n_clusters=3,
                epochs=1,
                batch_size=6,
                evaluation_epochs=(1,),
                fcm_max_iter=3,
                seed=3,
                device="cpu",
            )
            first, skipped = run_one(
                dataset_name="synthetic",
                data_home=root / "cache",
                run_dir=run_dir,
                run_key="synthetic/reference_shadow/seed-3",
                config=config,
                resume=True,
            )
            self.assertFalse(skipped)
            self.assertTrue((run_dir / "model.pt").is_file())
            self.assertTrue((run_dir / "history.csv").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertAlmostEqual(
                first["timing_seconds"]["wall"],
                first["timing_seconds"]["compute"] + first["timing_seconds"]["io"],
            )
            load_dataset_mock.reset_mock(side_effect=True)
            second, skipped = run_one(
                dataset_name="synthetic",
                data_home=root / "cache",
                run_dir=run_dir,
                run_key="synthetic/reference_shadow/seed-3",
                config=config,
                resume=True,
            )
            self.assertTrue(skipped)
            load_dataset_mock.assert_not_called()
            self.assertEqual(second["run_fingerprint"], first["run_fingerprint"])

    def test_decision_summary_uses_standard_false_and_missed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "covertype" / "reference_shadow" / "seed-42"
            run_dir.mkdir(parents=True)
            with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                handle.write(
                    "evaluated,silhouette_reference,silhouette_sampled,silhouette_simplified\n"
                    "True,0.7,0.3,0.8\n"
                    "True,0.2,0.7,0.1\n"
                    "True,,0.9,0.4\n"
                )
            summary = {
                "dataset": "covertype",
                "mode": "reference_shadow",
                "seed": 42,
            }

            rows = _decision_rows([summary], root)
            sampled = next(
                row for row in rows
                if row["alternative"] == "sampled" and row["threshold"] == 0.5
            )

        self.assertEqual(sampled["false_trigger"], 1)
        self.assertEqual(sampled["missed_trigger"], 1)
        self.assertEqual(sampled["undefined_evaluations"], 1)
        self.assertEqual(sampled["evaluations"], 2)


if __name__ == "__main__":
    unittest.main()
