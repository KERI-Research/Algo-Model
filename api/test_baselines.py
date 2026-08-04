"""Tests for the unsupervised baselines (PCA reconstruction, Isolation Forest)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from baselines import run_baselines  # noqa: E402
from self_supervised import SSLConfig, train_self_supervised  # noqa: E402
from test_data_integrity import synthetic_frame  # noqa: E402


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.dataset = root / "synthetic_cohort.csv"
        frame = synthetic_frame(rows=900)
        frame.to_csv(cls.dataset, index=False)
        cls.config = SSLConfig(
            epochs=2,
            batch_size=64,
            hidden_dim=32,
            latent_dim=8,
            patience=2,
            max_train_rows=400,
            backend="numpy",
            minimum_adult_rows=100,
            checkpoint_every=0,
            run_label="unit-test",
        )
        cls.artifact = root / "artifact"
        train_self_supervised(frame, cls.artifact, cls.config)
        cls.results = run_baselines(
            cls.dataset,
            root / "benchmarks",
            config=cls.config,
            ssl_artifact_dir=cls.artifact,
            max_train_rows=400,
            pca_components=8,
            isolation_trees=50,
        )
        cls.report_path = root / "benchmarks" / "baseline_report.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_report_is_written_and_parsable(self) -> None:
        self.assertTrue(self.report_path.exists())
        payload = json.loads(self.report_path.read_text())
        self.assertEqual(payload["output_type"], "unsupervised_deviation_baselines")

    def test_both_baselines_are_present(self) -> None:
        self.assertIn("pca_reconstruction", self.results["baselines"])
        self.assertIn("isolation_forest", self.results["baselines"])

    def test_pca_uses_the_same_latent_dimension_as_the_encoder(self) -> None:
        self.assertEqual(
            self.results["baselines"]["pca_reconstruction"]["n_components"],
            self.config.latent_dim,
        )

    def test_split_boundaries_match_the_encoder_policy(self) -> None:
        policy = self.results["split_policy"]
        self.assertEqual(policy["fit_partition"], "train")
        self.assertEqual(policy["seed"], self.config.random_seed)
        self.assertEqual(policy["fractions"], list(self.config.split_fractions))
        sizes = policy["sizes"]
        self.assertGreater(sizes["train"], sizes["holdout"])

    def test_ssl_artifact_is_compared_on_the_same_rows(self) -> None:
        self.assertIn("metaboguard_ssl", self.results["baselines"])
        self.assertIn(
            "isolation_forest__vs__metaboguard_ssl", self.results["agreement"]
        )
        for payload in self.results["agreement"].values():
            self.assertGreaterEqual(payload["top5pct_flag_jaccard"], 0.0)
            self.assertLessEqual(payload["top5pct_flag_jaccard"], 1.0)

    def test_no_disease_prediction_is_claimed(self) -> None:
        text = json.dumps(self.results).lower()
        for banned in ("auroc", "risk probability", "predicts cancer", "predicts diabetes"):
            self.assertNotIn(banned, text)
        self.assertIn("do not predict cancer or diabetes", self.results["disclaimer"])

    def test_baselines_refuse_invalidated_datasets(self) -> None:
        project_root = API_DIR.parent
        invalidated = project_root / "data" / "nhanes_multicycle.csv"
        if not invalidated.exists():
            self.skipTest("invalidated dataset file not present")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_baselines(invalidated, Path(directory), config=self.config)


if __name__ == "__main__":
    unittest.main()