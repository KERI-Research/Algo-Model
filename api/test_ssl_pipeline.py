"""Tests for the self-supervised training pipeline and its safety properties."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from data_integrity import group_split_indices  # noqa: E402
from self_supervised import (  # noqa: E402
    CODE_VERSION,
    NumpyAutoencoder,
    SSLConfig,
    build_preprocessor,
    render_model_card,
    score_records,
    select_prevention_features,
    train_self_supervised,
)
from test_data_integrity import synthetic_frame  # noqa: E402


def small_config(**overrides) -> SSLConfig:
    config = SSLConfig(
        epochs=2,
        batch_size=64,
        hidden_dim=32,
        latent_dim=8,
        patience=2,
        max_train_rows=400,
        backend="numpy",
        minimum_adult_rows=100,
        checkpoint_every=1,
        run_label="unit-test",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_frame(rows=900)
        cls.directory = tempfile.TemporaryDirectory()
        cls.output = Path(cls.directory.name) / "artifact"
        cls.metadata = train_self_supervised(
            cls.frame,
            cls.output,
            small_config(),
            dataset_fingerprint={"name": "synthetic.csv", "sha256": "0" * 64},
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_artifact_files_are_written(self) -> None:
        for name in (
            "metadata.json",
            "preprocessor.joblib",
            "autoencoder_weights.npz",
            "MODEL_CARD.md",
            "splits.npz",
            "checkpoint_weights.npz",
            "checkpoint_state.json",
        ):
            self.assertTrue((self.output / name).exists(), name)

    def test_metadata_records_version_and_provenance(self) -> None:
        self.assertEqual(self.metadata["code_version"], CODE_VERSION)
        self.assertEqual(self.metadata["dataset_fingerprint"]["sha256"], "0" * 64)
        manifest = self.metadata["run_manifest"]
        self.assertEqual(manifest["random_seed"], 42)
        self.assertEqual(manifest["backend"], "numpy")
        self.assertEqual(manifest["device"], "cpu")
        self.assertIn("numpy", manifest["package_versions"])
        self.assertGreater(manifest["training_seconds"], 0)

    def test_encoder_never_sees_outcome_labels(self) -> None:
        for label in ("Cancer", "Diabetes", "diabetes_subtype"):
            self.assertNotIn(label, self.metadata["features"])
        self.assertIn("Cancer", self.metadata["label_columns_present_but_unused_in_training"])

    def test_split_policy_is_grouped_and_train_only_preprocessing(self) -> None:
        policy = self.metadata["split_policy"]
        self.assertEqual(policy["grouped_by"], "global_participant_id")
        self.assertEqual(policy["preprocessing_fit_partition"], "train")
        self.assertEqual(policy["deviation_reference_partition"], "train")
        self.assertEqual(self.metadata["score_distribution"]["reference_partition"], "train")

    def test_persisted_splits_are_disjoint(self) -> None:
        archive = np.load(self.output / "splits.npz")
        train = set(archive["train"].tolist())
        validation = set(archive["validation"].tolist())
        holdout = set(archive["holdout"].tolist())
        self.assertFalse(train & validation)
        self.assertFalse(train & holdout)
        self.assertFalse(validation & holdout)

    def test_capabilities_and_model_card_are_honest(self) -> None:
        self.assertFalse(self.metadata["capabilities"]["supports_future_development_prediction"])
        self.assertFalse(self.metadata["capabilities"]["longitudinal_heads_enabled"])
        card = (self.output / "MODEL_CARD.md").read_text()
        self.assertIn("does not diagnose", card)
        self.assertIn("does not estimate the probability", card)
        self.assertIn("50-event", card)
        self.assertIn(CODE_VERSION, render_model_card(self.metadata))

    def test_posthoc_checks_are_labelled_cross_sectional(self) -> None:
        for payload in self.metadata["posthoc_association_checks"].values():
            self.assertIn(
                payload["status"],
                {"cross_sectional_association_only", "not_evaluated"},
            )
            if payload["status"] == "cross_sectional_association_only":
                self.assertIn("not measure future", payload["warning"])

    def test_scoring_output_is_deviation_not_probability(self) -> None:
        record = {
            "DEMO_RIDAGEYR": 55,
            "BMX_BMXBMI": 31.0,
            "GHB_LBXGH": 6.4,
            "GLU_LBXGLU": 130,
        }
        result = score_records(pd.DataFrame([record]), self.output)[0]
        self.assertFalse(result["is_future_risk_probability"])
        self.assertEqual(result["output_type"], "metabolic_deviation_and_representation")
        self.assertGreaterEqual(result["metabolic_deviation_score"], 0.0)
        self.assertGreaterEqual(result["reference_percentile"], 0.0)
        self.assertLessEqual(result["reference_percentile"], 100.0)
        self.assertEqual(len(result["latent_representation"]), self.metadata["latent_dim"])
        # No output field may present itself as a probability of disease.
        probability_keys = [
            key for key in result if "probab" in key and key != "is_future_risk_probability"
        ]
        self.assertEqual(probability_keys, [])
        self.assertIn("not a cancer or diabetes diagnosis", result["interpretation"])

    def test_training_is_deterministic_for_a_fixed_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repeated = train_self_supervised(
                self.frame, Path(directory) / "again", small_config()
            )
        first = np.load(self.output / "autoencoder_weights.npz")["enc1__weight"]
        second_metadata = repeated
        self.assertEqual(
            second_metadata["training_history"][-1]["train_loss"],
            self.metadata["training_history"][-1]["train_loss"],
        )
        self.assertEqual(first.shape[0], 32)

    def test_resume_continues_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resume"
            train_self_supervised(self.frame, output, small_config(epochs=2))
            state = json.loads((output / "checkpoint_state.json").read_text())
            self.assertEqual(state["next_epoch"], 2)
            resumed = train_self_supervised(
                self.frame, output, small_config(epochs=4, resume=True)
            )
            self.assertEqual(resumed["run_manifest"]["resumed_from_epoch"], 2)
            self.assertGreaterEqual(len(resumed["training_history"]), 3)

    def test_backend_mismatch_on_resume_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mismatch"
            train_self_supervised(self.frame, output, small_config())
            state_path = output / "checkpoint_state.json"
            state = json.loads(state_path.read_text())
            state["backend"] = "torch"
            state_path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                train_self_supervised(self.frame, output, small_config(resume=True))

    def test_too_few_adult_rows_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                train_self_supervised(
                    self.frame.head(50), Path(directory) / "tiny", small_config()
                )


class PreprocessingLeakageTests(unittest.TestCase):
    def test_preprocessor_statistics_ignore_holdout_rows(self) -> None:
        """Extreme values placed only in the holdout must not move training medians."""
        frame = synthetic_frame(rows=800).reset_index(drop=True)
        features = select_prevention_features(frame)
        splits = group_split_indices(frame, seed=42)

        clean = build_preprocessor(features)
        clean.fit(frame.iloc[splits["train"]][features])
        clean_centre = clean.named_transformers_["numeric"].named_steps["scale"].center_

        contaminated = frame.copy()
        contaminated.loc[splits["holdout"], "GHB_LBXGH"] = 99.0
        dirty = build_preprocessor(features)
        dirty.fit(contaminated.iloc[splits["train"]][features])
        dirty_centre = dirty.named_transformers_["numeric"].named_steps["scale"].center_

        np.testing.assert_allclose(clean_centre, dirty_centre)


class ScorerBackendTests(unittest.TestCase):
    def test_exported_weights_match_layer_contract(self) -> None:
        frame = synthetic_frame(rows=600)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            train_self_supervised(frame, output, small_config())
            scorer = NumpyAutoencoder(output / "autoencoder_weights.npz")
            for prefix in ("enc1", "enc2", "enc_out", "dec1", "dec2", "dec_out"):
                self.assertIn(f"{prefix}.weight", scorer.weights)
                self.assertIn(f"{prefix}.bias", scorer.weights)


if __name__ == "__main__":
    unittest.main()