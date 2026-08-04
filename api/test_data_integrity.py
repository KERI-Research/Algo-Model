"""Tests for the data-integrity, leakage and capability controls."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import data_integrity as di  # noqa: E402
from self_supervised import (  # noqa: E402
    FORBIDDEN_EARLY_WARNING_FEATURES,
    dataset_capabilities,
    select_prevention_features,
)

PROJECT_ROOT = API_DIR.parent
CORRECTED_DATASET = PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"


def synthetic_frame(rows: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "global_participant_id": [f"cycle:{index}" for index in range(rows)],
            "DEMO_RIDAGEYR": rng.integers(18, 85, rows),
            "DEMO_RIAGENDR": rng.integers(1, 3, rows),
            "DEMO_RIDRETH3": rng.integers(1, 7, rows),
            "BMX_BMXBMI": rng.normal(28, 5, rows),
            "BMX_BMXWAIST": rng.normal(98, 12, rows),
            "GHB_LBXGH": rng.normal(5.7, 0.9, rows),
            "GLU_LBXGLU": rng.normal(105, 25, rows),
            "INS_LBXIN": rng.normal(12, 6, rows),
            "TRIGLY_LBXTR": rng.normal(140, 60, rows),
            "HDL_LBDHDD": rng.normal(52, 13, rows),
            "TCHOL_LBXTC": rng.normal(190, 35, rows),
            "HSCRP_LBXHSCRP": np.abs(rng.normal(2.5, 2.0, rows)),
            "homa_ir": np.abs(rng.normal(3.0, 1.5, rows)),
            "smoking_status": rng.integers(0, 3, rows),
            "alcohol_status": rng.integers(0, 3, rows),
            "Cancer": rng.integers(0, 2, rows).astype(float),
            "Diabetes": rng.integers(0, 2, rows).astype(float),
            "diabetes_subtype": rng.choice([0.0, 2.0], rows),
        }
    )
    return frame


class CancerCodingTests(unittest.TestCase):
    def test_code_29_is_pancreas_and_39_is_other(self) -> None:
        self.assertEqual(di.NHANES_CANCER_SITE_CODES[29], "Pancreas")
        self.assertEqual(di.NHANES_CANCER_SITE_CODES[39], "Other")
        self.assertEqual(di.PANCREAS_SITE_CODE, 29)

    def test_recomputed_label_uses_code_29_only(self) -> None:
        frame = pd.DataFrame(
            {
                "Cancer": [1.0, 1.0, 1.0, 0.0],
                "MCQ_MCQ230A": [29, 39, 30, np.nan],
                "MCQ_MCQ230B": [np.nan, np.nan, 29, np.nan],
            }
        )
        recomputed = di.recompute_pancreatic_cancer(frame)
        self.assertEqual(list(recomputed), [1.0, 0.0, 1.0, 0.0])

    def test_validator_blocks_the_invalidated_39_coding(self) -> None:
        frame = pd.DataFrame(
            {
                "Cancer": [1.0, 1.0, 1.0],
                "MCQ_MCQ230A": [39, 39, 29],
                # Invalidated label: positives taken from code 39.
                "PancreaticCancer": [1.0, 1.0, 0.0],
            }
        )
        _, findings = di.validate_cancer_coding(frame)
        codes = {finding.code for finding in findings}
        self.assertIn("cancer_coding_mismatch", codes)
        self.assertTrue(any(finding.level == "blocking" for finding in findings))

    @unittest.skipUnless(CORRECTED_DATASET.exists(), "corrected dataset not present")
    def test_corrected_dataset_has_19_pancreas_cases_and_passes(self) -> None:
        report = di.validate_dataset(CORRECTED_DATASET, strict=True).as_dict()
        self.assertEqual(report["status"], "ok")
        coding = report["cancer_coding"]
        self.assertTrue(coding["matches_code_29"])
        self.assertEqual(coding["rows_with_code_29_pancreas"], 19)
        self.assertEqual(coding["stored_positives"], 19)
        self.assertNotEqual(coding["stored_positives"], coding["rows_with_code_39_other"])


class InvalidatedArtifactGateTests(unittest.TestCase):
    def test_invalidated_datasets_are_refused(self) -> None:
        for name in ("nhanes_merged.csv", "nhanes_multicycle.csv"):
            with self.assertRaises(ValueError):
                di.assert_dataset_allowed(PROJECT_ROOT / "data" / name)

    def test_pancreatic_targets_are_refused(self) -> None:
        for target in ("PancreaticCancer", "NODM_PancreaticCancer"):
            with self.assertRaises(ValueError):
                di.assert_target_allowed(target)

    def test_allowed_dataset_and_target_pass(self) -> None:
        di.assert_dataset_allowed(PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv")
        di.assert_target_allowed("Cancer")

    def test_supervised_training_refuses_pancreatic_target(self) -> None:
        from biomarker import train_biomarker_model

        with self.assertRaises(ValueError):
            train_biomarker_model(
                str(PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"),
                target="PancreaticCancer",
            )


class LeakageControlTests(unittest.TestCase):
    def test_labels_and_tcga_columns_are_denylisted(self) -> None:
        for column in (
            "Cancer",
            "PancreaticCancer",
            "Diabetes",
            "diabetes_subtype",
            "tcga_stage_ordinal",
            "tcga_followup_days",
            "tcga_anything_new",
        ):
            self.assertTrue(di.is_denylisted_input(column), column)

    def test_allowlist_contains_no_denylisted_column(self) -> None:
        for feature in di.prevention_allowlist():
            self.assertFalse(di.is_denylisted_input(feature), feature)

    def test_selected_features_exclude_labels_and_tcga(self) -> None:
        frame = synthetic_frame()
        frame["tcga_stage_ordinal"] = 2
        frame["tcga_followup_days"] = 900
        features = select_prevention_features(frame)
        self.assertTrue(set(features).isdisjoint(FORBIDDEN_EARLY_WARNING_FEATURES))
        self.assertFalse([f for f in features if f.startswith("tcga_")])
        self.assertNotIn("Cancer", features)
        self.assertNotIn("Diabetes", features)


class SplitBoundaryTests(unittest.TestCase):
    def test_splits_are_disjoint_deterministic_and_complete(self) -> None:
        frame = synthetic_frame(rows=1000)
        first = di.group_split_indices(frame, seed=42)
        second = di.group_split_indices(frame, seed=42)
        for name in ("train", "validation", "holdout"):
            np.testing.assert_array_equal(first[name], second[name])
        combined = np.concatenate([first[name] for name in first])
        self.assertEqual(len(combined), len(set(combined.tolist())))
        self.assertEqual(len(combined), len(frame))

    def test_repeated_participants_stay_in_one_partition(self) -> None:
        frame = synthetic_frame(rows=600)
        # Simulate longitudinal data: two rows per participant.
        frame = pd.concat([frame, frame], ignore_index=True)
        splits = di.group_split_indices(frame, seed=1)
        partitions = {
            name: set(frame.iloc[index]["global_participant_id"]) for name, index in splits.items()
        }
        self.assertFalse(partitions["train"] & partitions["holdout"])
        self.assertFalse(partitions["train"] & partitions["validation"])
        self.assertFalse(partitions["validation"] & partitions["holdout"])


class CapabilityGateTests(unittest.TestCase):
    def test_cross_sectional_data_reports_no_future_capability(self) -> None:
        capabilities = dataset_capabilities(synthetic_frame())
        self.assertFalse(capabilities["supports_future_development_prediction"])
        self.assertEqual(
            capabilities["supported_output"],
            "cross_sectional_representation_and_deviation_only",
        )
        self.assertFalse(capabilities["longitudinal_heads_enabled"])
        self.assertIn("future disease risk", capabilities["warning"])

    def test_horizon_gate_is_closed_without_follow_up(self) -> None:
        gates = di.horizon_gate_report(synthetic_frame())
        self.assertFalse(gates["any_horizon_eligible"])
        for horizon in ("365d", "1095d", "1825d"):
            self.assertFalse(gates["per_horizon"][horizon]["eligible"])

    def test_horizon_gate_requires_fifty_events_and_non_events(self) -> None:
        frame = pd.DataFrame(
            {
                "event_time_days": [200] * 60 + [2000] * 60,
                "event": [1] * 60 + [0] * 60,
            }
        )
        gates = di.horizon_gate_report(frame, (365, 1095, 1825))
        self.assertTrue(gates["per_horizon"]["365d"]["eligible"])
        self.assertTrue(gates["per_horizon"]["1095d"]["eligible"])

        # 49 events is one short of the gate, so the horizon must stay closed.
        short_events = pd.DataFrame(
            {"event_time_days": [200] * 49 + [2000] * 60, "event": [1] * 49 + [0] * 60}
        )
        self.assertFalse(
            di.horizon_gate_report(short_events, (365,))["per_horizon"]["365d"]["eligible"]
        )

        # 49 non-events is also one short.
        short_non_events = pd.DataFrame(
            {"event_time_days": [200] * 60 + [2000] * 49, "event": [1] * 60 + [0] * 49}
        )
        self.assertFalse(
            di.horizon_gate_report(short_non_events, (1825,))["per_horizon"]["1825d"]["eligible"]
        )

        too_few = pd.DataFrame(
            {"event_time_days": [200] * 49 + [2000] * 60, "event": [1] * 49 + [0] * 60}
        )
        self.assertFalse(
            di.horizon_gate_report(too_few, (365,))["per_horizon"]["365d"]["eligible"]
        )

    def test_longitudinal_head_request_fails_closed(self) -> None:
        from self_supervised import assert_longitudinal_capability

        with self.assertRaises(ValueError):
            assert_longitudinal_capability(synthetic_frame())


class StaleArtifactTests(unittest.TestCase):
    """No invalidated artifact may be reachable or selected by default."""

    ARTIFACT_ROOT = API_DIR / "model_artifacts"

    def test_invalidated_supervised_artifact_dirs_are_marked(self) -> None:
        for directory in self.ARTIFACT_ROOT.glob("*pancreaticcancer*"):
            self.assertTrue(
                (directory / "INVALIDATED.md").exists(),
                f"{directory.name} is missing its INVALIDATED.md marker",
            )

    def test_default_ssl_artifact_is_valid_and_not_a_smoke_run(self) -> None:
        import json

        default_artifact = (
            API_DIR.parent / "model_artifacts" / "metaboguard_ssl" / "nhanes_multicycle_v2"
        )
        if not (default_artifact / "metadata.json").exists():
            self.skipTest("default artifact not present in this checkout")
        metadata = json.loads((default_artifact / "metadata.json").read_text())
        self.assertNotEqual(metadata.get("run_label"), "smoke")
        self.assertFalse(
            metadata["capabilities"]["supports_future_development_prediction"]
        )
        self.assertNotIn("PancreaticCancer", metadata["features"])

    def test_promotion_pointer_never_references_a_smoke_run(self) -> None:
        import json

        pointer = API_DIR.parent / "model_artifacts" / "metaboguard_ssl" / "CURRENT.json"
        if not pointer.exists():
            self.skipTest("no promoted artifact")
        self.assertNotEqual(json.loads(pointer.read_text())["run_label"], "smoke")


class FingerprintTests(unittest.TestCase):
    @unittest.skipUnless(CORRECTED_DATASET.exists(), "corrected dataset not present")
    def test_fingerprint_is_stable(self) -> None:
        first = di.file_fingerprint(CORRECTED_DATASET)
        second = di.file_fingerprint(CORRECTED_DATASET)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(len(first["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()