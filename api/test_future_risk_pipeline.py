"""Tests for the simulation-only future-risk pipeline: schema, cohort, leakage, endpoints.

Run from ``api/``::

    python -m unittest test_future_risk_pipeline -v
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import longitudinal_schema as schema
from longitudinal_dataset import (
    EndpointProtocol,
    build_cohort,
    build_patient_features,
    build_splits,
)
from future_risk_models import (
    FutureRiskConfig,
    apply_calibrator,
    baseline_feature_columns,
    build_person_interval_frame,
    calibration_intercept_slope,
    decision_curve,
    fit_calibrator,
    harrell_c_index,
)
from synthetic_longitudinal import SimulatorConfig, simulate_longitudinal_cohort

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_longitudinal"
FIXTURE = DATA_DIR / "fixture_patient_events.csv"


def _fixture_events() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURE, low_memory=False)
    for column in ("observation_timestamp", "index_date", "event_date", "censoring_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True, format="ISO8601")
    return frame


class CapabilityGateTests(unittest.TestCase):
    """Fail-closed capability gates."""

    def test_clinical_future_risk_is_blocked_for_every_state_this_repo_can_reach(self):
        reachable = [
            schema.CapabilityState.CROSS_SECTIONAL,
            schema.CapabilityState.REPEATED_WITHOUT_OUTCOMES,
            schema.CapabilityState.SIMULATION_ONLY_LONGITUDINAL,
            schema.CapabilityState.POST_DIAGNOSIS,
        ]
        for state in reachable:
            with self.subTest(state=state), self.assertRaises(PermissionError):
                schema.assert_clinical_future_risk_allowed(state)
        # The only permitting state requires a real linked cohort with incident outcomes, which
        # this repository does not have; the API therefore never reports it.
        schema.assert_clinical_future_risk_allowed(
            schema.CapabilityState.LONGITUDINAL_WITH_INCIDENT_OUTCOMES
        )

    def test_simulated_future_risk_requires_flag_and_state(self):
        state = schema.CapabilityState.SIMULATION_ONLY_LONGITUDINAL
        schema.assert_simulated_future_risk_allowed(state, True)
        with self.assertRaises(PermissionError):
            schema.assert_simulated_future_risk_allowed(state, False)
        with self.assertRaises(PermissionError):
            schema.assert_simulated_future_risk_allowed(schema.CapabilityState.CROSS_SECTIONAL, True)

    def test_disabled_outcomes_stay_disabled(self):
        with self.assertRaises(PermissionError):
            schema.assert_outcome_allowed("type1_diabetes")
        with self.assertRaises(PermissionError):
            schema.assert_outcome_allowed("cancer_site")
        schema.assert_outcome_allowed("type2_diabetes")

    def test_horizon_gate_needs_fifty_events_and_non_events(self):
        def frame(events: int, non_events: int) -> pd.DataFrame:
            labels = [1] * events + [0] * non_events
            return pd.DataFrame(
                {
                    "type2_diabetes_1y_label": labels,
                    "type2_diabetes_1y_eligible": [1] * len(labels),
                }
            )

        gate = schema.horizon_gate(frame(49, 500), "type2_diabetes", horizons=(365,))
        self.assertFalse(gate["per_horizon"]["1y"]["eligible"])
        gate = schema.horizon_gate(frame(500, 49), "type2_diabetes", horizons=(365,))
        self.assertFalse(gate["per_horizon"]["1y"]["eligible"])
        gate = schema.horizon_gate(frame(50, 50), "type2_diabetes", horizons=(365,))
        self.assertTrue(gate["per_horizon"]["1y"]["eligible"])
        self.assertTrue(gate["any_horizon_eligible"])


class SchemaValidationTests(unittest.TestCase):
    def test_fixture_passes_strict_validation(self):
        _, report = schema.validate_event_frame(_fixture_events(), strict=True)
        payload = report.as_dict()
        self.assertEqual(payload["status"], "ok", payload.get("errors"))

    def test_missing_column_fails_closed(self):
        frame = _fixture_events().drop(columns=["unit"])
        with self.assertRaises(ValueError):
            schema.validate_event_frame(frame, strict=True)

    def test_implausible_value_is_removed_and_recorded(self):
        frame = _fixture_events().copy()
        mask = frame["feature_code"] == "DEMO_RIDAGEYR"
        target = frame[mask].index[:1]
        frame.loc[target, "value"] = 900.0
        cleaned, report = schema.validate_event_frame(frame, strict=True)
        payload = report.as_dict()
        # The impossible value is removed, not silently modelled, and the removal is reported.
        self.assertEqual(payload["impossible_values"]["DEMO_RIDAGEYR"]["removed"], 1)
        self.assertTrue(cleaned.loc[target, "value"].isna().all())
        self.assertEqual(
            cleaned.loc[target, "missingness_reason"].iloc[0], "invalid_value_removed"
        )

    def test_impossible_value_burden_fails_closed(self):
        frame = _fixture_events().copy()
        mask = frame["feature_code"] == "DEMO_RIDAGEYR"
        frame.loc[frame[mask].index, "value"] = 900.0
        with self.assertRaises(ValueError):
            schema.validate_event_frame(frame, strict=True)

    def test_hba1c_ifcc_harmonisation_is_affine(self):
        frame = pd.DataFrame(
            {
                "feature_code": ["GHB_LBXGH", "GHB_LBXGH"],
                "value": [53.0, 39.0],
                "unit": ["mmol/mol", "mmol/mol"],
            }
        )
        converted = schema.harmonise_units(frame)[0]["value"].round(2).tolist()
        self.assertEqual(converted, [7.0, 5.72])

    def test_visit_matrix_is_time_ordered_with_masks_and_deltas(self):
        matrix = schema.build_visit_matrix(_fixture_events())
        self.assertIn("delta_days_since_previous_visit", matrix.columns)
        for _, group in matrix.groupby("patient_id"):
            ordered = group.sort_values("visit_index")
            self.assertTrue((ordered["relative_time_days"].diff().dropna() > 0).all())
            self.assertEqual(float(ordered["delta_days_since_previous_visit"].iloc[0]), 0.0)
        mask_columns = [c for c in matrix.columns if c.startswith("mask_")]
        self.assertTrue(set(np.unique(matrix[mask_columns].to_numpy())) <= {0, 1})


class DeterministicGenerationTests(unittest.TestCase):
    def test_simulator_is_deterministic_for_a_fixed_seed(self):
        config = SimulatorConfig(patients=12, seed=20260805)
        first = simulate_longitudinal_cohort(config)[0]
        second = simulate_longitudinal_cohort(config)[0]
        self.assertEqual(schema.frame_fingerprint(first), schema.frame_fingerprint(second))

    def test_different_seed_changes_the_cohort(self):
        first = simulate_longitudinal_cohort(SimulatorConfig(patients=12, seed=1))[0]
        second = simulate_longitudinal_cohort(SimulatorConfig(patients=12, seed=2))[0]
        self.assertNotEqual(schema.frame_fingerprint(first), schema.frame_fingerprint(second))

    def test_manifest_declares_enrichment_strata_and_weights(self):
        manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text())
        strata = manifest["generator"]["config"]["enrichment_strata"]
        self.assertTrue(strata)
        for name, stratum in strata.items():
            self.assertIn("share", stratum, name)
            self.assertIn("weight", stratum, name)
        notes = " ".join(manifest["notes"])
        self.assertIn("SIMULATION ONLY", notes)
        self.assertIn("calibrat", notes.lower())

    def test_synthea_manifest_records_runtime_jar_hash_and_seeds(self):
        path = DATA_DIR / "synthea" / "dataset_manifest.json"
        if not path.exists():
            self.skipTest("Synthea-derived manifest not present in this checkout")
        manifest = json.loads(path.read_text())
        generator = manifest["generator"]
        self.assertEqual(generator["used_generator"], "synthea")
        self.assertEqual(manifest.get("cohort_class", "simulation_ordinary_incidence"),
                         "simulation_ordinary_incidence")
        self.assertEqual(
            generator["jar_sha256"],
            "8ba04f7d73abadd5a377e41edf24c5c83935a1cb07c6d982cd5db731ef1cf445",
        )
        self.assertIn("OpenJDK", generator["runtime"])
        seeds = generator["seeds"]
        # Single-batch manifests record a seed mapping; pooled manifests record one seed per
        # deterministic batch. Both must name the base seed.
        recorded = list(seeds.values()) if isinstance(seeds, dict) else list(seeds)
        self.assertIn(20260805, recorded)
        self.assertIn("1825", generator["index_rule"])
        self.assertFalse(generator["enrichment_declared"])
        if "batches" in generator:
            self.assertEqual(len(generator["batches"]), len(recorded))
            for batch in generator["batches"]:
                self.assertIn("compact_sha256", batch)
                self.assertIn("patients_converted", batch)


class CohortAndLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = EndpointProtocol()
        cls.events = _fixture_events()
        cls.cohort, cls.report = build_cohort(cls.events, cls.protocol)
        cls.matrix = schema.build_visit_matrix(cls.events)
        cls.features = build_patient_features(cls.matrix, cls.cohort)

    def test_prevalent_cases_are_excluded_before_index(self):
        self.assertIn("prevalent_type2_diabetes", self.report["exclusions"])
        for outcome in ("type2_diabetes", "pan_cancer"):
            self.assertTrue((self.cohort[f"{outcome}_time_days"] > 0).all())

    def test_censored_before_horizon_is_ineligible_not_negative(self):
        for outcome in ("type2_diabetes", "pan_cancer"):
            for suffix in ("1y", "3y", "5y"):
                censored = self.cohort[f"{outcome}_{suffix}_censored_before_horizon"] == 1
                eligible = self.cohort.loc[censored, f"{outcome}_{suffix}_eligible"]
                self.assertTrue((eligible == 0).all())
                labels = self.cohort.loc[censored, f"{outcome}_{suffix}_label"]
                self.assertTrue((labels == 0).all())  # stored as 0 but masked out everywhere

    def test_competing_death_is_coded_separately(self):
        for outcome in ("type2_diabetes", "pan_cancer"):
            causes = set(self.cohort[f"{outcome}_cause"].unique())
            self.assertTrue(causes <= {0, 1, 2})

    def test_feature_columns_contain_no_outcome_or_time_leakage(self):
        columns = baseline_feature_columns(self.features)
        self.assertTrue(columns)
        for column in columns:
            self.assertNotIn("label", column)
            self.assertNotIn("eligible", column)
            self.assertNotIn("cause", column)
            self.assertNotIn("outcome", column)
            self.assertNotIn("cancer_site", column)
            self.assertNotIn("event_date", column)
            self.assertNotIn("censor", column)
        self.assertNotIn("type2_diabetes_time_days", columns)

    def test_splits_are_disjoint_and_temporal_holdout_is_later(self):
        splits, manifest = build_splits(self.cohort, self.protocol)
        seen: set[str] = set()
        for name in ("train", "validation", "test", "temporal_holdout"):
            ids = set(splits.get(name, []))
            self.assertFalse(seen & ids, f"{name} overlaps an earlier split")
            seen |= ids
        self.assertEqual(manifest["split_manifest_version"], "future-risk-splits-v1")
        self.assertEqual(manifest["seed"], self.protocol.seed)
        self.assertTrue(all(value == 0 for value in manifest["overlaps"].values()))
        self.assertEqual(len(manifest["patient_id_fingerprints"]), 4)

    def test_person_interval_frame_stops_at_the_event(self):
        config = FutureRiskConfig()
        intervals = build_person_interval_frame(self.features, "type2_diabetes", config)
        for patient_id, group in intervals.groupby("patient_id"):
            events = group["event"].to_numpy()
            if events.sum():
                self.assertEqual(int(events[-1]), 1, patient_id)
                self.assertEqual(int(events[:-1].sum()), 0, patient_id)


class MetricAndCalibrationTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(7)
        self.scores = random.uniform(0, 1, 400)
        self.labels = (random.uniform(0, 1, 400) < self.scores).astype(int)

    def test_isotonic_calibration_improves_a_biased_score(self):
        biased = np.clip(self.scores * 0.4, 0, 1)
        calibrator = fit_calibrator(biased, self.labels)
        calibrated = apply_calibrator(calibrator, biased)
        before = abs(biased.mean() - self.labels.mean())
        after = abs(calibrated.mean() - self.labels.mean())
        self.assertLess(after, before)

    def test_calibration_slope_is_near_one_for_honest_probabilities(self):
        result = calibration_intercept_slope(self.scores, self.labels)
        self.assertIsNotNone(result["slope"])
        self.assertGreater(result["slope"], 0.4)

    def test_c_index_beats_chance_for_informative_risk(self):
        times = np.linspace(100, 2000, 300)
        events = np.ones(300, dtype=int)
        risk = -times  # earlier event => higher risk
        self.assertGreater(harrell_c_index(times, events, risk, seed=1), 0.8)

    def test_decision_curve_reports_false_alert_burden(self):
        rows = decision_curve(self.scores, self.labels, (0.1, 0.5))
        for row in rows:
            self.assertIn("false_alerts_per_100_screened", row)
            self.assertIn("net_benefit_model", row)
            self.assertLessEqual(row["flagged_fraction"], 1.0)


class FutureRiskEndpointTests(unittest.TestCase):
    """Endpoint gating. No artifact is required for these to be meaningful."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import main

        cls.client = TestClient(main.app)

    def test_clinical_future_risk_still_returns_409(self):
        response = self.client.post("/api/v1/prevention-future-risk", json={})
        self.assertEqual(response.status_code, 409)

    def test_capability_endpoint_reports_disabled_clinical_risk(self):
        payload = self.client.get("/api/v1/future-risk-capability").json()
        self.assertFalse(payload["clinical_future_risk_enabled"])
        self.assertIn("type1_diabetes", payload["disabled_outcomes"])
        self.assertEqual(payload["horizons_days"], [365, 1095, 1825])
        self.assertEqual(payload["event_gate"]["minimum_events"], 50)

    def test_simulation_endpoint_requires_explicit_simulation_mode(self):
        response = self.client.post(
            "/api/v1/simulation/future-risk-score",
            json={"patient_history": [{"days_before_index": 700}, {"days_before_index": 100}]},
        )
        self.assertEqual(response.status_code, 403)

    def test_simulation_endpoint_rejects_cross_sectional_uploads(self):
        response = self.client.post(
            "/api/v1/simulation/future-risk-score",
            json={"simulation_mode": True, "patient_record": {"GLU_LBXGH": 6.1}},
        )
        self.assertEqual(response.status_code, 422)

    def test_simulation_endpoint_rejects_a_single_visit(self):
        response = self.client.post(
            "/api/v1/simulation/future-risk-score",
            json={"simulation_mode": True, "patient_history": [{"days_before_index": 30}]},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)