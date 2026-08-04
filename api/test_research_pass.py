"""Tests for the research pass: evidence provenance, reliability tiers, clustering gates."""

from __future__ import annotations

import copy
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

import evidence_catalogue as ec  # noqa: E402
from clustering import (  # noqa: E402
    HDBSCAN_AVAILABLE,
    ClusterConfig,
    cramers_v,
    run_clustering,
)
from data_reliability import build_reliability_report  # noqa: E402
from self_supervised import SSLConfig, train_self_supervised  # noqa: E402
from test_data_integrity import synthetic_frame  # noqa: E402

PROJECT_ROOT = API_DIR.parent
DATASET = PROJECT_ROOT / "data" / "nhanes_multicycle_v2.csv"


# ---------------------------------------------------------------------------
# Evidence catalogue
# ---------------------------------------------------------------------------


class EvidenceCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = ec.load_catalogue(strict=True)

    def test_catalogue_is_valid_and_populated(self) -> None:
        self.assertGreaterEqual(len(self.catalogue.entries), 20)
        self.assertEqual(self.catalogue.hard_issues, [])

    def test_every_clinician_ready_row_has_a_source_and_a_grade(self) -> None:
        for entry in self.catalogue.doctor_facing_entries():
            has_source = bool(
                ec.URL_PATTERN.match(str(entry["primary_source_url"]))
                or ec.DOI_PATTERN.match(str(entry["doi"]))
            )
            self.assertTrue(has_source, entry["entry_id"])
            self.assertNotIn(
                str(entry["evidence_grade"]).lower(), ec.PLACEHOLDERS, entry["entry_id"]
            )

    def test_rows_without_a_source_are_never_clinician_facing(self) -> None:
        for entry in self.catalogue.research_only_entries():
            self.assertFalse(self.catalogue.is_doctor_facing_ready(entry))
            self.assertEqual(entry.get("allowlisted_statements", []), [])

    def test_urls_are_structurally_valid_everywhere(self) -> None:
        for entry in self.catalogue.entries:
            source = str(entry["primary_source_url"])
            if source not in ec.PLACEHOLDERS:
                self.assertRegex(source, r"^https?://")
            for related in entry["related_verified_sources"]:
                self.assertRegex(str(related), r"^https?://")

    def test_placeholders_are_explicit(self) -> None:
        for entry in self.catalogue.entries:
            for name in ec.REQUIRED_FIELDS:
                value = entry[name]
                if isinstance(value, str):
                    self.assertTrue(value.strip(), f"{entry['entry_id']}.{name}")

    def test_ca199_row_documents_lead_time_decay_and_screening_status(self) -> None:
        entry = self.catalogue.entry("ev-ca199-alone-pdac-bjsopen-2024")
        self.assertIn("0.55", entry["stage_or_lead_time"])
        self.assertIn("not_recommended", entry["screening_recommendation_status"])
        self.assertEqual(entry["doi"], "10.1093/bjsopen/zrae046")

    def test_catalogue_never_denies_that_specific_markers_exist(self) -> None:
        text = json.dumps(self.catalogue.entries).lower()
        for prohibited in (
            "no cancer has any specific biomarker",
            "cancers have no specific biomarkers",
            "there are no cancer biomarkers",
        ):
            self.assertNotIn(prohibited, text)
        # And a site-specific marker is catalogued, so the claim is refuted by data.
        self.assertTrue(self.catalogue.for_cancer_site("pancreas"))

    def test_no_causal_claim_without_a_causal_design(self) -> None:
        for entry in self.catalogue.entries:
            design = str(entry["study_design"]).lower()
            if design in ec.CAUSAL_CAPABLE_DESIGNS:
                continue
            for name in ec.TEXT_FIELDS:
                text = str(entry.get(name, "") or "").lower()
                self.assertNotRegex(text, r"\bcauses?\s+cancer\b", f"{entry['entry_id']}.{name}")

    def test_allowlisted_statements_carry_provenance(self) -> None:
        statements = self.catalogue.allowlisted_statements()
        self.assertGreaterEqual(len(statements), 10)
        for item in statements:
            self.assertTrue(
                ec.URL_PATTERN.match(str(item["primary_source_url"]))
                or ec.DOI_PATTERN.match(str(item["doi"]))
            )
            self.assertEqual(item["causal_status"], "causal_claim_not_established")

    def test_denied_statements_include_the_false_generalisation(self) -> None:
        denied = " ".join(item["statement"].lower() for item in self.catalogue.denied_statements())
        self.assertIn("no cancer has a specific biomarker", denied)
        self.assertIn("losing weight prevents cancer", denied)

    def test_claims_contract_standards_are_present_and_linked(self) -> None:
        names = {item["name"].split()[0] for item in self.catalogue.claims_contract["standards"]}
        self.assertTrue({"PRoBE", "TRIPOD+AI", "PROBAST+AI", "STARD"} <= names)
        for standard in self.catalogue.claims_contract["standards"]:
            self.assertRegex(standard["url"], r"^https://")

    def test_burden_projection_is_labelled_non_causal(self) -> None:
        projections = self.catalogue.disease_burden_projections
        self.assertEqual(projections["observed_2024"]["cases"], 531318)
        self.assertEqual(projections["projected_2050"]["cases"], 998663)
        self.assertEqual(projections["causal_status"], "causal_claim_not_established")
        self.assertIn("constant-rate", projections["projected_2050"]["method"])

    def test_loader_rejects_a_fabricated_row(self) -> None:
        payload = json.loads(ec.DEFAULT_CATALOGUE_PATH.read_text())
        broken = copy.deepcopy(payload)
        broken["entries"][0]["primary_source_url"] = "not-a-url"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                ec.load_catalogue(path, strict=True)

    def test_loader_rejects_an_empty_required_field(self) -> None:
        payload = json.loads(ec.DEFAULT_CATALOGUE_PATH.read_text())
        broken = copy.deepcopy(payload)
        broken["entries"][0]["limitations"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                ec.load_catalogue(path, strict=True)

    def test_loader_rejects_allowlisted_statement_without_source(self) -> None:
        payload = json.loads(ec.DEFAULT_CATALOGUE_PATH.read_text())
        broken = copy.deepcopy(payload)
        broken["entries"][0]["primary_source_url"] = ec.UNKNOWN
        broken["entries"][0]["doi"] = ec.UNKNOWN
        broken["entries"][0]["allowlisted_statements"] = ["Something clinicians may say."]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                ec.load_catalogue(path, strict=True)


# ---------------------------------------------------------------------------
# Data reliability
# ---------------------------------------------------------------------------


@unittest.skipUnless(DATASET.exists(), "corrected dataset not present")
class DataReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_reliability_report(DATASET, strict=True)
        cls.payload = cls.report.as_dict()

    def test_report_is_structured_and_passing(self) -> None:
        self.assertEqual(self.payload["status"], "ok")
        for section in (
            "provenance",
            "schema",
            "unit_range_plausibility",
            "duplicate_participants",
            "coverage_and_missingness",
            "assay_cycle_drift",
            "label_confidence",
            "leakage_controls",
            "survey_weights",
            "capability_state",
        ):
            self.assertIn(section, self.payload["sections"])

    def test_every_tier_name_is_known_and_features_are_assigned_once(self) -> None:
        self.assertEqual(set(self.payload["tiers"]), {"usable_now", "qualified_use", "unavailable", "prohibited"})
        assigned = [name for names in self.payload["tiers"].values() for name in names]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertTrue(self.payload["tiers"]["usable_now"])

    def test_labels_and_tcga_columns_are_prohibited(self) -> None:
        prohibited = set(self.payload["tiers"]["prohibited"])
        self.assertIn("Cancer", prohibited)
        self.assertIn("Diabetes", prohibited)
        self.assertIn("PancreaticCancer", prohibited)
        self.assertFalse(prohibited & set(self.payload["tiers"]["usable_now"]))

    def test_qualified_use_features_state_a_reason(self) -> None:
        for name in self.payload["tiers"]["qualified_use"]:
            reasons = self.payload["feature_eligibility"][name]["reasons"]
            self.assertTrue(reasons)
            self.assertNotEqual(reasons, ["Present, plausible and adequately covered."])

    def test_capability_state_keeps_future_and_site_outputs_disabled(self) -> None:
        state = self.payload["sections"]["capability_state"]
        self.assertFalse(state["future_risk_enabled"])
        self.assertFalse(state["cancer_site_assignment_enabled"])
        self.assertTrue(state["clustering_enabled"])
        self.assertIn("phenotypes", state["clustering_scope"])

    def test_label_confidence_blocks_site_assignment(self) -> None:
        labels = self.payload["sections"]["label_confidence"]
        self.assertFalse(labels["site_assignment_supported"])
        self.assertTrue(labels["prevalent_not_incident"])
        self.assertTrue(labels["type1_proxy"]["research_only"])

    def test_survey_weights_are_declared_unapplied(self) -> None:
        weights = self.payload["sections"]["survey_weights"]
        self.assertFalse(weights["weights_applied_in_modelling"])
        self.assertIn("unsupervised representation learning", weights["not_applicable_to"])

    def test_implausible_units_are_a_hard_violation(self) -> None:
        frame = pd.read_csv(DATASET, nrows=1200, low_memory=False)
        frame["GHB_LBXGH"] = 9999.0  # sentinel-style encoding error
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nhanes_broken_units_v2.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                build_reliability_report(path, strict=True)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def blobby_frame(rows: int = 900, seed: int = 3) -> pd.DataFrame:
    """Synthetic adults with three genuinely separated metabolic profiles."""
    rng = np.random.default_rng(seed)
    frame = synthetic_frame(rows=rows, seed=seed)
    group = rng.integers(0, 3, rows)
    offsets = {0: (-8.0, -1.2, -35.0), 1: (0.0, 0.0, 0.0), 2: (9.0, 1.6, 45.0)}
    for index in range(rows):
        bmi_shift, hba1c_shift, glucose_shift = offsets[int(group[index])]
        frame.loc[index, "BMX_BMXBMI"] += bmi_shift
        frame.loc[index, "GHB_LBXGH"] += hba1c_shift
        frame.loc[index, "GLU_LBXGLU"] += glucose_shift
        frame.loc[index, "BMX_BMXWAIST"] += bmi_shift * 2
        frame.loc[index, "homa_ir"] += hba1c_shift
    frame["true_group"] = group  # never used by the clustering code
    return frame


def small_cluster_config(**overrides) -> ClusterConfig:
    config = ClusterConfig(
        k_values=(2, 3),
        methods=("kmeans",),
        bootstrap_rounds=3,
        permutation_rounds=1,
        max_fit_rows=600,
        silhouette_sample=600,
        projection_sample=200,
        stability_seeds=(43,),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class ClusteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.frame = blobby_frame()
        cls.dataset = root / "synthetic_phenotypes.csv"
        cls.frame.to_csv(cls.dataset, index=False)
        cls.artifact = root / "artifact"
        train_self_supervised(
            cls.frame,
            cls.artifact,
            SSLConfig(
                epochs=3,
                batch_size=64,
                hidden_dim=32,
                latent_dim=8,
                patience=2,
                max_train_rows=400,
                backend="numpy",
                minimum_adult_rows=100,
                checkpoint_every=0,
                run_label="unit-test",
            ),
        )
        cls.report = run_clustering(
            cls.dataset, cls.artifact, root / "clusters", small_cluster_config()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_report_declares_research_only_semantics(self) -> None:
        self.assertEqual(self.report["output_type"], "exploratory_unsupervised_phenotypes")
        self.assertFalse(self.report["is_disease_classification"])
        self.assertFalse(self.report["labels_used_in_fit_or_selection"])
        self.assertTrue(any("not cancer diagnoses" in text for text in self.report["warnings"]))

    def test_no_label_column_reaches_the_clustering_space(self) -> None:
        text = json.dumps(self.report)
        self.assertNotIn("true_group", text)

    def test_status_is_either_a_finding_or_an_explicit_abstain(self) -> None:
        self.assertIn(self.report["status"], {"stable_clusters_found", "no_stable_clusters"})
        if self.report["status"] == "no_stable_clusters":
            self.assertIn("abstain_reason", self.report)
            self.assertIn("gate_failure_summary", self.report)

    def test_every_candidate_reports_the_full_diagnostic_set(self) -> None:
        evaluated = [c for c in self.report["candidates"] if c["status"] == "evaluated"]
        self.assertTrue(evaluated)
        for candidate in evaluated:
            for key in (
                "train_metrics",
                "bootstrap_stability",
                "seed_stability",
                "negative_controls",
                "permutation_null",
                "outlier_sensitivity",
                "size_profile",
            ):
                self.assertIn(key, candidate)
            metrics = candidate["train_metrics"]
            for metric in ("silhouette", "davies_bouldin", "calinski_harabasz"):
                self.assertIn(metric, metrics)
            self.assertIn("clusterwise_jaccard", candidate["bootstrap_stability"])

    def test_negative_controls_cover_the_required_nuisance_variables(self) -> None:
        candidate = next(c for c in self.report["candidates"] if c["status"] == "evaluated")
        controls = candidate["negative_controls"]["controls"]
        self.assertIn("missingness_burden", controls)
        self.assertIn("assay_availability_burden", controls)
        self.assertIn("age", controls)
        self.assertIn("sex", controls)
        self.assertFalse(
            controls["assay_availability_pattern_diagnostic"]["gating"],
            "high-cardinality pattern must be diagnostic only",
        )

    def test_permutation_null_is_computed(self) -> None:
        candidate = next(c for c in self.report["candidates"] if c["status"] == "evaluated")
        self.assertGreaterEqual(candidate["permutation_null"]["rounds"], 1)

    def test_selected_solution_is_characterised_without_disease_names(self) -> None:
        if self.report["status"] != "stable_clusters_found":
            self.skipTest("no stable solution on this synthetic sample")
        clusters = self.report["characterisation"]["clusters"]
        self.assertTrue(clusters)
        for cluster in clusters:
            self.assertRegex(cluster["cluster_id"], r"^cluster_\d+$")
            self.assertTrue(cluster["prototype_median_profile"])
            self.assertTrue(cluster["top_distinguishing_panel"])
            for banned in ("cancer", "diabetes", "tumour", "carcinoma"):
                self.assertNotIn(banned, cluster["cluster_id"].lower())
        self.assertIn("method", self.report["characterisation"]["membership_confidence"])

    def test_posthoc_labels_are_cross_sectional_and_suppressed_when_small(self) -> None:
        if self.report["status"] != "stable_clusters_found":
            self.skipTest("no stable solution on this synthetic sample")
        summary = self.report["characterisation"]["posthoc_label_summary"]
        self.assertEqual(summary["output_type"], "cross_sectional_association_only")
        self.assertEqual(summary["explanation_class"], "model_association")
        self.assertIn("not future risk", summary["warning"])
        pancreatic = summary["labels"].get("PancreaticCancer")
        if pancreatic:
            self.assertEqual(pancreatic["status"], "suppressed")

    def test_clustering_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repeated = run_clustering(
                self.dataset, self.artifact, Path(directory), small_cluster_config()
            )
        original = [
            (c["method"], c["k"], c.get("train_metrics", {}).get("silhouette"))
            for c in self.report["candidates"]
        ]
        again = [
            (c["method"], c["k"], c.get("train_metrics", {}).get("silhouette"))
            for c in repeated["candidates"]
        ]
        self.assertEqual(original, again)
        self.assertEqual(self.report["status"], repeated["status"])

    def test_chart_source_data_is_persisted(self) -> None:
        output = Path(self.directory.name) / "clusters"
        for name in ("candidate_metrics.csv", "negative_controls.csv", "clustering_report.json"):
            self.assertTrue((output / name).exists(), name)

    def test_density_arm_reports_availability(self) -> None:
        self.assertEqual(
            self.report["method_availability"]["density_arm"],
            "hdbscan" if HDBSCAN_AVAILABLE else "dbscan_fallback",
        )

    def test_nuisance_only_structure_is_flagged_or_abstained(self) -> None:
        """A clustering that only reproduces a batch variable must not be reported."""
        frame = synthetic_frame(rows=900, seed=11)
        frame["survey_cycle"] = np.where(np.arange(len(frame)) % 2 == 0, "1999-2000", "2017-2020")
        # Encode the batch variable directly into the features.
        offset = np.where(frame["survey_cycle"] == "1999-2000", -12.0, 12.0)
        frame["BMX_BMXBMI"] = frame["BMX_BMXBMI"] + offset
        frame["BMX_BMXWAIST"] = frame["BMX_BMXWAIST"] + offset * 2
        frame["GHB_LBXGH"] = frame["GHB_LBXGH"] + offset / 8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "synthetic_batch.csv"
            frame.to_csv(dataset, index=False)
            report = run_clustering(
                dataset, self.artifact, root / "out", small_cluster_config()
            )
        if report["status"] == "stable_clusters_found":
            controls = report["selected"]["negative_controls"]["controls"]
            self.assertLessEqual(controls["survey_cycle"]["value"], 0.30)
        else:
            failures = " ".join(
                " ".join(items) for items in report["gate_failure_summary"].values()
            )
            self.assertIn("negative_control_dominated", failures)


class CharacterisationTests(unittest.TestCase):
    """Characterisation is tested directly so it does not depend on gate outcomes."""

    @classmethod
    def setUpClass(cls) -> None:
        from sklearn.cluster import KMeans

        from clustering import characterise, prepare_inputs

        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        frame = blobby_frame(rows=900, seed=17)
        dataset = root / "synthetic_characterisation.csv"
        frame.to_csv(dataset, index=False)
        artifact = root / "artifact"
        train_self_supervised(
            frame,
            artifact,
            SSLConfig(
                epochs=3,
                batch_size=64,
                hidden_dim=32,
                latent_dim=8,
                patience=2,
                max_train_rows=400,
                backend="numpy",
                minimum_adult_rows=100,
                checkpoint_every=0,
                run_label="unit-test",
            ),
        )
        config = small_cluster_config()
        inputs = prepare_inputs(dataset, artifact, config)
        train_index = inputs.splits["train"]
        matrix = inputs.latent[train_index]
        model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(matrix)
        cls.result = characterise(
            inputs, model.predict(matrix), model, matrix, train_index, config
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_clusters_have_prototypes_and_feature_panels(self) -> None:
        self.assertEqual(len(self.result["clusters"]), 3)
        for cluster in self.result["clusters"]:
            self.assertRegex(cluster["cluster_id"], r"^cluster_\d+$")
            self.assertTrue(cluster["prototype_median_profile"])
            panel = cluster["top_distinguishing_panel"]
            self.assertTrue(panel)
            self.assertLessEqual(len(panel), 8)
            for item in panel:
                self.assertIn(item["direction"], {"higher", "lower"})
            self.assertIn("deviation_score_summary", cluster)

    def test_cluster_identifiers_never_name_a_disease(self) -> None:
        text = json.dumps(self.result["clusters"])
        for banned in ("cancer_cluster", "tumour", "carcinoma", "diabetic_cluster"):
            self.assertNotIn(banned, text.lower())
        for cluster in self.result["clusters"]:
            self.assertIn("prohibited", cluster["label_policy"])

    def test_membership_confidence_is_reported(self) -> None:
        confidence = self.result["membership_confidence"]
        self.assertEqual(confidence["method"], "kmeans_relative_margin")
        self.assertIn("mean_margin", confidence)

    def test_posthoc_summary_is_cross_sectional_and_pancreatic_is_suppressed(self) -> None:
        summary = self.result["posthoc_label_summary"]
        self.assertEqual(summary["output_type"], "cross_sectional_association_only")
        self.assertIn("not future risk", summary["warning"])
        for column, payload in summary["labels"].items():
            self.assertIn(payload["status"], {"reported", "suppressed"})
            if column == "PancreaticCancer":
                self.assertEqual(payload["status"], "suppressed")

    def test_panel_framing_matches_the_meeting_conclusion(self) -> None:
        self.assertIn("panels", self.result["panel_framing"])


class AssociationStatisticTests(unittest.TestCase):
    def test_bias_correction_reduces_high_cardinality_inflation(self) -> None:
        rng = np.random.default_rng(5)
        labels = rng.integers(0, 2, 2000)
        unrelated = rng.integers(0, 300, 2000).astype(str)  # many categories, no relation
        raw = cramers_v(labels, unrelated, bias_correction=False)
        corrected = cramers_v(labels, unrelated, bias_correction=True)
        self.assertLess(corrected, raw)
        self.assertLess(corrected, 0.15)

    def test_perfect_association_is_detected(self) -> None:
        labels = np.array([0] * 500 + [1] * 500)
        categories = np.array(["a"] * 500 + ["b"] * 500)
        self.assertGreater(cramers_v(labels, categories), 0.9)


if __name__ == "__main__":
    unittest.main()