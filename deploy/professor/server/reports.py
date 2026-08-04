"""Read-only research surfaces: reliability, clustering abstention, evidence.

Every payload here comes from a report that the authoritative pipeline already
produced (``model_artifacts/research_runs/research__20260804T164743Z`` and
``data/evidence/biomarker_evidence.json``). This deployment re-serves them; it
never regenerates or softens them.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from . import config
from .core_bridge import LABEL_DEFINITIONS, REFERENCE_LINKS, load_catalogue

RESEARCH_RUN = "research__20260804T164743Z"

EXPLANATION_CLASSES = {
    "data_observation": "Measured directly in the data file (counts, coverage, drift).",
    "model_association": "Produced by our model on our sample. Not validated, not causal.",
    "published_evidence": "From a catalogued source with URL, study design and evidence grade.",
    "causal_claim_not_established": "Default status for any mechanism statement.",
}

SUPPORTED_CLAIMS = [
    "This profile is unusual relative to the NHANES adult reference distribution.",
    "These features carry the largest reconstruction error for this record.",
    "This feature is usable, qualified, unavailable or prohibited in the current data.",
    "This biomarker has a catalogued published association at the stated evidence grade.",
    "A multi-marker panel is required; no single marker is sufficient for early detection.",
    "No stable metabolic phenotype cluster was found under the stated stability gates.",
]

PROHIBITED_CLAIMS = [
    "This patient has, or will develop, cancer or diabetes.",
    "This score is a 1, 3 or 5 year risk probability.",
    "This cluster is a cancer type, cancer site or disease subtype.",
    "This biomarker causes, or protects against, the outcome.",
    "The model is validated for screening, triage or diagnosis.",
    "The output can be used without clinician review.",
]

METHOD_REFERENCES = [
    {
        "id": "probe",
        "title": "PRoBE design: prospective specimen collection, retrospective blinded evaluation",
        "citation": "Pepe MS et al., Cancer Epidemiol Biomarkers Prev, 2008.",
        "url": "https://doi.org/10.1158/1055-9965.EPI-08-0234",
        "why_it_matters": (
            "Sets the specimen and blinding design an early-detection biomarker study must "
            "meet. The current cross-sectional survey data do not meet it, which is why no "
            "detection performance is claimed."
        ),
    },
    {
        "id": "tripod_ai",
        "title": "TRIPOD+AI reporting guideline for prediction models",
        "citation": "Collins GS et al., BMJ, 2024.",
        "url": "https://doi.org/10.1136/bmj-2023-078378",
        "why_it_matters": (
            "Defines the reporting items for a clinical prediction model. This deployment "
            "reports the items it can (data, features, definitions, limitations) and states "
            "openly that external validation and calibration are absent."
        ),
    },
    {
        "id": "nhanes_weighting",
        "title": "NHANES analytic guidance: survey weights",
        "citation": "US CDC / NCHS NHANES tutorials.",
        "url": REFERENCE_LINKS.get(
            "nhanes_weighting_tutorial",
            "https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx",
        ),
        "why_it_matters": (
            "Survey weights are not applied in the model, so all outputs describe the "
            "analytic sample rather than the US adult population."
        ),
    },
    {
        "id": "sklearn_leakage",
        "title": "Common pitfalls: data leakage",
        "citation": "scikit-learn documentation.",
        "url": REFERENCE_LINKS.get(
            "sklearn_leakage_pitfalls", "https://scikit-learn.org/stable/common_pitfalls.html"
        ),
        "why_it_matters": (
            "Basis for the input denylist: outcome, label-derived and post-diagnosis TCGA "
            "columns are refused as model inputs."
        ),
    },
]


#: Absolute filesystem paths from the authoring machine must never reach a client.
_ABSOLUTE_PATH = re.compile(r"(?:/Volumes|/Users|/home|/private|/var/folders|[A-Za-z]:\\\\)[^\s\"']*")


def sanitise(value: Any) -> Any:
    """Recursively reduce absolute filesystem paths to their file name."""
    if isinstance(value, dict):
        return {key: sanitise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitise(item) for item in value]
    if isinstance(value, str) and _ABSOLUTE_PATH.search(value):
        return _ABSOLUTE_PATH.sub(lambda match: match.group(0).replace("\\\\", "/").rsplit("/", 1)[-1], value)
    return value


def _read_report(name: str) -> dict[str, Any]:
    path = config.REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Bundled report '{name}' is missing from this deployment.")
    return sanitise(json.loads(path.read_text()))


@lru_cache(maxsize=8)
def reliability_report() -> dict[str, Any]:
    report = _read_report("data_reliability_report.json")
    sections = report.get("sections", {})
    return {
        "explanation_class": "data_observation",
        "explanation_classes": EXPLANATION_CLASSES,
        "source": {
            "research_run": RESEARCH_RUN,
            "dataset": report.get("dataset"),
            "generated_at": report.get("generated_at"),
            "report_schema_version": report.get("report_schema_version"),
        },
        "status": report.get("status"),
        "tier_definitions": report.get("tier_definitions"),
        "tiers": report.get("tiers"),
        "feature_eligibility": report.get("feature_eligibility"),
        "violations": report.get("violations", []),
        "sections": {
            key: sections.get(key)
            for key in (
                "provenance",
                "row_counts",
                "label_confidence",
                "survey_weights",
                "capability_state",
                "leakage_controls",
            )
            if key in sections
        },
        "assay_cycle_drift_summary": {
            "level_drift_features": (sections.get("assay_cycle_drift") or {}).get(
                "level_drift_features"
            ),
            "availability_gap_features": (sections.get("assay_cycle_drift") or {}).get(
                "availability_gap_features"
            ),
            "note": (sections.get("assay_cycle_drift") or {}).get("note"),
        },
        "label_definitions": LABEL_DEFINITIONS,
        "interpretation_note": report.get("interpretation_note"),
        "reference_links": report.get("reference_links", REFERENCE_LINKS),
        "fail_closed_controls": [
            "Future-horizon scoring is disabled: no horizon passes the event-count gate.",
            "Denylisted outcome, label-derived and TCGA post-diagnosis columns cannot be model "
            "inputs.",
            "Features below the coverage floor are marked unavailable rather than imputed and "
            "used.",
            "Clustering abstains rather than reporting an unstable solution.",
        ],
    }


@lru_cache(maxsize=8)
def clustering_report(variant: str = "complete_cases") -> dict[str, Any]:
    if variant not in {"complete_cases", "all_adults"}:
        raise KeyError(variant)
    report = _read_report(f"clustering_{variant}.json")
    payload = {
        "explanation_class": "model_association",
        "explanation_classes": EXPLANATION_CLASSES,
        "causal_status": "causal_claim_not_established",
        "run": RESEARCH_RUN,
        "variant": variant,
        "available_variants": ["complete_cases", "all_adults"],
        "status": report.get("status"),
        "output_type": report.get("output_type"),
        "is_disease_classification": False,
        "labels_used_in_fit_or_selection": report.get("labels_used_in_fit_or_selection"),
        "cluster_naming_policy": (
            "Clusters are patient/metabolic phenotypes. Labelling a cluster as a cancer type, "
            "cancer site or disease subtype is prohibited."
        ),
        "space": report.get("space"),
        "space_dimension": report.get("space_dimension"),
        "split_source": report.get("split_source"),
        "split_sizes": report.get("split_sizes"),
        "fit_rows": report.get("fit_rows"),
        "config": report.get("config"),
        "warnings": report.get("warnings", []),
        "validity_caveats": report.get("validity_caveats"),
        "method_references": report.get("method_references"),
        "candidate_summary": [
            {
                "method": item.get("method"),
                "k": item.get("k"),
                "silhouette": (item.get("train_metrics") or {}).get("silhouette"),
                "davies_bouldin": (item.get("train_metrics") or {}).get("davies_bouldin"),
                "calinski_harabasz": (item.get("train_metrics") or {}).get("calinski_harabasz"),
                "bootstrap_mean_ari": (item.get("bootstrap_stability") or {}).get("mean_ari"),
                "seed_mean_ari": (item.get("seed_stability") or {}).get("mean_ari"),
                "negative_controls": (item.get("negative_controls") or {}).get("controls"),
                "dominated_by": (item.get("negative_controls") or {}).get("dominated_by"),
                "passes_gates": item.get("passes_gates"),
                "gate_failures": item.get("gate_failures"),
            }
            for item in report.get("candidates", [])
            if item.get("status") == "evaluated"
        ],
    }
    if report.get("status") == "no_stable_clusters":
        payload["abstain"] = {
            "reason": report.get("abstain_reason"),
            "gate_failure_summary": report.get("gate_failure_summary"),
            "interpretation": (
                "No phenotype solution is reported because none passed the stability and "
                "negative-control gates. This is a result, not a failure to produce output."
            ),
            "survey_cycle_explanation": (
                "Every candidate was dominated by the survey-cycle negative control: the "
                "strongest structure in the data tracks which NHANES collection cycle a record "
                "came from, not metabolic phenotype. Assay methods, availability and calibration "
                "changed between cycles, so a cluster boundary that follows cycle boundaries "
                "cannot be interpreted biologically. Clusters also dissolved under bootstrap "
                "resampling, so they are not reproducible sub-populations."
            ),
        }
    else:  # pragma: no cover - the current run abstains
        payload["selected"] = report.get("selected")
    return payload


@lru_cache(maxsize=1)
def evidence_payload() -> dict[str, Any]:
    catalogue = load_catalogue(config.EVIDENCE_PATH)
    summary = sanitise(catalogue.summary())
    # Do not expose server filesystem paths in an API response.
    summary.pop("catalogue_path", None)
    return {
        "explanation_class": "published_evidence",
        "explanation_classes": EXPLANATION_CLASSES,
        "causal_status": "causal_claim_not_established",
        "summary": summary,
        "policy": catalogue.policy,
        "clinician_ready_entries": sanitise(catalogue.doctor_facing_entries()),
        "research_only_entries": [
            {
                "entry_id": entry.get("entry_id"),
                "marker_or_panel": entry.get("marker_or_panel"),
                "reason": "Missing source URL/DOI or ungraded evidence.",
            }
            for entry in catalogue.research_only_entries()
        ],
        "panel_framing": summary.get("panel_framing"),
        "allowlisted_statements": sanitise(catalogue.allowlisted_statements()),
        "denied_statements": sanitise(catalogue.denied_statements()),
        "claims_contract": sanitise(catalogue.claims_contract),
        "method_references": METHOD_REFERENCES,
        "supported_claims": SUPPORTED_CLAIMS,
        "prohibited_claims": PROHIBITED_CLAIMS,
    }


@lru_cache(maxsize=1)
def integrity_report() -> dict[str, Any]:
    report = _read_report("data_integrity_report.json")
    return {
        "explanation_class": "data_observation",
        "source": {"research_run": RESEARCH_RUN, "dataset": report.get("dataset")},
        "status": report.get("status"),
        "findings": report.get("findings", []),
        "horizon_gates": report.get("horizon_gates"),
        "row_counts": report.get("row_counts"),
        "capabilities": report.get("capabilities"),
        "label_definitions": report.get("label_definitions", LABEL_DEFINITIONS),
    }
