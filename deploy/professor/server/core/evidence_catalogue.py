"""Biomarker evidence catalogue: schema, loader and provenance gates.

Purpose: every statement MetaboGuard makes about a biomarker must be traceable to a
source, with its study design, validation status, evidence grade and limitations visible.
The catalogue is a plain JSON file (``data/evidence/biomarker_evidence.json``) so it can be
reviewed in a pull request by a clinician without reading Python.

Three rules are enforced in code, not just in prose:

1. **No fabrication.** Required fields must be present; unknowns must be the explicit
   string ``"unknown"`` (or an empty list), never guessed or silently omitted.
2. **Doctor-facing gate.** A row may only be surfaced to a clinician when it has a real
   ``primary_source_url`` (or DOI) *and* an ``evidence_grade`` that is not ``"unknown"``.
3. **No causal or universal-marker claims.** Free-text fields are screened for causal
   phrasing and for the false claim that a cancer has no specific biomarker. The
   defensible statement is that no single marker is universally sufficient for early
   detection, so panels and interacting features are required.

CLI::

    python evidence_catalogue.py --report /tmp/evidence_report.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOGUE_PATH = PROJECT_ROOT / "data" / "evidence" / "biomarker_evidence.json"

#: Every field a row must define. Missing keys are a hard validation error.
REQUIRED_FIELDS: tuple[str, ...] = (
    "entry_id",
    "cancer_site",
    "marker_or_panel",
    "marker_class",
    "specimen",
    "intended_use",
    "stage_or_lead_time",
    "direction",
    "study_design",
    "sample_size",
    "performance",
    "validation_status",
    "evidence_grade",
    "limitations",
    "primary_source_url",
    "doi",
    "related_verified_sources",
)

OPTIONAL_FIELDS: tuple[str, ...] = (
    "repo_reference",
    "available_in_current_data",
    "current_data_column",
    "notes",
    "screening_recommendation_status",
    "allowlisted_statements",
    "denied_statements",
)

#: Fields that must be lists of non-empty strings when present.
STATEMENT_FIELDS: tuple[str, ...] = ("allowlisted_statements", "denied_statements")

#: The two explicit placeholders. Anything else empty is a validation error.
UNKNOWN = "unknown"
NOT_APPLICABLE = "n.a."
PLACEHOLDERS = (UNKNOWN, NOT_APPLICABLE)

#: Free-text fields screened for prohibited phrasing.
TEXT_FIELDS: tuple[str, ...] = (
    "marker_or_panel",
    "intended_use",
    "stage_or_lead_time",
    "performance",
    "limitations",
    "notes",
    "screening_recommendation_status",
)

#: Causal phrasing that requires a causal study design to be permissible at all.
CAUSAL_PATTERNS: tuple[str, ...] = (
    r"\bcauses?\s+cancer\b",
    r"\bcausing\s+cancer\b",
    r"\bcause\s+of\s+cancer\b",
    r"\bbiomarkers?\s+that\s+cause\b",
    r"\bproves?\s+causation\b",
)

#: The specific false generalisation this project must never encode.
UNIVERSAL_DENIAL_PATTERNS: tuple[str, ...] = (
    r"no\s+cancer\s+has\s+(any\s+)?(a\s+)?specific\s+biomarker",
    r"cancers?\s+have\s+no\s+specific\s+biomarkers?",
    r"there\s+are\s+no\s+cancer\s+biomarkers",
)

CAUSAL_CAPABLE_DESIGNS: tuple[str, ...] = (
    "randomised_controlled_trial",
    "randomized_controlled_trial",
    "mendelian_randomisation",
    "mendelian_randomization",
)

URL_PATTERN = re.compile(r"^https?://\S+$")
DOI_PATTERN = re.compile(r"^10\.\d{4,}/\S+$")


@dataclass
class CatalogueIssue:
    entry_id: str
    field_name: str
    level: str  # "hard" | "soft"
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "entry_id": self.entry_id,
            "field": self.field_name,
            "level": self.level,
            "message": self.message,
        }


@dataclass
class EvidenceCatalogue:
    path: Path
    catalogue_version: str
    updated_at: str
    policy: dict[str, str]
    entries: list[dict[str, Any]]
    claims_contract: dict[str, Any] = field(default_factory=dict)
    disease_burden_projections: dict[str, Any] = field(default_factory=dict)
    issues: list[CatalogueIssue] = field(default_factory=list)

    # -- gates ---------------------------------------------------------------
    @property
    def hard_issues(self) -> list[CatalogueIssue]:
        return [issue for issue in self.issues if issue.level == "hard"]

    def entry(self, entry_id: str) -> dict[str, Any]:
        for item in self.entries:
            if item["entry_id"] == entry_id:
                return item
        raise KeyError(entry_id)

    def is_doctor_facing_ready(self, entry: dict[str, Any]) -> bool:
        """A row is clinician-presentable only with a real source and a graded strength."""
        has_source = bool(
            URL_PATTERN.match(str(entry.get("primary_source_url", "")))
            or DOI_PATTERN.match(str(entry.get("doi", "")))
        )
        graded = str(entry.get("evidence_grade", UNKNOWN)).strip().lower() not in PLACEHOLDERS
        return has_source and graded

    def doctor_facing_entries(self) -> list[dict[str, Any]]:
        return [entry for entry in self.entries if self.is_doctor_facing_ready(entry)]

    def research_only_entries(self) -> list[dict[str, Any]]:
        return [entry for entry in self.entries if not self.is_doctor_facing_ready(entry)]

    def allowlisted_statements(self) -> list[dict[str, Any]]:
        """Statements a clinician-facing surface may make, with their source."""
        return [
            {
                "entry_id": entry["entry_id"],
                "statement": statement,
                "evidence_grade": entry["evidence_grade"],
                "study_design": entry["study_design"],
                "primary_source_url": entry["primary_source_url"],
                "doi": entry["doi"],
                "explanation_class": "published_evidence",
                "causal_status": "causal_claim_not_established",
            }
            for entry in self.entries
            if self.is_doctor_facing_ready(entry)
            for statement in entry.get("allowlisted_statements", [])
        ]

    def denied_statements(self) -> list[dict[str, str]]:
        """Statements that must never be made, with the row that forbids them."""
        return [
            {"entry_id": entry["entry_id"], "statement": statement}
            for entry in self.entries
            for statement in entry.get("denied_statements", [])
        ] + [
            {"entry_id": "disease_burden_projections", "statement": statement}
            for statement in self.disease_burden_projections.get("denied_wording", [])
        ]

    def for_cancer_site(self, site: str) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self.entries
            if str(entry.get("cancer_site", "")).lower() == site.lower()
        ]

    def for_current_data_column(self, column: str) -> list[dict[str, Any]]:
        return [entry for entry in self.entries if entry.get("current_data_column") == column]

    # -- reporting -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        graded = [
            str(entry.get("evidence_grade", UNKNOWN)) for entry in self.entries
        ]
        return {
            "catalogue_path": str(self.path),
            "catalogue_version": self.catalogue_version,
            "updated_at": self.updated_at,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "invalid" if self.hard_issues else "valid",
            "entry_count": len(self.entries),
            "doctor_facing_ready": len(self.doctor_facing_entries()),
            "research_only": len(self.research_only_entries()),
            "entries_available_in_current_data": sum(
                1 for entry in self.entries if entry.get("available_in_current_data")
            ),
            "evidence_grades": {grade: graded.count(grade) for grade in sorted(set(graded))},
            "cancer_sites": sorted({str(entry.get("cancer_site")) for entry in self.entries}),
            "allowlisted_statement_count": len(self.allowlisted_statements()),
            "denied_statement_count": len(self.denied_statements()),
            "claims_contract_standards": [
                {"name": item["name"], "url": item["url"], "doi": item["doi"]}
                for item in self.claims_contract.get("standards", [])
            ],
            "disease_burden_projections": self.disease_burden_projections,
            "issues": [issue.as_dict() for issue in self.issues],
            "explanation_class": "published_evidence",
            "disclaimer": (
                "Catalogue rows summarise published risk-associated features, "
                "early-development signals and biological pathways. They are not causal "
                "claims and not statements about this project's own performance."
            ),
            "panel_framing": (
                "Early detection is treated as a panel and feature-interaction problem. "
                "Site-specific markers exist (CA19-9 in pancreatic disease is catalogued "
                "here with its lead-time limits); the catalogue asserts only that no single "
                "marker is universally sufficient across cancers."
            ),
        }


def _validate_entry(entry: dict[str, Any], seen_ids: set[str]) -> list[CatalogueIssue]:
    entry_id = str(entry.get("entry_id", "<missing entry_id>"))
    issues: list[CatalogueIssue] = []

    for name in REQUIRED_FIELDS:
        if name not in entry:
            issues.append(
                CatalogueIssue(entry_id, name, "hard", "Required field is missing.")
            )
            continue
        value = entry[name]
        if name == "related_verified_sources":
            if not isinstance(value, list):
                issues.append(
                    CatalogueIssue(entry_id, name, "hard", "Must be a list (possibly empty).")
                )
            else:
                for source in value:
                    if not URL_PATTERN.match(str(source)):
                        issues.append(
                            CatalogueIssue(
                                entry_id, name, "hard", f"Not a URL: {source!r}"
                            )
                        )
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(
                CatalogueIssue(
                    entry_id,
                    name,
                    "hard",
                    f"Empty value. Use '{UNKNOWN}' when unknown or '{NOT_APPLICABLE}' when "
                    "the field does not apply.",
                )
            )

    unknown_keys = sorted(set(entry) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown_keys:
        issues.append(
            CatalogueIssue(
                entry_id, ",".join(unknown_keys), "hard", "Unrecognised field(s) in row."
            )
        )

    if entry_id in seen_ids:
        issues.append(CatalogueIssue(entry_id, "entry_id", "hard", "Duplicate entry_id."))
    seen_ids.add(entry_id)

    source = str(entry.get("primary_source_url", UNKNOWN))
    doi = str(entry.get("doi", UNKNOWN))
    if source not in PLACEHOLDERS and not URL_PATTERN.match(source):
        issues.append(
            CatalogueIssue(
                entry_id,
                "primary_source_url",
                "hard",
                f"Not a structurally valid URL and not one of {PLACEHOLDERS}.",
            )
        )
    if doi not in PLACEHOLDERS and not DOI_PATTERN.match(doi):
        issues.append(
            CatalogueIssue(
                entry_id, "doi", "hard", f"Not a DOI and not one of {PLACEHOLDERS}."
            )
        )
    for name in STATEMENT_FIELDS:
        if name not in entry:
            continue
        values = entry[name]
        if not isinstance(values, list) or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            issues.append(
                CatalogueIssue(entry_id, name, "hard", "Must be a list of non-empty strings.")
            )
    if entry.get("allowlisted_statements") and not (
        URL_PATTERN.match(source) or DOI_PATTERN.match(doi)
    ):
        issues.append(
            CatalogueIssue(
                entry_id,
                "allowlisted_statements",
                "hard",
                "Allowlisted statements require a real source URL or DOI.",
            )
        )
    if source in PLACEHOLDERS and doi in PLACEHOLDERS:
        issues.append(
            CatalogueIssue(
                entry_id,
                "primary_source_url",
                "soft",
                "No source: row is research-only and must not reach a clinician view.",
            )
        )

    design = str(entry.get("study_design", UNKNOWN)).lower()
    for name in TEXT_FIELDS:
        text = str(entry.get(name, "") or "")
        lowered = text.lower()
        for pattern in CAUSAL_PATTERNS:
            if re.search(pattern, lowered) and design not in CAUSAL_CAPABLE_DESIGNS:
                issues.append(
                    CatalogueIssue(
                        entry_id,
                        name,
                        "hard",
                        "Causal phrasing without a causal study design. Use "
                        "'risk-associated feature', 'early-development signal' or "
                        "'biological pathway'.",
                    )
                )
        for pattern in UNIVERSAL_DENIAL_PATTERNS:
            if re.search(pattern, lowered):
                issues.append(
                    CatalogueIssue(
                        entry_id,
                        name,
                        "hard",
                        "Prohibited claim that cancers have no specific biomarkers. "
                        "State instead that no single marker is universally sufficient.",
                    )
                )
    return issues


def load_catalogue(
    path: str | Path | None = None, strict: bool = False
) -> EvidenceCatalogue:
    """Load and validate the catalogue. ``strict=True`` raises on hard issues."""
    catalogue_path = Path(path or DEFAULT_CATALOGUE_PATH)
    if not catalogue_path.exists():
        raise FileNotFoundError(
            f"Evidence catalogue not found at {catalogue_path}. "
            "Doctor-facing evidence statements are blocked without it."
        )
    payload = json.loads(catalogue_path.read_text())
    for key in ("catalogue_version", "updated_at", "policy", "entries"):
        if key not in payload:
            raise ValueError(f"Catalogue is missing the '{key}' key.")
    if not isinstance(payload["entries"], list) or not payload["entries"]:
        raise ValueError("Catalogue must contain a non-empty 'entries' list.")

    issues: list[CatalogueIssue] = []
    seen_ids: set[str] = set()
    for entry in payload["entries"]:
        issues.extend(_validate_entry(entry, seen_ids))

    claims_contract = dict(payload.get("claims_contract", {}))
    for standard in claims_contract.get("standards", []):
        if not URL_PATTERN.match(str(standard.get("url", ""))):
            issues.append(
                CatalogueIssue(
                    f"claims_contract:{standard.get('name')}",
                    "url",
                    "hard",
                    "Claims-contract standard has no structurally valid URL.",
                )
            )
        doi_value = str(standard.get("doi", NOT_APPLICABLE))
        if doi_value not in PLACEHOLDERS and not DOI_PATTERN.match(doi_value):
            issues.append(
                CatalogueIssue(
                    f"claims_contract:{standard.get('name')}",
                    "doi",
                    "hard",
                    f"DOI must be valid or one of {PLACEHOLDERS}.",
                )
            )
    projections = dict(payload.get("disease_burden_projections", {}))
    for key in ("observed_2024", "projected_2050"):
        block = projections.get(key)
        if block and not URL_PATTERN.match(str(block.get("source_url", ""))):
            issues.append(
                CatalogueIssue(
                    f"disease_burden_projections:{key}",
                    "source_url",
                    "hard",
                    "Projection figures require a structurally valid source URL.",
                )
            )

    catalogue = EvidenceCatalogue(
        path=catalogue_path,
        catalogue_version=str(payload["catalogue_version"]),
        updated_at=str(payload["updated_at"]),
        policy=dict(payload["policy"]),
        entries=list(payload["entries"]),
        claims_contract=claims_contract,
        disease_burden_projections=projections,
        issues=issues,
    )
    if strict and catalogue.hard_issues:
        raise ValueError(
            "Evidence catalogue has hard validation issues: "
            + "; ".join(f"{i.entry_id}.{i.field_name}: {i.message}" for i in catalogue.hard_issues)
        )
    return catalogue


def evidence_for_features(
    features: list[str], catalogue: EvidenceCatalogue | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Map model input columns to catalogued evidence rows (clinician-ready only)."""
    catalogue = catalogue or load_catalogue()
    mapping: dict[str, list[dict[str, Any]]] = {}
    for feature in features:
        rows = [
            {
                "entry_id": entry["entry_id"],
                "marker_or_panel": entry["marker_or_panel"],
                "direction": entry["direction"],
                "evidence_grade": entry["evidence_grade"],
                "study_design": entry["study_design"],
                "limitations": entry["limitations"],
                "primary_source_url": entry["primary_source_url"],
                "doi": entry["doi"],
                "explanation_class": "published_evidence",
                "causal_claim": "causal_claim_not_established",
            }
            for entry in catalogue.for_current_data_column(feature)
            if catalogue.is_doctor_facing_ready(entry)
        ]
        if rows:
            mapping[feature] = rows
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", default=None)
    parser.add_argument("--report", default=None, help="Write the summary JSON here.")
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()

    catalogue = load_catalogue(arguments.catalogue, strict=arguments.strict)
    summary = catalogue.summary()
    if arguments.report:
        report_path = Path(arguments.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if catalogue.hard_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()