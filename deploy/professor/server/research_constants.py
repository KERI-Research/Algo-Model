"""Serverless fallback for the two vendored modules that require scikit-learn.

``server/core/self_supervised.py`` imports scikit-learn at module level (it also
contains the training code), and ``server/core/data_reliability.py`` imports
from it. The serverless runtime deliberately installs only NumPy and pandas, so
those imports are unavailable there.

Everything this deployment actually needs from those two modules is either a
constant or one small pandas-only function:

* constants are exported to ``assets/research_constants.json`` by
  ``prepare_assets.py``, straight from the vendored modules, so they cannot
  drift silently;
* :func:`dataset_capabilities` is a verbatim copy of
  ``self_supervised.dataset_capabilities`` (pandas only, no scikit-learn).

``tests/test_research_constants.py`` asserts that the exported constants and
this function match the vendored originals whenever scikit-learn is installed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from . import config

CONSTANTS_FILENAME = "research_constants.json"


@lru_cache(maxsize=1)
def load_constants() -> dict[str, Any]:
    path = config.ASSETS_DIR / CONSTANTS_FILENAME
    if not path.exists():  # pragma: no cover - build error, surfaced loudly
        raise FileNotFoundError(
            f"{CONSTANTS_FILENAME} is missing. Run prepare_assets.py to generate it."
        )
    payload = json.loads(path.read_text())
    payload["plausible_ranges"] = {
        key: tuple(value) for key, value in payload["plausible_ranges"].items()
    }
    return payload


def export_constants() -> dict[str, Any]:
    """Read the constants out of the vendored modules (needs scikit-learn)."""
    import sys

    core_dir = str(config.APP_ROOT / "server" / "core")
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    import data_reliability as dr  # noqa: E402
    import self_supervised as ss  # noqa: E402

    return {
        "schema_version": 1,
        "note": (
            "Exported verbatim from api/self_supervised.py and api/data_reliability.py "
            "so the serverless runtime does not need scikit-learn."
        ),
        "code_version": ss.CODE_VERSION,
        "prevention_features": list(ss.PREVENTION_FEATURES),
        "categorical_features": list(ss.CATEGORICAL_FEATURES),
        "forbidden_early_warning_features": sorted(ss.FORBIDDEN_EARLY_WARNING_FEATURES),
        "plausible_ranges": {
            key: [float(low), float(high)] for key, (low, high) in dr.PLAUSIBLE_RANGES.items()
        },
        "min_coverage_usable": float(dr.MIN_COVERAGE_USABLE),
        "min_coverage_qualified": float(dr.MIN_COVERAGE_QUALIFIED),
        "max_implausible_fraction": float(dr.MAX_IMPLAUSIBLE_FRACTION),
        "reference_links": dict(dr.REFERENCE_LINKS),
    }


# ---------------------------------------------------------------------------
# Verbatim copy of self_supervised.dataset_capabilities (pandas only).
# Keep byte-identical in behaviour; tests compare it to the vendored function.
# ---------------------------------------------------------------------------


def dataset_capabilities(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe, honestly, what this dataset can and cannot support."""
    participant_id = None
    for candidate in (
        "global_participant_id",
        "patient_id",
        "person_id",
        "subject_id",
        "bcr_patient_barcode",
        "SEQN",
    ):
        if candidate in frame.columns:
            participant_id = candidate
            break
    has_repeated_patients = bool(
        participant_id and frame[participant_id].duplicated(keep=False).any()
    )
    time_columns = [
        column
        for column in frame.columns
        if any(
            token in column.lower()
            for token in ("event_date", "diagnosis_date", "followup_days", "event_time_days")
        )
    ]
    has_longitudinal_outcomes = has_repeated_patients and bool(time_columns)
    return {
        "rows": int(len(frame)),
        "participant_id_column": participant_id,
        # Retained for backward compatibility with existing API consumers.
        "repeated_patient_id": participant_id if has_repeated_patients else None,
        "has_repeated_patient_measurements": has_repeated_patients,
        "time_columns": time_columns,
        "supports_future_development_prediction": has_longitudinal_outcomes,
        "supported_output": (
            "multi_horizon_risk"
            if has_longitudinal_outcomes
            else "cross_sectional_representation_and_deviation_only"
        ),
        "longitudinal_heads_enabled": False,
        "output_vocabulary": {
            "metabolic_deviation_score": "How unusual a metabolic profile is versus the training reference.",
            "reference_percentile": "Rank of that deviation score within the training reference distribution.",
            "latent_representation": "16-dimensional learned encoding of the input features.",
            "cross_sectional_association": "Association with an already-present condition. Not future risk.",
        },
        "warning": (
            None
            if has_longitudinal_outcomes
            else "No patient-level longitudinal outcomes: do not describe scores as future disease risk."
        ),
    }
