"""Single import surface for the vendored MetaboGuard research modules.

``server/core/`` holds byte-for-byte copies of the authoritative modules from
``api/`` in the KERI repository (``self_supervised.py``, ``data_integrity.py``,
``data_reliability.py``, ``evidence_catalogue.py``). They use flat imports, so
that directory is put on ``sys.path`` here and nowhere else.

Nothing in this deployment trains a model. Only the NumPy inference path
(:func:`score_records` -> ``NumpyAutoencoder``) and the constant definitions are
used, so PyTorch is never imported and is not in the deployment requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from data_integrity import (  # noqa: E402
    DENYLISTED_INPUT_COLUMNS,
    DENYLISTED_INPUT_PREFIXES,
    LABEL_DEFINITIONS,
    is_denylisted_input,
)
from data_reliability import (  # noqa: E402
    MAX_IMPLAUSIBLE_FRACTION,
    MIN_COVERAGE_QUALIFIED,
    MIN_COVERAGE_USABLE,
    PLAUSIBLE_RANGES,
    REFERENCE_LINKS,
)
from evidence_catalogue import load_catalogue  # noqa: E402
from self_supervised import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CODE_VERSION,
    FORBIDDEN_EARLY_WARNING_FEATURES,
    PREVENTION_FEATURES,
    dataset_capabilities,
    score_records,
)

TIER_DEFINITIONS = {
    "usable_now": (
        "Present, plausible, coverage >= 50% of rows, no blocking range or availability gap."
    ),
    "qualified_use": (
        "Usable only with a stated caveat: low coverage, implausible-value burden, or a "
        "declared evidence gap."
    ),
    "unavailable": "Absent from this file, or effectively unmeasured.",
    "prohibited": (
        "Denylisted as a model input (outcome label, label-derived, or post-diagnosis "
        "TCGA context)."
    ),
}

__all__ = [
    "CATEGORICAL_FEATURES",
    "CODE_VERSION",
    "DENYLISTED_INPUT_COLUMNS",
    "DENYLISTED_INPUT_PREFIXES",
    "FORBIDDEN_EARLY_WARNING_FEATURES",
    "LABEL_DEFINITIONS",
    "MAX_IMPLAUSIBLE_FRACTION",
    "MIN_COVERAGE_QUALIFIED",
    "MIN_COVERAGE_USABLE",
    "PLAUSIBLE_RANGES",
    "PREVENTION_FEATURES",
    "REFERENCE_LINKS",
    "TIER_DEFINITIONS",
    "dataset_capabilities",
    "is_denylisted_input",
    "load_catalogue",
    "score_records",
]
