"""Single import surface for the vendored MetaboGuard research modules.

``server/core/`` holds byte-for-byte copies of the authoritative modules from
``api/`` in the KERI repository (``self_supervised.py``, ``data_integrity.py``,
``data_reliability.py``, ``evidence_catalogue.py``). They use flat imports, so
that directory is put on ``sys.path`` here and nowhere else.

Nothing in this deployment trains a model. Only the NumPy inference path and the
constant definitions are used, so PyTorch is never imported.

Two runtime profiles
--------------------
* **Full profile** (development, CI, the pplx sandbox deployment): scikit-learn
  is installed, so ``self_supervised`` and ``data_reliability`` import directly
  and every constant comes from the authoritative module.
* **Trimmed profile** (Vercel Hobby serverless function): scikit-learn and SciPy
  are not installed, because the four-library stack exceeds the practical
  unzipped function-size budget. The constants then come from
  ``assets/research_constants.json`` - exported from those same modules by
  ``prepare_assets.py`` - and ``dataset_capabilities`` comes from the verbatim
  copy in :mod:`server.research_constants`. ``data_integrity`` and
  ``evidence_catalogue`` import normally in both profiles; they never needed
  scikit-learn.

Scoring is unaffected either way: :mod:`server.inference` replays the exported
preprocessor constants with NumPy and is proven equal to the scikit-learn path
in ``tests/test_inference_parity.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# These two vendored modules are dependency-light and always import.
from data_integrity import (  # noqa: E402
    DENYLISTED_INPUT_COLUMNS,
    DENYLISTED_INPUT_PREFIXES,
    LABEL_DEFINITIONS,
    is_denylisted_input,
)
from evidence_catalogue import load_catalogue  # noqa: E402

try:  # Full profile: scikit-learn present.
    from data_reliability import (  # noqa: E402
        MAX_IMPLAUSIBLE_FRACTION,
        MIN_COVERAGE_QUALIFIED,
        MIN_COVERAGE_USABLE,
        PLAUSIBLE_RANGES,
        REFERENCE_LINKS,
    )
    from self_supervised import (  # noqa: E402
        CATEGORICAL_FEATURES,
        CODE_VERSION,
        FORBIDDEN_EARLY_WARNING_FEATURES,
        PREVENTION_FEATURES,
        dataset_capabilities,
        score_records,
    )

    VENDORED_MODULES_AVAILABLE = True
except ImportError:  # Trimmed profile: fall back to exported constants.
    from .research_constants import dataset_capabilities, load_constants

    _constants = load_constants()
    CATEGORICAL_FEATURES = tuple(_constants["categorical_features"])
    CODE_VERSION = _constants["code_version"]
    FORBIDDEN_EARLY_WARNING_FEATURES = frozenset(
        _constants["forbidden_early_warning_features"]
    )
    PREVENTION_FEATURES = list(_constants["prevention_features"])
    PLAUSIBLE_RANGES = _constants["plausible_ranges"]
    MIN_COVERAGE_USABLE = _constants["min_coverage_usable"]
    MIN_COVERAGE_QUALIFIED = _constants["min_coverage_qualified"]
    MAX_IMPLAUSIBLE_FRACTION = _constants["max_implausible_fraction"]
    REFERENCE_LINKS = _constants["reference_links"]

    def score_records(*args, **kwargs):  # pragma: no cover - guard, never called
        raise RuntimeError(
            "The scikit-learn scoring path is unavailable in this runtime profile. "
            "server.inference.score_records replays the exported preprocessor "
            "constants with NumPy instead."
        )

    VENDORED_MODULES_AVAILABLE = False

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
    "VENDORED_MODULES_AVAILABLE",
    "dataset_capabilities",
    "is_denylisted_input",
    "load_catalogue",
    "score_records",
]
