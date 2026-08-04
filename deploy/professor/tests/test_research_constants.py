"""The trimmed serverless profile must expose the same constants and helpers.

Safety proof for the fallback in ``server/core_bridge.py``: whenever
scikit-learn is installed, the exported constants and the copied
``dataset_capabilities`` must match the authoritative vendored modules exactly.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from server import config, core_bridge, research_constants

vendored = pytest.importorskip(
    "server.core.self_supervised", reason="vendored scikit-learn path unavailable"
)
vendored_reliability = pytest.importorskip("server.core.data_reliability")


@pytest.fixture
def exported():
    return json.loads((config.ASSETS_DIR / research_constants.CONSTANTS_FILENAME).read_text())


def test_exported_file_is_present_and_versioned(exported):
    assert exported["schema_version"] == 1
    assert exported["note"]


def test_feature_lists_match_the_vendored_module(exported):
    assert exported["prevention_features"] == list(vendored.PREVENTION_FEATURES)
    assert exported["categorical_features"] == list(vendored.CATEGORICAL_FEATURES)
    assert exported["forbidden_early_warning_features"] == sorted(
        vendored.FORBIDDEN_EARLY_WARNING_FEATURES
    )
    assert exported["code_version"] == vendored.CODE_VERSION


def test_reliability_thresholds_match_the_vendored_module(exported):
    assert exported["min_coverage_usable"] == vendored_reliability.MIN_COVERAGE_USABLE
    assert exported["min_coverage_qualified"] == vendored_reliability.MIN_COVERAGE_QUALIFIED
    assert exported["max_implausible_fraction"] == vendored_reliability.MAX_IMPLAUSIBLE_FRACTION
    assert exported["reference_links"] == dict(vendored_reliability.REFERENCE_LINKS)


def test_plausible_ranges_match_the_vendored_module(exported):
    loaded = research_constants.load_constants()["plausible_ranges"]
    assert set(loaded) == set(vendored_reliability.PLAUSIBLE_RANGES)
    for feature, bounds in vendored_reliability.PLAUSIBLE_RANGES.items():
        assert loaded[feature] == tuple(float(value) for value in bounds)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"DEMO_RIDAGEYR": [40, 50]}),
        pd.DataFrame({"SEQN": [1, 1, 2], "GHB_LBXGH": [5.1, 5.4, 6.0]}),
        pd.DataFrame({"patient_id": ["a", "a"], "followup_days": [10, 20]}),
        pd.DataFrame({"subject_id": ["a", "b"], "event_date": ["2020-01-01", None]}),
        pd.DataFrame({"bcr_patient_barcode": ["x", "x"], "diagnosis_date": ["d", "e"]}),
        pd.DataFrame(),
    ],
)
def test_dataset_capabilities_copy_matches_the_vendored_function(frame):
    assert research_constants.dataset_capabilities(frame) == vendored.dataset_capabilities(
        frame
    )


def test_bridge_reports_the_full_profile_in_this_environment():
    assert core_bridge.VENDORED_MODULES_AVAILABLE is True
    frame = pd.DataFrame({"SEQN": [1, 1, 2], "followup_days": [1, 2, 3]})
    assert core_bridge.dataset_capabilities(frame) == vendored.dataset_capabilities(frame)


def test_trimmed_profile_imports_without_sklearn(tmp_path):
    """Reimport the bridge with scikit-learn blocked: constants must still load."""
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__
    blocked = {"sklearn", "self_supervised", "data_reliability", "scipy", "joblib"}

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name in {"server.core_bridge", "self_supervised", "data_reliability"}
    }
    for name in saved:
        sys.modules.pop(name, None)
    builtins.__import__ = guard
    try:
        module = importlib.import_module("server.core_bridge")
        assert module.VENDORED_MODULES_AVAILABLE is False
        assert list(module.PREVENTION_FEATURES) == list(vendored.PREVENTION_FEATURES)
        assert module.MIN_COVERAGE_USABLE == vendored_reliability.MIN_COVERAGE_USABLE
        assert module.is_denylisted_input("Cancer") is True
        assert module.is_denylisted_input("tcga_stage_ordinal") is True
        with pytest.raises(RuntimeError):
            module.score_records(pd.DataFrame())
    finally:
        builtins.__import__ = real_import
        sys.modules.pop("server.core_bridge", None)
        sys.modules.update(saved)
        importlib.import_module("server.core_bridge")
