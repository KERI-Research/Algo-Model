"""Integrity checks for the aggregate browser-side synthetic profile asset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from data_integrity import file_fingerprint

ROOT = API_DIR.parent
CANONICAL = ROOT / "frontend" / "src" / "interface" / "synthetic_profile_model.json"
DEPLOYED = ROOT / "deploy" / "professor" / "client" / "src" / "lib" / "synthetic_profile_model.json"
DATASET = ROOT / "data" / "nhanes_multicycle_v2.csv"


def _model() -> dict:
    return json.loads(CANONICAL.read_text())


def test_asset_uses_the_current_dataset_training_partition():
    model = _model()
    assert model["schema_version"] == "metaboguard-synthetic-profile-v1"
    assert model["source"]["dataset_sha256"] == file_fingerprint(DATASET)["sha256"]
    assert model["source"]["partition"] == "participant-grouped training split"
    assert model["source"]["split_seed"] == 42


def test_asset_contains_aggregates_not_rows_or_identifiers():
    model = _model()
    assert model["privacy"]["contains_source_rows"] is False
    assert model["privacy"]["contains_identifiers"] is False
    text = CANONICAL.read_text()
    for forbidden in ("global_participant_id", '"SEQN"', "/Volumes/", "/Users/", "/home/"):
        assert forbidden not in text


def test_profile_quantiles_and_correlation_factors_are_valid():
    model = _model()
    fields = model["continuous_fields"]
    assert len(fields) >= 15
    for profile in model["profiles"].values():
        assert profile["training_rows"] >= 250
        factor = np.asarray(profile["correlation_cholesky"], dtype=float)
        assert factor.shape == (len(fields), len(fields))
        assert np.allclose(factor, np.tril(factor))
        assert np.all(np.diag(factor) > 0)
        correlation = factor @ factor.T
        assert np.all(np.linalg.eigvalsh(correlation) > 0)
        for field in fields:
            quantiles = np.asarray(profile["quantiles"][field], dtype=float)
            assert len(quantiles) == len(model["quantile_probabilities"]) == 99
            assert np.all(np.diff(quantiles) >= 0)


def test_professor_copy_matches_the_canonical_asset():
    assert json.loads(DEPLOYED.read_text()) == _model()