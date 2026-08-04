"""The dependency-light NumPy inference path must equal the vendored path.

These tests are the safety proof for dropping scikit-learn, SciPy and joblib
from the serverless runtime requirements. They are skipped automatically when
scikit-learn is unavailable (that is, in the trimmed serverless environment
itself), and they run in development and CI where it is installed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from server import config, inference
from server.core_bridge import PREVENTION_FEATURES

warnings.simplefilter("ignore")

sklearn_score_records = pytest.importorskip(
    "server.core.self_supervised", reason="vendored scikit-learn path unavailable"
).score_records

CATEGORICAL = ("DEMO_RIAGENDR", "DEMO_RIDRETH3", "smoking_status", "alcohol_status")


def _random_frame(rows: int, seed: int, as_object: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(rows):
        record = {}
        for feature in PREVENTION_FEATURES:
            draw = rng.random()
            if draw < 0.25:
                record[feature] = np.nan
            elif feature in CATEGORICAL:
                record[feature] = float(rng.choice([0, 1, 2, 3, 4, 6, 7, 9]))
            else:
                record[feature] = float(rng.normal(60, 40))
        records.append(record)
    frame = pd.DataFrame(records)
    return frame.astype(object) if as_object else frame


def _adversarial_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        [{feature: None for feature in PREVENTION_FEATURES} for _ in range(4)],
        dtype=object,
    )
    frame.loc[0, "DEMO_RIAGENDR"] = 9          # unseen categorical level
    frame.loc[1, "smoking_status"] = "1"       # numeric-looking string level
    frame.loc[1, "DEMO_RIDAGEYR"] = "55"       # numeric-looking string value
    frame.loc[2, "alcohol_status"] = 2.0
    frame.loc[3, "DEMO_RIDRETH3"] = "3"
    return frame


def _assert_identical(frame: pd.DataFrame) -> None:
    expected = sklearn_score_records(frame, config.SSL_ARTIFACT_DIR)
    actual = inference.score_records(frame, config.SSL_ARTIFACT_DIR)
    assert len(actual) == len(expected)
    for index, (want, got) in enumerate(zip(expected, actual)):
        assert got == want, f"row {index} differs: {want} != {got}"


def test_exported_parameters_are_present():
    assert inference.params_available() is True
    params = inference._load_params(str(config.SSL_ARTIFACT_DIR))
    kinds = [block["kind"] for block in params["blocks"]]
    assert kinds == ["numeric", "categorical"]
    numeric = params["blocks"][0]
    assert numeric["add_indicator"] is True
    assert len(numeric["centre"]) == len(numeric["scale"])
    assert len(numeric["centre"]) == len(numeric["columns"]) + len(
        numeric["indicator_columns"]
    )
    assert params["feature_names_in"] == PREVENTION_FEATURES


def test_transformed_dimension_matches_metadata():
    frame = _random_frame(3, seed=1, as_object=False)
    params = inference._load_params(str(config.SSL_ARTIFACT_DIR))
    matrix = inference.transform(frame[PREVENTION_FEATURES], params)
    from server.model import artifact_metadata

    assert matrix.shape == (3, artifact_metadata()["transformed_dimension"])


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize("rows,seed", [(1, 11), (7, 12), (200, 13)])
def test_random_frames_match_the_sklearn_path(rows, seed, as_object):
    _assert_identical(_random_frame(rows, seed, as_object))


def test_adversarial_frame_matches_the_sklearn_path():
    _assert_identical(_adversarial_frame())


def test_all_missing_and_partial_records_match():
    frame = pd.DataFrame(
        [{feature: np.nan for feature in PREVENTION_FEATURES} for _ in range(3)]
    )
    frame.loc[1, "DEMO_RIAGENDR"] = 2.0
    frame.loc[2, "DEMO_RIDAGEYR"] = 44.0
    _assert_identical(frame)


def test_csv_fixture_matches_the_sklearn_path():
    frame = pd.read_csv(config.APP_ROOT / "fixtures" / "safe_deidentified_cohort.csv")
    _assert_identical(frame.reindex(columns=PREVENTION_FEATURES))


def test_probe_record_matches_the_sklearn_path():
    record = {
        "DEMO_RIDAGEYR": 61,
        "DEMO_RIAGENDR": 1,
        "BMX_BMXBMI": 33.4,
        "GHB_LBXGH": 6.4,
        "GLU_LBXGLU": 128,
        "INS_LBXIN": 21.5,
        "homa_ir": 6.8,
    }
    _assert_identical(pd.DataFrame([record]))


def test_fallback_used_when_exported_parameters_are_missing(tmp_path, monkeypatch):
    """With no exported constants the vendored scikit-learn path is used."""
    import shutil

    for name in ("metadata.json", "autoencoder_weights.npz", "preprocessor.joblib"):
        shutil.copy(config.SSL_ARTIFACT_DIR / name, tmp_path / name)
    assert inference.params_available(tmp_path) is False
    frame = _random_frame(4, seed=21, as_object=False)
    assert inference.score_records(frame, tmp_path) == sklearn_score_records(
        frame, tmp_path
    )


def test_full_runtime_works_with_sklearn_scipy_and_joblib_blocked(tmp_path):
    """Emulate the trimmed serverless runtime: import the app, score, upload.

    A subprocess installs an import blocker for scikit-learn, SciPy, joblib and
    PyTorch, then exercises the real request paths. This is the check that the
    Vercel function still works with the trimmed requirements file.
    """
    import json
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys, json

        BLOCKED = {"sklearn", "scipy", "joblib", "torch", "self_supervised", "data_reliability"}

        class Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked for test: {name}")
                return None

        sys.meta_path.insert(0, Blocker())
        sys.path.insert(0, APP_ROOT)

        import pandas as pd
        from fastapi.testclient import TestClient
        from server import config, core_bridge, dataset, model
        from server.app import app

        assert core_bridge.VENDORED_MODULES_AVAILABLE is False

        probe = model.score_single_record(
            {"DEMO_RIDAGEYR": 61, "BMX_BMXBMI": 33.4, "GHB_LBXGH": 6.4,
             "GLU_LBXGLU": 128, "INS_LBXIN": 21.5, "homa_ir": 6.8}
        )

        frame = pd.read_csv(config.APP_ROOT / "fixtures" / "safe_deidentified_cohort.csv")
        intake = dataset.build_intake_report(frame, "safe.csv")
        mask = dataset.eligible_row_mask(frame, intake["schema"]["mapped_features"])
        batch = model.score_dataset(frame, intake["schema"]["mapped_features"], mask)
        csv_text = model.results_csv(batch["rows"])

        with TestClient(app, base_url="https://testserver") as client:
            health = client.get("/api/v1/health").status_code
            anonymous = client.get("/api/v1/model").status_code
            login = client.post("/api/v1/auth/login", json={"access_key": ACCESS_KEY})
            authorised = client.get("/api/v1/model").json()

        print(json.dumps({
            "probe_score": probe["score"]["metabolic_deviation_score"],
            "probe_percentile": probe["score"]["reference_percentile"],
            "rows_scored": batch["aggregate"]["rows_scored"],
            "median_percentile": batch["aggregate"]["reference_percentile"]["median"],
            "csv_lines": len(csv_text.strip().splitlines()),
            "intake_accepted": intake["rows"]["accepted"],
            "health": health,
            "anonymous_model": anonymous,
            "login": login.status_code,
            "preprocessor_path": authorised["preprocessor_path"],
            "heavy_modules": sorted(m for m in ("sklearn", "scipy", "joblib", "torch")
                                    if m in sys.modules),
        }))
        """
    )
    from conftest import TEST_ACCESS_KEY

    preamble = (
        f"APP_ROOT = {str(config.APP_ROOT)!r}\nACCESS_KEY = {TEST_ACCESS_KEY!r}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", preamble + script],
        capture_output=True,
        text=True,
        cwd=str(config.APP_ROOT),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["heavy_modules"] == []
    assert payload["preprocessor_path"] == "exported_constants"
    assert payload["health"] == 200
    assert payload["anonymous_model"] == 401
    assert payload["login"] == 200
    assert payload["intake_accepted"] == 238
    assert payload["rows_scored"] == 238
    assert payload["csv_lines"] == 239
    assert 0 <= payload["median_percentile"] <= 100

    # Identical numbers from the full profile in this process.
    expected = inference.score_records(
        pd.DataFrame(
            [{"DEMO_RIDAGEYR": 61, "BMX_BMXBMI": 33.4, "GHB_LBXGH": 6.4,
              "GLU_LBXGLU": 128, "INS_LBXIN": 21.5, "homa_ir": 6.8}]
        )
    )[0]
    assert payload["probe_score"] == expected["metabolic_deviation_score"]
    assert payload["probe_percentile"] == expected["reference_percentile"]
