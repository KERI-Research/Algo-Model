#!/usr/bin/env python3
"""Materialise `deploy/professor/assets/` from the authoritative KERI repository.

The deployment needs a small set of runtime assets: the exported NumPy inference
artifact, the research run's reliability/integrity/clustering reports, and the
biomarker evidence catalogue. They are not duplicated in version control - this
script copies them out of the repository and writes the trimmed metadata the
deployment uses.

Run from the repository root (or anywhere; paths are resolved from this file):

    .venv/bin/python deploy/professor/prepare_assets.py

It also exports two derived files that let the serverless runtime drop
scikit-learn, SciPy and joblib entirely:

* ``assets/ssl_artifact/preprocessor_params.json`` - the fitted preprocessor
  constants, replayed with NumPy by ``server/inference.py``;
* ``assets/research_constants.json`` - the feature allowlist, plausibility ranges
  and reliability thresholds, read by ``server/core_bridge.py`` when the vendored
  scikit-learn modules are unavailable.

Two deliberate transformations:

1. ``metadata.json`` keeps every field of the authoritative artifact, but the
   63,041-value reference score distribution is replaced by an 8,001-point
   quantile grid of the same distribution (percentile resolution 0.0125 pp).
   Weights, preprocessor and score definition are untouched, and probe scores
   match the authoritative artifact to six decimal places.
2. Absolute filesystem paths from the authoring machine are stripped from every
   bundled JSON, so no local path can reach a client.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

DEPLOY_DIR = Path(__file__).resolve().parent
#: Repository root. Override with KERI_REPO_ROOT when the deployment copy lives
#: outside the repository (for example a read-only checkout or a build sandbox).
REPO_ROOT = Path(os.environ.get("KERI_REPO_ROOT", DEPLOY_DIR.parent.parent)).resolve()

LEGACY_ARTIFACT_DIR = REPO_ROOT / "model_artifacts" / "metaboguard_ssl" / "nhanes_multicycle_v2"
SSL_POINTER = REPO_ROOT / "model_artifacts" / "metaboguard_ssl" / "CURRENT.json"
RESEARCH_RUN = REPO_ROOT / "model_artifacts" / "research_runs" / "research__20260804T164743Z"
EVIDENCE_SRC = REPO_ROOT / "data" / "evidence" / "biomarker_evidence.json"
FUTURE_RISK_ARTIFACT = (
    REPO_ROOT
    / "model_artifacts"
    / "future_risk"
    / "simulation_synthea_pooled"
    / "artifact"
)

#: Authoritative research modules vendored into ``server/core/`` unchanged.
CORE_MODULES = (
    "self_supervised.py",
    "data_integrity.py",
    "data_reliability.py",
    "evidence_catalogue.py",
    # Simulation-only future-risk sources. Vendored for the offline export and for
    # reference; the request path uses the portable artifact, not these modules.
    "future_risk_models.py",
    "longitudinal_schema.py",
    "longitudinal_dataset.py",
)

#: Frontend modules vendored unchanged from the existing dashboard.
CLIENT_MODULES = (
    "synthetic_patient.js",
    "synthetic_patient.test.js",
    "synthetic_prevention.js",
    "synthetic_profile_model.json",
    "synthetic_history.js",
)

REFERENCE_GRID_POINTS = 8001

_ABSOLUTE_PATH = re.compile(r"(?:/Volumes|/Users|/home|/private|/var/folders|[A-Za-z]:\\\\)[^\s\"']*")


def resolve_ssl_artifact() -> Path:
    if not SSL_POINTER.exists():
        return LEGACY_ARTIFACT_DIR
    payload = json.loads(SSL_POINTER.read_text())
    candidate = Path(payload["artifact_dir"])
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    artifact_root = (REPO_ROOT / "model_artifacts" / "metaboguard_ssl").resolve()
    if candidate != artifact_root and artifact_root not in candidate.parents:
        raise ValueError("Promoted SSL artifact must stay inside model_artifacts/metaboguard_ssl.")
    if not (candidate / "metadata.json").exists():
        raise FileNotFoundError(f"Promoted SSL artifact is incomplete: {candidate}")
    return candidate


def sanitise(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitise(item) for item in value]
    if isinstance(value, str) and _ABSOLUTE_PATH.search(value):
        return _ABSOLUTE_PATH.sub(
            lambda match: match.group(0).replace("\\\\", "/").rsplit("/", 1)[-1], value
        )
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(sanitise(payload), indent=1))
    print(f"  wrote {path.relative_to(DEPLOY_DIR)} ({path.stat().st_size:,} bytes)")


def main() -> int:
    import numpy as np

    artifact_dir = resolve_ssl_artifact()

    for required in (artifact_dir, RESEARCH_RUN, EVIDENCE_SRC, FUTURE_RISK_ARTIFACT):
        if not required.exists():
            print(f"missing source: {required}", file=sys.stderr)
            return 1

    artifact_out = DEPLOY_DIR / "assets" / "ssl_artifact"
    reports_out = DEPLOY_DIR / "assets" / "reports"
    evidence_out = DEPLOY_DIR / "assets" / "evidence"
    for directory in (artifact_out, reports_out, evidence_out):
        directory.mkdir(parents=True, exist_ok=True)

    core_out = DEPLOY_DIR / "server" / "core"
    core_out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(DEPLOY_DIR))
    print("vendored research modules (byte-for-byte from api/):")
    for name in CORE_MODULES:
        shutil.copy(REPO_ROOT / "api" / name, core_out / name)
        print(f"  copied api/{name}")

    client_out = DEPLOY_DIR / "client" / "src" / "lib"
    client_out.mkdir(parents=True, exist_ok=True)
    print("vendored frontend modules (byte-for-byte from frontend/src/interface/):")
    for name in CLIENT_MODULES:
        shutil.copy(REPO_ROOT / "frontend" / "src" / "interface" / name, client_out / name)
        print(f"  copied frontend/src/interface/{name}")

    print("model artifact:")
    for name in ("preprocessor.joblib", "autoencoder_weights.npz"):
        shutil.copy(artifact_dir / name, artifact_out / name)
        print(f"  copied {name}")
    support_files = {
        "MODEL_CARD.md": artifact_dir / "MODEL_CARD.md",
        "README.md": (
            artifact_dir / "README.md"
            if (artifact_dir / "README.md").exists()
            else artifact_dir / "MODEL_CARD.md"
        ),
        "sample_input.json": (
            artifact_dir / "sample_input.json"
            if (artifact_dir / "sample_input.json").exists()
            else LEGACY_ARTIFACT_DIR / "sample_input.json"
        ),
        "promotion_report.json": artifact_dir / "promotion_report.json",
    }
    for name, source in support_files.items():
        destination = artifact_out / name
        if destination.exists():
            destination.unlink()
        if source.exists():
            shutil.copy(source, destination)
            print(f"  copied {name}")

    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    distribution = metadata["score_distribution"]
    reference = np.asarray(distribution["combined_sorted"], dtype=np.float64)
    grid = np.quantile(reference, np.linspace(0.0, 1.0, REFERENCE_GRID_POINTS))
    distribution["combined_sorted"] = [float(value) for value in grid]
    distribution["combined_sorted_note"] = (
        "Deployment copy: the reference distribution is an "
        f"{REFERENCE_GRID_POINTS}-point quantile grid of the full {reference.size}-row "
        "training reference distribution (percentile resolution 0.0125 pp). Weights, "
        "preprocessor and score definition are unchanged."
    )
    write_json(artifact_out / "metadata.json", metadata)

    print("research reports:")
    for name in ("data_reliability_report.json", "data_integrity_report.json"):
        write_json(reports_out / name, json.loads((RESEARCH_RUN / name).read_text()))
    for variant in ("complete_cases", "all_adults"):
        source = RESEARCH_RUN / f"clustering_{variant}" / "clustering_report.json"
        write_json(reports_out / f"clustering_{variant}.json", json.loads(source.read_text()))

    print("evidence catalogue:")
    write_json(evidence_out / "biomarker_evidence.json", json.loads(EVIDENCE_SRC.read_text()))

    print("future-risk portable artifact (simulation only):")
    from server.future_risk_export import export as export_future_risk

    report = export_future_risk(FUTURE_RISK_ARTIFACT, DEPLOY_DIR / "assets" / "future_risk")
    print(
        f"  parity verdict={report['verdict']} "
        f"max_abs_difference={report['max_abs_difference']:.3e} "
        f"histories={report['histories_compared']}"
    )
    if report["verdict"] != "parity":
        print("  REFUSING: portable scoring does not match the authoritative artifact.")
        return 1

    print("serverless exports (NumPy-only runtime):")
    from server.inference import export_preprocessor_params  # noqa: E402
    from server.research_constants import export_constants  # noqa: E402

    write_json(artifact_out / "preprocessor_params.json", export_preprocessor_params(artifact_out))
    write_json(DEPLOY_DIR / "assets" / "research_constants.json", export_constants())

    print("\nassets ready. Next: pip install -r requirements-deploy.txt, "
          "cd client && npm ci && npm run build, then ./start.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
