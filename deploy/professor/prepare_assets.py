#!/usr/bin/env python3
"""Materialise `deploy/professor/assets/` from the authoritative KERI repository.

The deployment needs a small set of runtime assets: the exported NumPy inference
artifact, the research run's reliability/integrity/clustering reports, and the
biomarker evidence catalogue. They are not duplicated in version control - this
script copies them out of the repository and writes the trimmed metadata the
deployment uses.

Run from the repository root (or anywhere; paths are resolved from this file):

    .venv/bin/python deploy/professor/prepare_assets.py

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

ARTIFACT_DIR = REPO_ROOT / "model_artifacts" / "metaboguard_ssl" / "nhanes_multicycle_v2"
RESEARCH_RUN = REPO_ROOT / "model_artifacts" / "research_runs" / "research__20260804T164743Z"
EVIDENCE_SRC = REPO_ROOT / "data" / "evidence" / "biomarker_evidence.json"

#: Authoritative research modules vendored into ``server/core/`` unchanged.
CORE_MODULES = (
    "self_supervised.py",
    "data_integrity.py",
    "data_reliability.py",
    "evidence_catalogue.py",
)

#: Frontend modules vendored unchanged from the existing dashboard.
CLIENT_MODULES = ("synthetic_patient.js", "synthetic_patient.test.js")

REFERENCE_GRID_POINTS = 8001

_ABSOLUTE_PATH = re.compile(r"(?:/Volumes|/Users|/home|/private|/var/folders|[A-Za-z]:\\\\)[^\s\"']*")


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

    for required in (ARTIFACT_DIR, RESEARCH_RUN, EVIDENCE_SRC):
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
    for name in ("preprocessor.joblib", "autoencoder_weights.npz", "README.md", "sample_input.json"):
        shutil.copy(ARTIFACT_DIR / name, artifact_out / name)
        print(f"  copied {name}")

    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
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

    print("\nassets ready. Next: pip install -r requirements-deploy.txt, "
          "cd client && npm ci && npm run build, then ./start.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
