"""Deterministic batched Synthea generation with immediate conversion and raw cleanup.

Each batch: generate with a recorded seed -> trim raw CSVs to the four needed tables ->
convert to the MetaboGuard compact longitudinal schema -> hash and record -> delete raw files
before the next batch starts. This keeps peak disk bounded while accumulating enough unique
synthetic patients to test the 50-event gates.

Usage (sandbox, official pinned jar required)::

    python synthea_batch_runner.py --jar /tmp/synthea.jar --batches 10 --per-batch 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_longitudinal import SYNTHEA_CONFIG, convert_synthea_csv  # noqa: E402

KEEP_TABLES = ("patients.csv", "observations.csv", "conditions.csv", "encounters.csv")
OBSERVATION_CODES = (
    "4548-4",
    "2339-0",
    "2093-3",
    "2085-9",
    "2571-8",
    "39156-5",
    "29463-7",
    "8480-6",
    "56086-2",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_batch(jar: Path, output_root: Path, seed: int, population: int, index: int) -> dict:
    raw = output_root / f"raw_batch_{index:02d}"
    if raw.exists():
        shutil.rmtree(raw)
    started = time.perf_counter()
    command = [
        "java",
        "-Xmx3g",
        "-jar",
        str(jar),
        "-s",
        str(seed),
        "-cs",
        str(seed),
        "-r",
        str(seed),
        "-p",
        str(population),
        "-a",
        SYNTHEA_CONFIG["age_filter"],
        "--exporter.csv.export",
        "true",
        "--exporter.fhir.export",
        "false",
        "--exporter.hospital.fhir.export",
        "false",
        "--exporter.practitioner.fhir.export",
        "false",
        "--exporter.baseDirectory",
        str(raw),
        SYNTHEA_CONFIG["state"],
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return {
            "batch": index,
            "seed": seed,
            "status": "generation_failed",
            "stderr_tail": (completed.stderr or "")[-800:],
        }

    csv_dir = raw / "csv"
    trimmed = output_root / f"trim_batch_{index:02d}"
    trimmed.mkdir(parents=True, exist_ok=True)
    # Keep only the tables the schema needs, and only the mapped observation codes.
    header = None
    with (csv_dir / "observations.csv").open() as source, (
        trimmed / "observations.csv"
    ).open("w") as target:
        for line_number, line in enumerate(source):
            if line_number == 0:
                header = line
                target.write(line)
                continue
            if any(f",{code}," in line for code in OBSERVATION_CODES):
                target.write(line)
    for table in KEEP_TABLES:
        if table == "observations.csv":
            continue
        shutil.copy(csv_dir / table, trimmed / table)
    raw_sizes = {path.name: path.stat().st_size for path in csv_dir.glob("*.csv")}
    shutil.rmtree(raw)  # raw batch deleted before the next batch starts

    events = convert_synthea_csv(trimmed)
    compact = output_root / f"events_batch_{index:02d}.csv"
    events.to_csv(compact, index=False)
    shutil.rmtree(trimmed)

    record = {
        "batch": index,
        "seed": seed,
        "status": "ok",
        "population_requested": population,
        "patients_converted": int(events["patient_id"].nunique()),
        "rows": int(len(events)),
        "compact_path": str(compact),
        "compact_sha256": sha256_file(compact),
        "raw_bytes_deleted": int(sum(raw_sizes.values())),
        "seconds": round(time.perf_counter() - started, 1),
        "jar_sha256": SYNTHEA_CONFIG["jar_sha256"],
        "runtime": SYNTHEA_CONFIG["recorded_runtime"],
        "release": SYNTHEA_CONFIG["pinned_release"],
        "age_filter": SYNTHEA_CONFIG["age_filter"],
        "state": SYNTHEA_CONFIG["state"],
        "header_columns": (header or "").strip(),
    }
    (output_root / f"batch_{index:02d}.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar", required=True)
    parser.add_argument("--output-root", default="/tmp/synthea_batches")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--per-batch", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=2026080500)
    parser.add_argument("--start-index", type=int, default=1)
    arguments = parser.parse_args()

    output_root = Path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for offset in range(arguments.batches):
        index = arguments.start_index + offset
        seed = arguments.base_seed + index
        record = run_batch(Path(arguments.jar), output_root, seed, arguments.per_batch, index)
        print(json.dumps(record), flush=True)
        if record.get("status") != "ok":
            break


if __name__ == "__main__":
    main()