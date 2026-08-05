"""Synthetic longitudinal cohort generation for MetaboGuard (simulation only).

Two paths, in order of preference:

1. **Official Synthea** (`synthetichealth/synthea`, Apache-2.0,
   https://github.com/synthetichealth/synthea). Synthea simulates synthetic patients across
   their lifetime and exports CSV/FHIR (Walonoski et al., JAMIA 2018,
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7651916/, DOI 10.1093/jamia/ocx079). This module
   pins the version/seed/config, runs it when Java and the jar are available, and converts
   its CSV export into the MetaboGuard longitudinal schema.
2. **Deterministic in-repo simulator** (`simulate_longitudinal_cohort`) used when Synthea
   cannot run in the current environment - for example no Java runtime or no network to
   fetch the release. It is a transparent, seeded generative model, not a clinical model.

Both paths produce **simulation-only** data. Synthea's own validity work shows synthetic
records reproduce some population characteristics while diverging from real distributions
(Chen et al., BMC Med Inform Decis Mak 2019,
https://pmc.ncbi.nlm.nih.gov/articles/PMC6416981/, DOI 10.1186/s12911-019-0793-0), so
nothing generated here can establish real-world calibration, clinical utility or early
detection. It exists to prove the software and the protocol are correct.

Event enrichment is explicit: when incidence is boosted to reach the 50-event gate, the
sampling stratum and weight are recorded per patient and the manifest states that raw
predicted probabilities are not population-calibrated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from longitudinal_schema import (
    CapabilityState,
    DatasetManifest,
    PREVENTION_SAFE_FEATURES,
    SCHEMA_VERSION,
    frame_fingerprint,
    validate_event_frame,
    write_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_longitudinal"

# ---------------------------------------------------------------------------
# Synthea configuration (pinned)
# ---------------------------------------------------------------------------

SYNTHEA_CONFIG: dict[str, Any] = {
    "repository": "https://github.com/synthetichealth/synthea",
    "licence": "Apache-2.0",
    "pinned_release": "v3.3.0",
    "pinned_jar": "synthea-with-dependencies.jar",
    "jar_download": (
        "https://github.com/synthetichealth/synthea/releases/download/v3.3.0/"
        "synthea-with-dependencies.jar"
    ),
    "methodology_paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7651916/",
    "methodology_doi": "10.1093/jamia/ocx079",
    "validity_limitations": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6416981/",
    "validity_doi": "10.1186/s12911-019-0793-0",
    "seed": 20260805,
    "clinician_seed": 20260805,
    "population": 20000,
    "state": "Massachusetts",
    "modules": ["metabolic_syndrome_care", "diabetes", "lung_cancer", "colorectal_cancer", "breast_cancer"],
    "export_settings": {
        "exporter.csv.export": True,
        "exporter.fhir.export": False,
        "exporter.years_of_history": 0,
        "generate.only_alive_patients": False,
    },
    "command_template": (
        "java -jar {jar} -s {seed} -cs {clinician_seed} -p {population} "
        "--exporter.csv.export true --exporter.fhir.export false "
        "--exporter.years_of_history 0 --exporter.baseDirectory {output} {state}"
    ),
}

#: Synthea condition codes (SNOMED CT) mapped to MetaboGuard outcomes.
SYNTHEA_OUTCOME_CODES: dict[str, dict[str, Any]] = {
    "type2_diabetes": {
        "snomed": ["44054006"],
        "description": "Diabetes mellitus type 2 (disorder)",
        "outcome_type": "type2_diabetes",
    },
    "pan_cancer": {
        "snomed": [
            "254637007",  # Non-small cell lung cancer
            "254632001",  # Small cell carcinoma of lung
            "363406005",  # Malignant tumor of colon
            "254837009",  # Malignant neoplasm of breast
            "126906006",  # Neoplasm of prostate
            "93761005",  # Primary malignant neoplasm of colon
        ],
        "description": "Any malignant neoplasm recorded by Synthea's oncology modules",
        "outcome_type": "pan_cancer",
    },
}

#: Site mapping, used only when a site clears the event gate downstream.
SYNTHEA_SITE_CODES: dict[str, str] = {
    "254637007": "lung",
    "254632001": "lung",
    "363406005": "colorectal",
    "93761005": "colorectal",
    "254837009": "breast",
    "126906006": "prostate",
}

#: Synthea observation LOINC codes mapped to MetaboGuard feature codes and units.
SYNTHEA_OBSERVATION_CODES: dict[str, dict[str, str]] = {
    "4548-4": {"feature_code": "GHB_LBXGH", "unit": "%"},
    "2339-0": {"feature_code": "GLU_LBXGLU", "unit": "mg/dL"},
    "2093-3": {"feature_code": "TCHOL_LBXTC", "unit": "mg/dL"},
    "2085-9": {"feature_code": "HDL_LBDHDD", "unit": "mg/dL"},
    "2571-8": {"feature_code": "TRIGLY_LBXTR", "unit": "mg/dL"},
    "39156-5": {"feature_code": "BMX_BMXBMI", "unit": "kg/m2"},
    "29463-7": {"feature_code": "BMX_BMXWT", "unit": "kg"},
    "8480-6": {"feature_code": "BPX_SYSTOLIC", "unit": "mmHg"},
    "56086-2": {"feature_code": "BMX_BMXWAIST", "unit": "cm"},
}


def synthea_availability(jar_path: str | Path | None = None) -> dict[str, Any]:
    """Report whether official Synthea can run here, and why not when it cannot."""
    java = shutil.which("java")
    java_works = False
    java_version = None
    if java:
        try:
            completed = subprocess.run(
                [java, "-version"], capture_output=True, text=True, timeout=20
            )
            java_version = (completed.stderr or completed.stdout).strip().splitlines()[:1]
            java_works = completed.returncode == 0
        except Exception as error:  # pragma: no cover - environment dependent
            java_version = [str(error)[:120]]
    candidate = Path(jar_path) if jar_path else PROJECT_ROOT / "vendor" / SYNTHEA_CONFIG["pinned_jar"]
    reasons: list[str] = []
    if not java:
        reasons.append("No java executable on PATH.")
    elif not java_works:
        reasons.append("java is present but no working JRE (java -version failed).")
    if not candidate.exists():
        reasons.append(f"Synthea jar not found at {candidate}.")
    return {
        "synthea_available": bool(java_works and candidate.exists()),
        "java_path": java,
        "java_version": java_version,
        "jar_path": str(candidate),
        "jar_present": candidate.exists(),
        "reasons_unavailable": reasons,
        "config": SYNTHEA_CONFIG,
        "how_to_enable": (
            "Install a JRE (for example `brew install openjdk@21`), download "
            f"{SYNTHEA_CONFIG['jar_download']} into vendor/, then re-run with "
            "--generator synthea. The pinned seed makes the cohort reproducible."
        ),
    }


def run_synthea(output_dir: str | Path, jar_path: str | Path | None = None) -> dict[str, Any]:
    """Run pinned Synthea into ``output_dir``. Raises when it cannot run."""
    availability = synthea_availability(jar_path)
    if not availability["synthea_available"]:
        raise RuntimeError(
            "Official Synthea cannot run in this environment: "
            + " ".join(availability["reasons_unavailable"])
            + " " + availability["how_to_enable"]
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = SYNTHEA_CONFIG["command_template"].format(
        jar=availability["jar_path"],
        seed=SYNTHEA_CONFIG["seed"],
        clinician_seed=SYNTHEA_CONFIG["clinician_seed"],
        population=SYNTHEA_CONFIG["population"],
        output=str(output),
        state=SYNTHEA_CONFIG["state"],
    )
    started = datetime.now(UTC)
    completed = subprocess.run(command, shell=True, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Synthea failed: {completed.stderr[-2000:]}")
    return {
        "command": command,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output),
        "config": SYNTHEA_CONFIG,
    }


def convert_synthea_csv(
    csv_dir: str | Path, index_offset_days: int = 0
) -> pd.DataFrame:
    """Convert a Synthea CSV export into the MetaboGuard patient-event schema.

    Only observations dated at or before each patient's index date become model inputs; the
    caller sets index dates, so this function emits every observation with its timestamp and
    lets `longitudinal_dataset` apply the endpoint protocol.
    """
    directory = Path(csv_dir)
    patients = pd.read_csv(directory / "patients.csv")
    observations = pd.read_csv(directory / "observations.csv")
    conditions = pd.read_csv(directory / "conditions.csv")
    encounters = pd.read_csv(directory / "encounters.csv", usecols=["Id", "PATIENT", "START"])
    provenance = f"synthea:{SYNTHEA_CONFIG['pinned_release']}:seed={SYNTHEA_CONFIG['seed']}"

    observations = observations[observations["CODE"].astype(str).isin(SYNTHEA_OBSERVATION_CODES)]
    mapped = observations.assign(
        feature_code=observations["CODE"].astype(str).map(
            lambda code: SYNTHEA_OBSERVATION_CODES[code]["feature_code"]
        ),
        unit=observations["CODE"].astype(str).map(
            lambda code: SYNTHEA_OBSERVATION_CODES[code]["unit"]
        ),
    )
    event_rows = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "patient_id": mapped["PATIENT"],
            "observation_timestamp": pd.to_datetime(mapped["DATE"], utc=True, errors="coerce"),
            "source": "synthea",
            "feature_code": mapped["feature_code"],
            "value": pd.to_numeric(mapped["VALUE"], errors="coerce"),
            "unit": mapped["unit"],
            "missingness_reason": "observed",
            "visit_id": mapped.get("ENCOUNTER"),
            "index_date": pd.NaT,
            "outcome_type": "none",
            "event_date": pd.NaT,
            "cancer_site": None,
            "cancer_stage": None,
            "censoring_date": pd.NaT,
            "provenance": provenance,
        }
    )

    outcome_frames = [event_rows]
    for outcome, spec in SYNTHEA_OUTCOME_CODES.items():
        selected = conditions[conditions["CODE"].astype(str).isin(spec["snomed"])]
        if selected.empty:
            continue
        first = (
            selected.assign(START=pd.to_datetime(selected["START"], utc=True, errors="coerce"))
            .sort_values("START")
            .groupby("PATIENT", as_index=False)
            .first()
        )
        outcome_frames.append(
            pd.DataFrame(
                {
                    "schema_version": SCHEMA_VERSION,
                    "patient_id": first["PATIENT"],
                    "observation_timestamp": first["START"],
                    "source": "synthea",
                    "feature_code": f"outcome:{outcome}",
                    "value": 1.0,
                    "unit": None,
                    "missingness_reason": "observed",
                    "visit_id": None,
                    "index_date": pd.NaT,
                    "outcome_type": spec["outcome_type"],
                    "event_date": first["START"],
                    "cancer_site": first["CODE"].astype(str).map(SYNTHEA_SITE_CODES)
                    if outcome == "pan_cancer"
                    else None,
                    "cancer_stage": None,
                    "censoring_date": pd.NaT,
                    "provenance": provenance,
                }
            )
        )

    deaths = patients[patients["DEATHDATE"].notna()]
    if not deaths.empty:
        outcome_frames.append(
            pd.DataFrame(
                {
                    "schema_version": SCHEMA_VERSION,
                    "patient_id": deaths["Id"],
                    "observation_timestamp": pd.to_datetime(deaths["DEATHDATE"], utc=True),
                    "source": "synthea",
                    "feature_code": "outcome:death",
                    "value": 1.0,
                    "unit": None,
                    "missingness_reason": "observed",
                    "visit_id": None,
                    "index_date": pd.NaT,
                    "outcome_type": "death",
                    "event_date": pd.to_datetime(deaths["DEATHDATE"], utc=True),
                    "cancer_site": None,
                    "cancer_stage": None,
                    "censoring_date": pd.NaT,
                    "provenance": provenance,
                }
            )
        )
    del encounters, index_offset_days
    return pd.concat(outcome_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Deterministic in-repo simulator (fallback)
# ---------------------------------------------------------------------------


@dataclass
class SimulatorConfig:
    """Transparent generative configuration. Every number here is an engineering choice."""

    patients: int = 4000
    seed: int = 20260805
    #: Simulated calendar window.
    start_date: str = "2008-01-01"
    index_window_years: tuple[int, int] = (2012, 2016)
    max_follow_up_years: float = 7.0
    #: Visits before the index date.
    min_visits: int = 3
    max_visits: int = 10
    visit_interval_days: tuple[int, int] = (150, 500)
    #: Per-visit probability a given laboratory feature is measured.
    measurement_probability: dict[str, float] = field(
        default_factory=lambda: {
            "DEMO_RIDAGEYR": 1.0,
            "BMX_BMXBMI": 0.95,
            "BMX_BMXWT": 0.9,
            "BMX_BMXWAIST": 0.55,
            "BPX_SYSTOLIC": 0.9,
            "GHB_LBXGH": 0.7,
            "GLU_LBXGLU": 0.65,
            "INS_LBXIN": 0.3,
            "TCHOL_LBXTC": 0.6,
            "HDL_LBDHDD": 0.6,
            "TRIGLY_LBXTR": 0.55,
        }
    )
    #: Explicit enrichment strata. `weight` is the inverse sampling weight to report.
    enrichment: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "population_typical": {"share": 0.55, "diabetes_multiplier": 1.0, "cancer_multiplier": 1.0, "weight": 1.0},
            "metabolic_high_risk": {"share": 0.30, "diabetes_multiplier": 6.0, "cancer_multiplier": 2.5, "weight": 0.18},
            "older_high_risk": {"share": 0.15, "diabetes_multiplier": 3.0, "cancer_multiplier": 6.0, "weight": 0.12},
        }
    )
    #: Baseline yearly hazards before multipliers (engineering values, not epidemiology).
    baseline_yearly_hazard: dict[str, float] = field(
        default_factory=lambda: {"type2_diabetes": 0.010, "pan_cancer": 0.006, "death": 0.008}
    )
    site_mix: dict[str, float] = field(
        default_factory=lambda: {"lung": 0.28, "colorectal": 0.26, "breast": 0.24, "prostate": 0.22}
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "patients": self.patients,
            "seed": self.seed,
            "start_date": self.start_date,
            "index_window_years": list(self.index_window_years),
            "max_follow_up_years": self.max_follow_up_years,
            "visits": {"min": self.min_visits, "max": self.max_visits, "interval_days": list(self.visit_interval_days)},
            "measurement_probability": self.measurement_probability,
            "enrichment_strata": self.enrichment,
            "baseline_yearly_hazard": self.baseline_yearly_hazard,
            "site_mix": self.site_mix,
        }


def _draw_stratum(random: np.random.Generator, config: SimulatorConfig) -> tuple[str, dict[str, float]]:
    names = list(config.enrichment)
    shares = np.array([config.enrichment[name]["share"] for name in names], dtype=float)
    shares = shares / shares.sum()
    chosen = names[int(random.choice(len(names), p=shares))]
    return chosen, config.enrichment[chosen]


def simulate_longitudinal_cohort(
    config: SimulatorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate a deterministic simulated cohort in the patient-event schema.

    Returns (events, patient_strata, generator_metadata). The trajectory model is simple and
    stated openly: a latent metabolic burden drives both the observed trajectory (rising
    HbA1c/glucose/BMI) and the event hazards, so a temporal model has real signal to find
    while nothing about the numbers is claimed to be clinically realistic.
    """
    config = config or SimulatorConfig()
    random = np.random.default_rng(config.seed)
    start = pd.Timestamp(config.start_date, tz="UTC")
    provenance = f"metaboguard_simulator:v1:seed={config.seed}"

    event_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []

    for patient_number in range(config.patients):
        patient_id = f"SIMPT{patient_number:06d}"
        stratum_name, stratum = _draw_stratum(random, config)
        sex = int(random.integers(1, 3))
        age_at_index = float(random.normal(58 if stratum_name == "older_high_risk" else 50, 11))
        age_at_index = float(np.clip(age_at_index, 25, 88))
        burden = float(
            np.clip(
                random.normal(
                    0.2 if stratum_name == "population_typical" else 0.75, 0.28
                ),
                0.0,
                1.6,
            )
        )
        index_year = int(random.integers(config.index_window_years[0], config.index_window_years[1] + 1))
        index_date = pd.Timestamp(f"{index_year}-{int(random.integers(1, 13)):02d}-15", tz="UTC")

        visit_count = int(random.integers(config.min_visits, config.max_visits + 1))
        gaps = random.integers(config.visit_interval_days[0], config.visit_interval_days[1], size=visit_count)
        offsets = np.cumsum(gaps[::-1])[::-1]
        visit_dates = [index_date - timedelta(days=int(offset)) for offset in offsets]
        visit_dates = [date for date in visit_dates if date >= start] or [index_date - timedelta(days=180)]

        for visit_index, visit_date in enumerate(visit_dates):
            years_before_index = float((index_date - visit_date).days) / 365.25
            trend = burden * (1.0 - years_before_index / 8.0)
            values = {
                "DEMO_RIDAGEYR": age_at_index - years_before_index,
                "BMX_BMXBMI": 26.5 + 6.0 * trend + random.normal(0, 1.1),
                "BMX_BMXWT": 78 + 18 * trend + random.normal(0, 4.0),
                "BMX_BMXWAIST": 92 + 16 * trend + (6 if sex == 1 else 0) + random.normal(0, 3.5),
                "BPX_SYSTOLIC": 118 + 14 * trend + random.normal(0, 7.0),
                "GHB_LBXGH": 5.2 + 1.5 * trend + random.normal(0, 0.25),
                "GLU_LBXGLU": 92 + 34 * trend + random.normal(0, 8.0),
                "INS_LBXIN": 8 + 16 * trend + random.normal(0, 3.0),
                "TCHOL_LBXTC": 185 + 22 * trend + random.normal(0, 15.0),
                "HDL_LBDHDD": 56 - 12 * trend + random.normal(0, 6.0),
                "TRIGLY_LBXTR": 118 + 70 * trend + random.normal(0, 25.0),
            }
            for feature, value in values.items():
                probability = config.measurement_probability.get(feature, 0.6)
                measured = feature == "DEMO_RIDAGEYR" or random.random() < probability
                event_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "patient_id": patient_id,
                        "observation_timestamp": visit_date,
                        "source": "metaboguard_simulator",
                        "feature_code": feature,
                        "value": round(float(max(value, 0.1)), 3) if measured else None,
                        "unit": {"DEMO_RIDAGEYR": "years"}.get(feature),
                        "missingness_reason": "observed" if measured else "not_measured_at_visit",
                        "visit_id": f"{patient_id}:V{visit_index}",
                        "index_date": index_date,
                        "outcome_type": "none",
                        "event_date": None,
                        "cancer_site": None,
                        "cancer_stage": None,
                        "censoring_date": None,
                        "provenance": provenance,
                    }
                )

        # Event generation after the index date, in yearly steps, with competing death.
        follow_up_days = float(config.max_follow_up_years * 365.25)
        hazards = {
            "type2_diabetes": config.baseline_yearly_hazard["type2_diabetes"]
            * stratum["diabetes_multiplier"]
            * (1.0 + 1.8 * burden),
            "pan_cancer": config.baseline_yearly_hazard["pan_cancer"]
            * stratum["cancer_multiplier"]
            * (1.0 + 0.9 * burden)
            * (1.0 + max(age_at_index - 50, 0) / 40.0),
            "death": config.baseline_yearly_hazard["death"]
            * (1.0 + max(age_at_index - 50, 0) / 25.0),
        }
        outcome_type = "censored"
        event_day = follow_up_days
        site = None
        step_days = 30.0
        day = 0.0
        while day < follow_up_days:
            day += step_days
            draws = {
                name: 1.0 - np.exp(-hazard * (step_days / 365.25))
                for name, hazard in hazards.items()
            }
            uniforms = {name: float(random.random()) for name in draws}
            fired = [name for name, probability in draws.items() if uniforms[name] < probability]
            if fired:
                # Earliest-firing wins; death is a competing event, not an outcome.
                outcome_type = sorted(fired, key=lambda name: uniforms[name] / max(draws[name], 1e-9))[0]
                event_day = day
                if outcome_type == "pan_cancer":
                    sites = list(config.site_mix)
                    probabilities = np.array([config.site_mix[name] for name in sites])
                    probabilities = probabilities / probabilities.sum()
                    site = sites[int(random.choice(len(sites), p=probabilities))]
                break

        # Administrative censoring: uniform loss to follow-up before the study end.
        administrative_censor_day = float(
            min(follow_up_days, random.uniform(0.35 * follow_up_days, follow_up_days))
        )
        if outcome_type == "censored" or event_day > administrative_censor_day:
            censoring_date = index_date + timedelta(days=administrative_censor_day)
            event_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "patient_id": patient_id,
                    "observation_timestamp": censoring_date,
                    "source": "metaboguard_simulator",
                    "feature_code": "outcome:censored",
                    "value": 1.0,
                    "unit": None,
                    "missingness_reason": "observed",
                    "visit_id": None,
                    "index_date": index_date,
                    "outcome_type": "censored",
                    "event_date": censoring_date,
                    "cancer_site": None,
                    "cancer_stage": None,
                    "censoring_date": censoring_date,
                    "provenance": provenance,
                }
            )
            realised_outcome = "censored"
        else:
            event_date = index_date + timedelta(days=event_day)
            event_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "patient_id": patient_id,
                    "observation_timestamp": event_date,
                    "source": "metaboguard_simulator",
                    "feature_code": f"outcome:{outcome_type}",
                    "value": 1.0,
                    "unit": None,
                    "missingness_reason": "observed",
                    "visit_id": None,
                    "index_date": index_date,
                    "outcome_type": outcome_type,
                    "event_date": event_date,
                    "cancer_site": site,
                    "cancer_stage": None,
                    "censoring_date": index_date + timedelta(days=administrative_censor_day),
                    "provenance": provenance,
                }
            )
            realised_outcome = outcome_type

        strata_rows.append(
            {
                "patient_id": patient_id,
                "sampling_stratum": stratum_name,
                "sampling_weight": stratum["weight"],
                "diabetes_multiplier": stratum["diabetes_multiplier"],
                "cancer_multiplier": stratum["cancer_multiplier"],
                "latent_burden": round(burden, 4),
                "sex_code": sex,
                "age_at_index": round(age_at_index, 2),
                "index_date": index_date,
                "realised_outcome": realised_outcome,
                "cancer_site": site,
            }
        )

    events = pd.DataFrame(event_rows)
    strata = pd.DataFrame(strata_rows)
    metadata = {
        "generator": "metaboguard_simulator",
        "generator_version": "v1",
        "why_not_synthea": synthea_availability()["reasons_unavailable"],
        "config": config.as_dict(),
        "enrichment_declared": True,
        "calibration_warning": (
            "Event rates were deliberately enriched through the declared sampling strata. "
            "Raw predicted probabilities are therefore NOT population-calibrated; only "
            "within-sample discrimination and re-calibrated outputs are meaningful, and "
            "neither transfers to real patients."
        ),
        "features": list(PREVENTION_SAFE_FEATURES),
    }
    return events, strata, metadata


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    generator: str = "auto",
    patients: int = 4000,
    seed: int = 20260805,
    synthea_csv_dir: str | Path | None = None,
    fixture_patients: int = 60,
) -> dict[str, Any]:
    """Generate, validate and persist a simulation-only longitudinal dataset."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    availability = synthea_availability()

    used_generator = generator
    strata = pd.DataFrame()
    if generator in {"auto", "synthea"}:
        if synthea_csv_dir:
            events = convert_synthea_csv(synthea_csv_dir)
            metadata = {
                "generator": "synthea",
                "generator_version": SYNTHEA_CONFIG["pinned_release"],
                "config": SYNTHEA_CONFIG,
                "csv_dir": str(synthea_csv_dir),
                "enrichment_declared": False,
                "calibration_warning": (
                    "Synthea incidence follows its own modules, not a real population. "
                    "Predicted probabilities are not calibrated to any real cohort."
                ),
            }
            used_generator = "synthea"
        elif availability["synthea_available"] and generator == "synthea":
            run_info = run_synthea(output / "synthea_raw")
            events = convert_synthea_csv(Path(run_info["output_dir"]) / "csv")
            metadata = {"generator": "synthea", "run": run_info, "config": SYNTHEA_CONFIG}
            used_generator = "synthea"
        elif generator == "synthea":
            raise RuntimeError(
                "Synthea requested but unavailable: "
                + " ".join(availability["reasons_unavailable"])
            )
        else:
            events, strata, metadata = simulate_longitudinal_cohort(
                SimulatorConfig(patients=patients, seed=seed)
            )
            used_generator = "metaboguard_simulator"
    else:
        events, strata, metadata = simulate_longitudinal_cohort(
            SimulatorConfig(patients=patients, seed=seed)
        )
        used_generator = "metaboguard_simulator"

    validated, report = validate_event_frame(events, dataset_name=f"{used_generator}_events", strict=True)

    events_path = output / "patient_events.csv"
    validated.to_csv(events_path, index=False)
    if not strata.empty:
        strata.to_csv(output / "patient_strata.csv", index=False)
    (output / "event_validation_report.json").write_text(json.dumps(report.as_dict(), indent=2))

    # Small committed fixture for tests: first N patients, deterministic.
    fixture_ids = sorted(validated["patient_id"].unique())[:fixture_patients]
    fixture = validated[validated["patient_id"].isin(fixture_ids)]
    fixture_path = output / "fixture_patient_events.csv"
    fixture.to_csv(fixture_path, index=False)

    manifest = DatasetManifest(
        dataset_name=f"synthetic_longitudinal_{used_generator}",
        created_at=datetime.now(UTC),
        capability_state=CapabilityState.SIMULATION_ONLY_LONGITUDINAL,
        simulation_only=True,
        generator={
            **metadata,
            "synthea_availability": availability,
            "requested_generator": generator,
            "used_generator": used_generator,
        },
        row_counts={
            "event_rows": int(len(validated)),
            "patients": int(validated["patient_id"].nunique()),
            "fixture_rows": int(len(fixture)),
            "fixture_patients": int(len(fixture_ids)),
        },
        fingerprints={
            "events_frame": frame_fingerprint(validated.drop(columns=["provenance"])),
            "events_file": "",
            "fixture_frame": frame_fingerprint(fixture.drop(columns=["provenance"])),
        },
        notes=[
            "SIMULATION ONLY. Not real patients. Cannot establish real-world calibration, "
            "clinical utility or early detection.",
            f"Synthea preferred source: {SYNTHEA_CONFIG['repository']} ({SYNTHEA_CONFIG['licence']}), "
            f"methodology DOI {SYNTHEA_CONFIG['methodology_doi']}, validity limits DOI "
            f"{SYNTHEA_CONFIG['validity_doi']}.",
            metadata.get("calibration_warning", ""),
        ],
    )
    write_manifest(manifest, output / "dataset_manifest.json")

    return {
        "output_dir": str(output),
        "generator": used_generator,
        "events_path": str(events_path),
        "fixture_path": str(fixture_path),
        "rows": int(len(validated)),
        "patients": int(validated["patient_id"].nunique()),
        "validation_status": report.as_dict()["status"],
        "outcome_counts": report.outcomes["counts"],
        "capability_state": CapabilityState.SIMULATION_ONLY_LONGITUDINAL.value,
        "synthea_available": availability["synthea_available"],
        "synthea_reasons": availability["reasons_unavailable"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--generator", choices=["auto", "synthea", "simulator"], default="auto")
    parser.add_argument("--patients", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--synthea-csv-dir", default=None)
    parser.add_argument("--print-synthea-status", action="store_true")
    arguments = parser.parse_args()

    if arguments.print_synthea_status:
        print(json.dumps(synthea_availability(), indent=2, default=str))
        return
    result = generate_dataset(
        arguments.output_dir,
        generator=arguments.generator,
        patients=arguments.patients,
        seed=arguments.seed,
        synthea_csv_dir=arguments.synthea_csv_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()