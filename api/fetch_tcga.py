"""
TCGA-CDR Fetch and Harmonization
================================
Downloads the TCGA Pan-Cancer Clinical Data Resource (Liu et al., 2018, Cell)
supplemental table S1 from the NCI GDC and reshapes it into a CSV that is
compatible with the existing DiaPan pipeline (biomarker.py, predictive.py,
engine.py) by projecting TCGA fields onto NHANES-shaped columns.

Target definition
-----------------
`Cancer` in DiaPan is repurposed for the TCGA cohort as "died within 5 years of
initial pathologic diagnosis" (5-year overall mortality). This is the standard
5-year overall survival endpoint flipped so 1 = event, 0 = censored-alive.

Rows with insufficient follow-up (alive but observed < 5 years) are dropped so
the label is well-defined.

Feature mapping (TCGA -> NHANES-shaped column)
----------------------------------------------
- bcr_patient_barcode              -> tcga_patient_barcode
- age_at_initial_pathologic_diagnosis -> DEMO_RIDAGEYR
- gender (MALE/FEMALE)             -> DEMO_RIAGENDR (1 male, 2 female to match NHANES)
- race                             -> DEMO_RIDRETH3 (integer encoded)
- type (cancer type code)          -> tcga_cancer_type (kept as label, plus one-hot flags)
- ajcc_pathologic_tumor_stage      -> tcga_stage_ordinal (0..4)
- histological_grade               -> tcga_grade_ordinal (1..4)
- tumor_status                     -> tcga_tumor_status (1 = with tumor, 0 = tumor free)
- treatment_outcome_first_course   -> tcga_treatment_response (0..4 ordinal, worse->better)
- OS.time / OS                     -> tcga_followup_days / tcga_event (METADATA, not features)
- PFI.time / PFI                   -> tcga_pfi_days / tcga_pfi_event (METADATA)
- Cancer (derived)                 -> 5-year all-cause mortality label
- Progression (derived)            -> 5-year progression-free-interval event label
- Diabetes                         -> NaN (unavailable in TCGA-CDR)
- Obesity                          -> NaN (BMI not in TCGA-CDR)

The resulting CSV is written to both:
- <project_root>/data/tcga_cdr.csv
- <api_dir>/nhanes_data/tcga_cdr.csv

so the existing dataset discovery in main.py picks it up automatically.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


TCGA_CDR_URL = "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
TCGA_CDR_FILENAME = "TCGA-CDR-SupplementalTableS1.xlsx"
TCGA_CDR_SHEET = "TCGA-CDR"

FIVE_YEARS_DAYS = 365 * 5

# Race string -> NHANES-style integer (approximation, kept internally consistent)
RACE_MAP = {
    "WHITE": 3,
    "BLACK OR AFRICAN AMERICAN": 4,
    "ASIAN": 6,
    "AMERICAN INDIAN OR ALASKA NATIVE": 7,
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": 7,
    "[NOT AVAILABLE]": np.nan,
    "[UNKNOWN]": np.nan,
    "[NOT EVALUATED]": np.nan,
}

GENDER_MAP = {"MALE": 1, "FEMALE": 2}

# Histological grade -> ordinal. G1 low ... G4 high. "High Grade" collapses to G3
# and "Low Grade" collapses to G1 as pragmatic mid-points.
GRADE_MAP = {
    "G1": 1, "G2": 2, "G3": 3, "G4": 4,
    "LOW GRADE": 1, "HIGH GRADE": 3,
    "GX": np.nan, "GB": np.nan,
    "[NOT AVAILABLE]": np.nan, "[NOT APPLICABLE]": np.nan, "[UNKNOWN]": np.nan,
    "[DISCREPANCY]": np.nan,
}

# Tumor status: 1 = still with tumor (worse prognosis), 0 = tumor free
TUMOR_STATUS_MAP = {
    "WITH TUMOR": 1,
    "TUMOR FREE": 0,
    "[NOT AVAILABLE]": np.nan,
    "[UNKNOWN]": np.nan,
    "[DISCREPANCY]": np.nan,
}

# Treatment response after first course, ordinal: 0 progressive -> 4 complete remission.
TREATMENT_RESPONSE_MAP = {
    "PROGRESSIVE DISEASE": 0,
    "STABLE DISEASE": 1,
    "PARTIAL REMISSION/RESPONSE": 2,
    "COMPLETE REMISSION/RESPONSE": 3,
    "[NOT AVAILABLE]": np.nan,
    "[NOT APPLICABLE]": np.nan,
    "[UNKNOWN]": np.nan,
    "[DISCREPANCY]": np.nan,
}

# Ordinal encoding for AJCC stage. Non-numeric / unknown -> NaN.
STAGE_MAP = {
    "STAGE 0": 0,
    "STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1, "STAGE IC": 1,
    "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2, "STAGE IIC": 2,
    "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
    "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4, "STAGE IVC": 4,
    "STAGE X": np.nan, "[NOT AVAILABLE]": np.nan, "[NOT APPLICABLE]": np.nan,
    "[UNKNOWN]": np.nan, "[DISCREPANCY]": np.nan, "I/II NOS": 1,
    "IS": 0,
}


def _download_xlsx(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urlopen(request) as response:
        return response.read()


def _load_cdr_frame(cache_path: Path | None = None) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        payload = cache_path.read_bytes()
    else:
        payload = _download_xlsx(TCGA_CDR_URL)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
    return pd.read_excel(BytesIO(payload), sheet_name=TCGA_CDR_SHEET)


def _encode_stage(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    key = value.strip().upper()
    if key in STAGE_MAP:
        return STAGE_MAP[key]
    return np.nan


def _encode_race(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    key = value.strip().upper()
    return RACE_MAP.get(key, np.nan)


def _encode_gender(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    return GENDER_MAP.get(value.strip().upper(), np.nan)


def _encode_by_map(value: object, mapping: dict) -> float:
    if not isinstance(value, str):
        return np.nan
    return mapping.get(value.strip().upper(), np.nan)


def _five_year_event(event: object, time_days: object) -> float:
    """
    Generic 5-year event labeler for right-censored survival data.

    1   -> event observed within 5 years of diagnosis
    0   -> event-free with follow-up >= 5 years (censored beyond horizon)
    NaN -> event-free but censored before 5 years (indeterminate)
    """
    if pd.isna(event) or pd.isna(time_days):
        return np.nan

    try:
        time_days = float(time_days)
        event = int(event)
    except (TypeError, ValueError):
        return np.nan

    if event == 1:
        return 1.0 if time_days <= FIVE_YEARS_DAYS else 0.0
    if time_days >= FIVE_YEARS_DAYS:
        return 0.0
    return np.nan


def build_tcga_cdr_frame(source_frame: pd.DataFrame) -> pd.DataFrame:
    df = source_frame.copy()

    # Targets (5-year horizons)
    df["Cancer"] = [_five_year_event(e, t) for e, t in zip(df["OS"], df["OS.time"])]
    df["Progression"] = [_five_year_event(e, t) for e, t in zip(df["PFI"], df["PFI.time"])]

    # Demographic features
    df["DEMO_RIDAGEYR"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce")
    df["DEMO_RIAGENDR"] = df["gender"].map(_encode_gender)
    df["DEMO_RIDRETH3"] = df["race"].map(_encode_race)

    # Tumor / disease features
    df["tcga_stage_ordinal"] = df["ajcc_pathologic_tumor_stage"].map(_encode_stage)
    df["tcga_grade_ordinal"] = df["histological_grade"].map(lambda v: _encode_by_map(v, GRADE_MAP))
    df["tcga_tumor_status"] = df["tumor_status"].map(lambda v: _encode_by_map(v, TUMOR_STATUS_MAP))
    df["tcga_treatment_response"] = df["treatment_outcome_first_course"].map(
        lambda v: _encode_by_map(v, TREATMENT_RESPONSE_MAP)
    )

    # Survival metadata (NEVER used as features -- leaks the labels)
    df["tcga_followup_days"] = pd.to_numeric(df["OS.time"], errors="coerce")
    df["tcga_event"] = pd.to_numeric(df["OS"], errors="coerce")
    df["tcga_pfi_days"] = pd.to_numeric(df["PFI.time"], errors="coerce")
    df["tcga_pfi_event"] = pd.to_numeric(df["PFI"], errors="coerce")

    df["tcga_cancer_type"] = df["type"].astype(str)

    # Numeric SEQN for compatibility with existing biomarker case memory
    df["SEQN"] = np.arange(1, len(df) + 1, dtype=int)
    df["tcga_patient_barcode"] = df["bcr_patient_barcode"].astype(str)

    # NHANES-shaped columns DiaPan expects but TCGA does not carry
    df["Diabetes"] = np.nan
    df["Obesity"] = np.nan
    df["BMX_BMXBMI"] = np.nan
    df["BMX_BMXWAIST"] = np.nan

    # One-hot flags for the 33 cancer types
    type_dummies = pd.get_dummies(df["tcga_cancer_type"], prefix="tcga_type", dtype=int)
    df = pd.concat([df, type_dummies], axis=1)

    keep_cols = [
        "SEQN",
        "tcga_patient_barcode",
        "tcga_cancer_type",
        "Cancer",
        "Progression",
        "Diabetes",
        "Obesity",
        "DEMO_RIDAGEYR",
        "DEMO_RIAGENDR",
        "DEMO_RIDRETH3",
        "BMX_BMXBMI",
        "BMX_BMXWAIST",
        "tcga_stage_ordinal",
        "tcga_grade_ordinal",
        "tcga_tumor_status",
        "tcga_treatment_response",
        "tcga_followup_days",
        "tcga_event",
        "tcga_pfi_days",
        "tcga_pfi_event",
        *type_dummies.columns.tolist(),
    ]
    result = df[keep_cols].copy()
    return result


def ensure_tcga_cdr_dataset(force: bool = False) -> Path:
    api_dir = Path(__file__).resolve().parent
    project_root = api_dir.parent

    output_paths = [
        project_root / "data" / "tcga_cdr.csv",
        api_dir / "nhanes_data" / "tcga_cdr.csv",
    ]
    cache_xlsx = api_dir / "nhanes_data" / TCGA_CDR_FILENAME

    if not force and all(path.exists() for path in output_paths):
        return output_paths[0]

    print(f"Downloading TCGA-CDR from {TCGA_CDR_URL} ...")
    source = _load_cdr_frame(cache_path=cache_xlsx)
    print(f"Loaded TCGA-CDR: {len(source):,} rows x {source.shape[1]} columns")

    built = build_tcga_cdr_frame(source)
    # Keep any row that has at least one resolvable target
    labelled = built.dropna(subset=["Cancer", "Progression"], how="all")
    print(
        f"After 5-year target labelling: {len(labelled):,} rows "
        f"(dropped {len(built) - len(labelled):,} indeterminate rows)"
    )
    cancer_labelled = labelled.dropna(subset=["Cancer"])
    prog_labelled = labelled.dropna(subset=["Progression"])
    print(
        "Cancer (5y all-cause mortality): "
        f"{int(cancer_labelled['Cancer'].sum()):,} positives / "
        f"{int((cancer_labelled['Cancer'] == 0).sum()):,} negatives "
        f"({len(cancer_labelled):,} total)"
    )
    print(
        "Progression (5y PFI event): "
        f"{int(prog_labelled['Progression'].sum()):,} positives / "
        f"{int((prog_labelled['Progression'] == 0).sum()):,} negatives "
        f"({len(prog_labelled):,} total)"
    )

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labelled.to_csv(output_path, index=False)
        print(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)")

    return output_paths[0]


if __name__ == "__main__":
    ensure_tcga_cdr_dataset(force=False)
