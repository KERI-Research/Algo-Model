"""
Build a harmonised NHANES 1999-March 2020 research cohort for DiaPan.

Design decisions
----------------
1. Use non-overlapping survey periods: 1999-2016 biennial cycles plus the
   combined 2017-March 2020 pre-pandemic release. Do not also include the
   standalone 2017-2018 J cycle because those participants are contained in P.
2. Preserve the existing DiaPan column contract (for example GHB_LBXGH) while
   adding survey-cycle provenance and derived features.
3. Treat repeated NHANES cycles as repeated cross-sections, not longitudinal
   patient follow-up. "Trajectory" features below are cohort/age/cycle proxies.
4. Keep source-specific failures visible in a JSON build report instead of
   silently dropping whole cycles.

Outputs
-------
- data/nhanes_multicycle.csv
- api/nhanes_data/nhanes_multicycle.csv
- data/nhanes_multicycle_build_report.json
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start}/DataFiles/{filename}"
PANCREAS_MCQ_CODE = 39


@dataclass(frozen=True)
class Cycle:
    label: str
    start: int
    suffix: str
    midpoint: float
    index: int
    prefix_style: bool = False

    def filename(self, component: str) -> str:
        if self.prefix_style:
            return f"P_{component}.XPT"
        if not self.suffix:
            return f"{component}.XPT"
        return f"{component}_{self.suffix}.XPT"


# P is the combined, non-overlapping 2017-March 2020 pre-pandemic release.
CYCLES = [
    Cycle("1999-2000", 1999, "", 1999.5, 0),
    Cycle("2001-2002", 2001, "B", 2001.5, 1),
    Cycle("2003-2004", 2003, "C", 2003.5, 2),
    Cycle("2005-2006", 2005, "D", 2005.5, 3),
    Cycle("2007-2008", 2007, "E", 2007.5, 4),
    Cycle("2009-2010", 2009, "F", 2009.5, 5),
    Cycle("2011-2012", 2011, "G", 2011.5, 6),
    Cycle("2013-2014", 2013, "H", 2013.5, 7),
    Cycle("2015-2016", 2015, "I", 2015.5, 8),
    Cycle("2017-March2020", 2017, "P", 2018.5, 9, prefix_style=True),
]


# Required components define labels/demographics. Optional lab files vary more
# across cycles and are recorded as missing rather than aborting the build.
COMPONENTS = {
    "DEMO": {"required": True},
    "BMX": {"required": False},
    "DIQ": {"required": True},
    "MCQ": {"required": True},
    "GHB": {"required": False},
    "GLU": {"required": False},
    "INS": {"required": False},
    "CPEP": {"required": False},
    "TRIGLY": {"required": False},
    "HDL": {"required": False},
    "TCHOL": {"required": False},
    "HSCRP": {"required": False},
    "WHQ": {"required": False},
}


# Only retain variables used by the brief or survey design. Missing columns are
# added as NaN after the per-cycle merge.
KEEP = {
    "DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3",
             "WTMEC2YR", "WTMEC4YR", "WTMECPRP", "WTINT2YR",
             "SDMVPSU", "SDMVSTRA"],
    "BMX": ["SEQN", "BMXWT", "BMXBMI", "BMXWAIST"],
    "DIQ": ["SEQN", "DIQ010", "DID040", "DIQ050", "DIQ160", "DIQ170",
            "DIQ172", "DIQ180"],
    "MCQ": ["SEQN", "MCQ220", "MCQ230A", "MCQ230B", "MCQ230C", "MCQ230D"],
    "GHB": ["SEQN", "LBXGH"],
    "GLU": ["SEQN", "LBXGLU", "LBXIN", "LBDINSI"],
    "INS": ["SEQN", "LBXIN", "LBDINSI"],
    "CPEP": ["SEQN", "LBXCP", "LBXCPSI"],
    "TRIGLY": ["SEQN", "LBXTR", "LBDLDL"],
    "HDL": ["SEQN", "LBDHDD"],
    "TCHOL": ["SEQN", "LBXTC"],
    "HSCRP": ["SEQN", "LBXHSCRP"],
    "WHQ": ["SEQN", "WHD020", "WHD050", "WHD140"],
}


def _candidate_filenames(cycle: Cycle, component: str) -> list[str]:
    """Return current and cycle-specific legacy filename candidates."""
    # 1999-2004 bundled analytes in numbered laboratory files.
    if cycle.start == 1999:
        aliases = {
            "GHB": ["LAB10.XPT"],
            "GLU": ["LAB10AM.XPT"],
            "INS": ["LAB10AM.XPT"],
            "CPEP": ["LAB10AM.XPT"],
            "TRIGLY": ["LAB13.XPT"],
            "HDL": ["LAB13.XPT"],
            "TCHOL": ["LAB13.XPT"],
        }
        if component in aliases:
            return aliases[component]
    if cycle.start in (2001, 2003):
        suffix = "B" if cycle.start == 2001 else "C"
        aliases = {
            "GHB": [f"L10_{suffix}.XPT"],
            "GLU": [f"L10AM_{suffix}.XPT"],
            "INS": [f"L10AM_{suffix}.XPT"],
            "CPEP": [f"L10AM_{suffix}.XPT"],
            "TRIGLY": [f"L13_{suffix}.XPT"],
            "HDL": [f"L13_{suffix}.XPT"],
            "TCHOL": [f"L13_{suffix}.XPT"],
        }
        if component in aliases:
            return aliases[component]
    return [cycle.filename(component)]


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def _read_xpt(payload: bytes) -> pd.DataFrame:
    frame = pd.read_sas(BytesIO(payload), format="xport")
    frame.columns = [c.decode("utf-8") if isinstance(c, bytes) else c for c in frame.columns]
    if "SEQN" not in frame.columns:
        raise ValueError("XPT file has no SEQN")
    frame["SEQN"] = pd.to_numeric(frame["SEQN"], errors="coerce")
    return frame.dropna(subset=["SEQN"]).assign(SEQN=lambda x: x["SEQN"].astype("int64"))


def _component_frame(cycle: Cycle, component: str) -> tuple[pd.DataFrame | None, dict]:
    candidates = _candidate_filenames(cycle, component)
    record = {"cycle": cycle.label, "component": component, "url": None,
              "filename": None, "candidate_filenames": candidates,
              "status": "missing", "rows": 0}
    errors: list[str] = []
    raw = None
    for filename in candidates:
        url = BASE_URL.format(start=cycle.start, filename=filename)
        try:
            payload = _download(url)
            raw = _read_xpt(payload)
            record.update(url=url, filename=filename)
            break
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            errors.append(f"{filename}: {type(error).__name__}: {error}")
    if raw is None:
        record["error"] = " | ".join(errors)
        return None, record

    wanted = [c for c in KEEP[component] if c in raw.columns]
    frame = raw[wanted].copy()
    frame = frame.rename(columns={c: f"{component}_{c}" for c in frame.columns if c != "SEQN"})
    record.update(status="loaded", rows=len(frame), columns=list(frame.columns))
    return frame, record


def _binary(series: pd.Series) -> pd.Series:
    value = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(value == 1, 1.0, np.where(value == 2, 0.0, np.nan)),
                     index=series.index)


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # In 2005-2012 insulin was bundled in the GLU file. Normalize back to the
    # existing DiaPan name so one model feature works across all cycles.
    if "INS_LBXIN" not in out.columns and "GLU_LBXIN" in out.columns:
        out["INS_LBXIN"] = out["GLU_LBXIN"]
    elif "GLU_LBXIN" in out.columns:
        out["INS_LBXIN"] = out["INS_LBXIN"].fillna(out["GLU_LBXIN"])
    diabetes_raw = _safe_numeric(out, "DIQ_DIQ010")
    cancer_raw = _safe_numeric(out, "MCQ_MCQ220")
    out["Diabetes"] = _binary(diabetes_raw)
    out["Cancer"] = _binary(cancer_raw)

    cancer_slots = [_safe_numeric(out, f"MCQ_MCQ230{x}") for x in "ABCD"]
    slot_matrix = pd.concat(cancer_slots, axis=1)
    any_pancreas = (slot_matrix == PANCREAS_MCQ_CODE).any(axis=1)
    out["PancreaticCancer"] = np.where(
        any_pancreas, 1.0, np.where(cancer_raw.isin([1, 2]), 0.0, np.nan)
    )

    bmi = _safe_numeric(out, "BMX_BMXBMI")
    out["Obesity"] = np.where(bmi.notna(), (bmi >= 30).astype(float), np.nan)

    age = _safe_numeric(out, "DEMO_RIDAGEYR")
    onset = _safe_numeric(out, "DIQ_DID040")
    insulin_use = _safe_numeric(out, "DIQ_DIQ050")
    duration = (age - onset).where(lambda x: (x >= 0) & (x <= 90))
    out["diabetes_duration_years"] = duration
    out["recent_diabetes_onset"] = np.where(
        duration.notna(), (duration <= 3).astype(float), np.nan
    )
    out["new_onset_diabetes"] = out["recent_diabetes_onset"]
    out["diabetes_subtype"] = np.where(
        diabetes_raw == 2, 0.0,
        np.where(
            diabetes_raw == 1,
            np.where((onset < 25) & (insulin_use == 1), 1.0, 2.0),
            np.nan,
        ),
    )

    glucose = _safe_numeric(out, "GLU_LBXGLU")
    insulin = _safe_numeric(out, "INS_LBXIN")
    hba1c = _safe_numeric(out, "GHB_LBXGH")
    out["homa_ir"] = (glucose * insulin) / 405.0
    out["elevated_hba1c"] = np.where(hba1c.notna(), (hba1c >= 6.5).astype(float), np.nan)
    out["fasting_hyperglycemia"] = np.where(
        glucose.notna(), (glucose >= 126).astype(float), np.nan
    )

    current_weight = _safe_numeric(out, "WHQ_WHD020").where(lambda x: x < 700)
    prior_weight = _safe_numeric(out, "WHQ_WHD050").where(lambda x: x < 700)
    decade_weight = _safe_numeric(out, "WHQ_WHD140").where(lambda x: x < 700)
    out["weight_loss_1yr_lb"] = prior_weight - current_weight
    out["significant_weight_loss_flag"] = np.where(
        out["weight_loss_1yr_lb"].notna(),
        (out["weight_loss_1yr_lb"] >= 10).astype(float),
        np.nan,
    )
    out["weight_loss_10yr_lb"] = decade_weight - current_weight

    waist = _safe_numeric(out, "BMX_BMXWAIST")
    out["age_bmi_interaction"] = age * bmi
    out["waist_bmi_interaction"] = waist * bmi
    out["hba1c_age_interaction"] = hba1c * age
    out["hba1c_diabetes_duration_interaction"] = hba1c * duration
    out["hba1c_weight_loss_interaction"] = hba1c * out["weight_loss_1yr_lb"]

    out["age_band"] = pd.cut(
        age, bins=[0, 29, 39, 49, 59, 69, 79, np.inf],
        labels=["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    ).astype("string")

    # Cohort-relative HbA1c z-score. This is a repeated-cross-sectional
    # trajectory proxy, not a within-patient slope.
    group_keys = ["survey_cycle", "age_band", "DEMO_RIAGENDR"]
    valid_keys = [k for k in group_keys if k in out.columns]
    if valid_keys:
        means = out.groupby(valid_keys, observed=True)["GHB_LBXGH"].transform("mean")
        stds = out.groupby(valid_keys, observed=True)["GHB_LBXGH"].transform("std")
        out["hba1c_cycle_age_sex_z"] = (hba1c - means) / stds.replace(0, np.nan)
    else:
        out["hba1c_cycle_age_sex_z"] = np.nan

    # Combined MEC examination weight for descriptive population estimates.
    # Total represented duration = 18 years (1999-2016) + 3.2 years
    # (2017-March 2020 pre-pandemic) = 21.2 years.
    #
    # 1999-2002 uses the NCHS-provided 4-year bridge weight because the first
    # cycle used a different Census population base. Later biennial cycles use
    # 2-year weights; the pre-pandemic release uses WTMECPRP and a 3.2-year
    # contribution. This weight is NOT a fasting-subsample weight and therefore
    # must not be used for population estimates restricted to glucose/insulin.
    total_years = 21.2
    mec2 = _safe_numeric(out, "DEMO_WTMEC2YR")
    mec4 = _safe_numeric(out, "DEMO_WTMEC4YR")
    mec_prp = _safe_numeric(out, "DEMO_WTMECPRP")
    out["combined_mec_weight_1999_2020"] = np.select(
        [
            out["survey_cycle"].isin(["1999-2000", "2001-2002"]),
            out["survey_cycle"] == "2017-March2020",
        ],
        [
            mec4 * (4.0 / total_years),
            mec_prp * (3.2 / total_years),
        ],
        default=mec2 * (2.0 / total_years),
    )
    return out


def build_multicycle(output_root: Path | None = None) -> tuple[Path, Path, Path]:
    api_dir = Path(__file__).resolve().parent
    project_root = output_root or api_dir.parent
    reports: list[dict] = []
    cycle_frames: list[pd.DataFrame] = []

    for cycle in CYCLES:
        merged: pd.DataFrame | None = None
        cycle_records: list[dict] = []
        for component, config in COMPONENTS.items():
            frame, record = _component_frame(cycle, component)
            reports.append(record)
            cycle_records.append(record)
            if frame is None:
                if config["required"]:
                    raise RuntimeError(
                        f"Required NHANES component unavailable: {cycle.label} {component}: "
                        f"{record.get('error')}"
                    )
                continue
            merged = frame if merged is None else merged.merge(frame, on="SEQN", how="outer")

        if merged is None:
            continue
        merged["survey_cycle"] = cycle.label
        merged["survey_year_midpoint"] = cycle.midpoint
        merged["survey_cycle_index"] = cycle.index
        cycle_frames.append(merged)
        print(f"{cycle.label}: {len(merged):,} rows; "
              f"{sum(r['status'] == 'loaded' for r in cycle_records)}/{len(cycle_records)} files")

    pooled = pd.concat(cycle_frames, ignore_index=True, sort=False)
    pooled = _derive(pooled)
    pooled["global_participant_id"] = pooled["survey_cycle"].astype(str) + ":" + pooled["SEQN"].astype(str)

    data_path = project_root / "data" / "nhanes_multicycle.csv"
    api_path = project_root / "api" / "nhanes_data" / "nhanes_multicycle.csv"
    report_path = project_root / "data" / "nhanes_multicycle_build_report.json"
    for path in (data_path, api_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(data_path, index=False)
    pooled.to_csv(api_path, index=False)

    target_summary = {}
    for target in ("Diabetes", "Cancer", "PancreaticCancer"):
        values = pd.to_numeric(pooled[target], errors="coerce")
        target_summary[target] = {
            "labelled": int(values.notna().sum()),
            "positive": int((values == 1).sum()),
            "negative": int((values == 0).sum()),
        }
    build_report = {
        "design": "Repeated cross-sectional pooled NHANES; not longitudinal follow-up",
        "cycles": [cycle.label for cycle in CYCLES],
        "rows": len(pooled),
        "columns": len(pooled.columns),
        "target_summary": target_summary,
        "weighting": {
            "combined_weight_column": "combined_mec_weight_1999_2020",
            "total_represented_years": 21.2,
            "formula": {
                "1999-2002": "WTMEC4YR * (4 / 21.2)",
                "2003-2016": "WTMEC2YR * (2 / 21.2)",
                "2017-March2020": "WTMECPRP * (3.2 / 21.2)",
            },
            "warning": (
                "MEC combined weight is for descriptive examined-sample estimates. "
                "It is not a fasting-subsample weight and is not currently used in "
                "the predictive train/test pipeline."
            ),
        },
        "source_files": reports,
    }
    report_path.write_text(json.dumps(build_report, indent=2))
    print(json.dumps({k: v for k, v in build_report.items() if k != "source_files"}, indent=2))
    return data_path, api_path, report_path


if __name__ == "__main__":
    build_multicycle()
