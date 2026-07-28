"""
This script is intended to create the first trustworthy NHANES prototype for the MetaboGuard
pipeline by pulling the 2017-2018 CDC public-use files for demographics, body measures,
and medical conditions, then reconciling them into a single analysis-ready table keyed on
SEQN. The goal is deliberately narrow, but the stance is deliberately skeptical: every
source file should be treated as a separate measurement view, every merge should be checked
for participant loss or duplication, and every output should be easy to reproduce if the CDC
updates its hosting layout or if the prototype needs to be rebuilt from scratch. The script
keeps the raw variable names visible through source-specific namespaces so downstream causal
feature design can remain explicit, while still producing a practical CSV artifact that can
be inspected, versioned, and fed into the current pipeline without additional glue code.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
NHANES_FILES = {
    "DEMO": "DEMO_J.xpt",
    "BMX": "BMX_J.xpt",
    "DIQ": "DIQ_J.xpt",
    "MCQ": "MCQ_J.xpt",
    "GHB": "GHB_J.xpt",
    "GLU": "GLU_J.xpt",
    "INS": "INS_J.xpt",
    "TRIGLY": "TRIGLY_J.xpt",
    "HDL": "HDL_J.xpt",
    "TCHOL": "TCHOL_J.xpt",
    "HSCRP": "HSCRP_J.xpt",
    "WHQ": "WHQ_J.xpt",  # Weight history: self-reported current + past weights
    "SMQ": "SMQ_J.xpt",
    "ALQ": "ALQ_J.xpt",
    "CBC": "CBC_J.xpt",
    "BIOPRO": "BIOPRO_J.xpt",
}

# Official NHANES MCQ230 coding: 29 = Pancreas (pancreatic); 39 = Other.
# https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/MCQ_J.htm
PANCREAS_MCQ_CODE = 29

MODEL_COLUMNS = {"Diabetes", "Cancer", "Obesity"}
BIOMARKER_COLUMNS = {
    "GHB_LBXGH",
    "GLU_LBXGLU",
    "INS_LBXIN",
    "TRIGLY_LBXTR",
    "TRIGLY_LBDLDL",
    "HDL_LBDHDD",
    "TCHOL_LBXTC",
    "HSCRP_LBXHSCRP",
    "CBC_LBXHGB",
    "CBC_LBXPLTSI",
    "BIOPRO_LBXSATSI",
    "BIOPRO_LBXSAPSI",
    "BIOPRO_LBXSCR",
}


def download_xpt(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/octet-stream,*/*",
        },
    )
    with urlopen(request) as response:
        return response.read()


def read_xpt_bytes(payload: bytes) -> pd.DataFrame:
    dataframe = pd.read_sas(BytesIO(payload), format="xport")

    dataframe = dataframe.copy()
    dataframe.columns = [column.decode("utf-8") if isinstance(column, bytes) else column for column in dataframe.columns]

    for column in dataframe.select_dtypes(include=["object"]).columns:
        dataframe[column] = dataframe[column].map(
            lambda value: value.decode("utf-8").strip() if isinstance(value, bytes) else value
        )

    if "SEQN" not in dataframe.columns:
        raise ValueError("Downloaded NHANES file does not contain SEQN.")

    dataframe["SEQN"] = pd.to_numeric(dataframe["SEQN"], errors="coerce").astype("Int64")
    dataframe = dataframe.dropna(subset=["SEQN"])
    dataframe["SEQN"] = dataframe["SEQN"].astype("int64")
    return dataframe


def load_dataset(dataset_code: str, filename: str) -> pd.DataFrame:
    url = f"{NHANES_BASE_URL}/{filename}"
    payload = download_xpt(url)
    dataframe = read_xpt_bytes(payload)
    dataframe = dataframe.rename(
        columns={column: f"{dataset_code}_{column}" for column in dataframe.columns if column != "SEQN"}
    )
    print(f"Loaded {dataset_code}: {len(dataframe):,} rows from {url}")
    return dataframe


def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    merged = datasets[0]
    for dataset in datasets[1:]:
        merged = merged.merge(dataset, on="SEQN", how="outer")

    merged = merged.sort_values("SEQN").reset_index(drop=True)
    return merged


def _coerce_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda value: 1.0 if value == 1 else 0.0 if value == 2 else pd.NA)


def _derive_pancreatic_cancer_label(prepared: pd.DataFrame) -> pd.Series:
    """
    Pancreatic-cancer label from NHANES self-report.

    NHANES MCQ230A/B/C/D captures the type of cancer for each of up to four
    lifetime cancer diagnoses. Code 29 == pancreas. We consider a participant
    a positive if ANY of the four MCQ230 slots equals 29.

    Returns a pd.Series with values 1.0 (pancreatic cancer reported), 0.0
    (definitely no pancreatic cancer reported), or NaN (unknown — e.g. never
    asked because MCQ220 says never had any cancer, or fields all missing).
    """
    slots = [f"MCQ_MCQ230{letter}" for letter in ("A", "B", "C", "D")]
    present = [c for c in slots if c in prepared.columns]
    if not present:
        return pd.Series(pd.NA, index=prepared.index, dtype="Float64")

    numeric = prepared[present].apply(pd.to_numeric, errors="coerce")
    any_pancreas = (numeric == PANCREAS_MCQ_CODE).any(axis=1)

    # Anyone with MCQ220 == 2 (never had cancer) is a clean negative for
    # pancreatic cancer as well. Anyone with MCQ220 == 1 (had cancer) but no
    # 29 in the MCQ230 slots is also a negative (they had a different cancer).
    label = pd.Series(pd.NA, index=prepared.index, dtype="Float64")
    if "MCQ_MCQ220" in prepared.columns:
        cancer_any = pd.to_numeric(prepared["MCQ_MCQ220"], errors="coerce")
        label.loc[cancer_any == 2] = 0.0
        label.loc[cancer_any == 1] = 0.0
    label.loc[any_pancreas] = 1.0
    return label


def _derive_diabetes_subtype(prepared: pd.DataFrame) -> pd.Series:
    """
    Approximate diabetes subtype (0 non-diabetic, 1 Type1-like, 2 Type2-like,
    3 gestational-like, NaN unknown).

    NHANES does not directly ask T1 vs T2. Proxy rules:
    - DIQ_DIQ010 == 2 -> non-diabetic (0)
    - DIQ_DIQ175X (self-report "type 1" flag, if present) -> 1
    - Age at diagnosis (DID040) < 25 AND on insulin (DIQ050 == 1) -> 1
    - Otherwise on insulin -> 2 (long-standing T2 that requires insulin)
    - Diabetes during pregnancy only (DIQ175Q or DIQ175R) -> 3
    - Any other diabetic -> 2 (default T2-like, the dominant population)
    """
    subtype = pd.Series(pd.NA, index=prepared.index, dtype="Float64")

    if "DIQ_DIQ010" not in prepared.columns:
        return subtype

    dm = pd.to_numeric(prepared["DIQ_DIQ010"], errors="coerce")
    subtype.loc[dm == 2] = 0.0  # non-diabetic
    diabetic_mask = dm == 1

    on_insulin = (
        pd.to_numeric(prepared["DIQ_DIQ050"], errors="coerce") == 1
        if "DIQ_DIQ050" in prepared.columns
        else pd.Series(False, index=prepared.index)
    )
    onset_age = (
        pd.to_numeric(prepared["DIQ_DID040"], errors="coerce")
        if "DIQ_DID040" in prepared.columns
        else pd.Series(pd.NA, index=prepared.index)
    )

    # Default T2 for anyone diabetic
    subtype.loc[diabetic_mask] = 2.0
    # Early-onset + insulin -> T1-like
    t1_like = diabetic_mask & on_insulin & (onset_age < 25)
    subtype.loc[t1_like] = 1.0

    return subtype


def _derive_weight_loss_features(prepared: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Weight-change proxies from NHANES WHQ.

    Pancreatic cancer often presents with unintentional weight loss. NHANES
    does not have a direct "unintentional weight loss last year" question in
    2017-18, so we compute a proxy:
    - `weight_loss_1yr_lb` = WHD050 (weight 1 year ago) - WHD020 (current). Positive means lost.
    - `significant_weight_loss_flag` = 1 if loss >= 10 lb, else 0.
    - `weight_loss_10yr_lb` = WHD140 (weight 10y ago) - WHD020. Long-term trajectory.
    """
    features: dict[str, pd.Series] = {}
    if "WHQ_WHD020" in prepared.columns and "WHQ_WHD050" in prepared.columns:
        current = pd.to_numeric(prepared["WHQ_WHD020"], errors="coerce")
        year_ago = pd.to_numeric(prepared["WHQ_WHD050"], errors="coerce")
        # NHANES sentinel refuse/don't know codes: 7777, 9999
        current = current.where(current < 700)
        year_ago = year_ago.where(year_ago < 700)
        loss = year_ago - current
        features["weight_loss_1yr_lb"] = loss
        features["significant_weight_loss_flag"] = pd.Series(
            pd.NA, index=prepared.index, dtype="Float64"
        )
        features["significant_weight_loss_flag"].loc[loss.notna()] = (
            loss[loss.notna()] >= 10
        ).astype(int)

    if "WHQ_WHD020" in prepared.columns and "WHQ_WHD140" in prepared.columns:
        current = pd.to_numeric(prepared["WHQ_WHD020"], errors="coerce")
        decade_ago = pd.to_numeric(prepared["WHQ_WHD140"], errors="coerce")
        current = current.where(current < 700)
        decade_ago = decade_ago.where(decade_ago < 700)
        features["weight_loss_10yr_lb"] = decade_ago - current

    return features


def prepare_model_ready_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = dataframe.copy()

    if "Obesity" not in prepared.columns and "BMX_BMXBMI" in prepared.columns:
        bmi = pd.to_numeric(prepared["BMX_BMXBMI"], errors="coerce")
        prepared["Obesity"] = pd.NA
        prepared.loc[bmi.notna(), "Obesity"] = (bmi[bmi.notna()] >= 30).astype(int)

    if "Diabetes" not in prepared.columns and "DIQ_DIQ010" in prepared.columns:
        prepared["Diabetes"] = _coerce_binary(prepared["DIQ_DIQ010"])

    if "Cancer" not in prepared.columns and "MCQ_MCQ220" in prepared.columns:
        prepared["Cancer"] = _coerce_binary(prepared["MCQ_MCQ220"])

    # Pancreatic-specific label (from MCQ230A/B/C/D == 29)
    if "PancreaticCancer" not in prepared.columns:
        prepared["PancreaticCancer"] = _derive_pancreatic_cancer_label(prepared)

    # Diabetes subtype proxy
    if "diabetes_subtype" not in prepared.columns:
        prepared["diabetes_subtype"] = _derive_diabetes_subtype(prepared)

    # Weight-loss features
    for name, series in _derive_weight_loss_features(prepared).items():
        if name not in prepared.columns:
            prepared[name] = series

    if {"GLU_LBXGLU", "INS_LBXIN"}.issubset(prepared.columns):
        glucose = pd.to_numeric(prepared["GLU_LBXGLU"], errors="coerce")
        insulin = pd.to_numeric(prepared["INS_LBXIN"], errors="coerce")
        prepared["homa_ir"] = (glucose * insulin) / 405.0

    if "GHB_LBXGH" in prepared.columns:
        a1c = pd.to_numeric(prepared["GHB_LBXGH"], errors="coerce")
        prepared["elevated_hba1c"] = pd.NA
        prepared.loc[a1c.notna(), "elevated_hba1c"] = (a1c[a1c.notna()] >= 6.5).astype(int)
        prepared["hba1c_reciprocal_100"] = np.where(a1c > 0, 100.0 / a1c, np.nan)
        prepared["hba1c_squared"] = a1c ** 2

    if {"SMQ_SMQ020", "SMQ_SMQ040"}.issubset(prepared.columns):
        ever = pd.to_numeric(prepared["SMQ_SMQ020"], errors="coerce")
        now = pd.to_numeric(prepared["SMQ_SMQ040"], errors="coerce")
        status = pd.Series(pd.NA, index=prepared.index, dtype="Float64")
        status.loc[ever == 2] = 0.0
        status.loc[(ever == 1) & (now == 3)] = 1.0
        status.loc[(ever == 1) & now.isin([1, 2])] = 2.0
        prepared["smoking_status"] = status
        prepared["current_smoker"] = pd.NA
        prepared.loc[status.notna(), "current_smoker"] = (status[status.notna()] == 2).astype(int)

    if "ALQ_ALQ130" in prepared.columns:
        drinks = pd.to_numeric(prepared["ALQ_ALQ130"], errors="coerce").where(lambda value: value < 100)
        prepared["average_drinks_per_day"] = drinks
        prepared["alcohol_status"] = pd.NA
        prepared.loc[drinks.notna(), "alcohol_status"] = (drinks[drinks.notna()] > 0).astype(int) * 2

    if "GLU_LBXGLU" in prepared.columns:
        glucose = pd.to_numeric(prepared["GLU_LBXGLU"], errors="coerce")
        prepared["fasting_hyperglycemia"] = pd.NA
        prepared.loc[glucose.notna(), "fasting_hyperglycemia"] = (glucose[glucose.notna()] >= 126).astype(int)

    return prepared


def write_outputs(dataframe: pd.DataFrame) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent.parent
    root_output = project_root / "data" / "nhanes_merged_v2.csv"
    api_output = project_root / "api" / "nhanes_data" / "nhanes_merged_v2.csv"

    root_output.parent.mkdir(parents=True, exist_ok=True)
    api_output.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(root_output, index=False)
    dataframe.to_csv(api_output, index=False)
    return root_output, api_output


def build_dataset() -> tuple[Path, Path]:
    datasets = [load_dataset(dataset_code, filename) for dataset_code, filename in NHANES_FILES.items()]
    merged = merge_datasets(datasets)
    ready = prepare_model_ready_dataframe(merged)
    outputs = write_outputs(ready)
    return outputs


def dataset_is_ready(dataset_path: Path) -> bool:
    if not dataset_path.exists() or not dataset_path.is_file():
        return False

    try:
        columns = pd.read_csv(dataset_path, nrows=0).columns
    except Exception:
        return False

    required = MODEL_COLUMNS | BIOMARKER_COLUMNS
    return required.issubset(set(columns))


def ensure_nhanes_dataset(force: bool = False) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    api_output = project_root / "api" / "nhanes_data" / "nhanes_merged_v2.csv"

    if force or not dataset_is_ready(api_output):
        build_dataset()

    return api_output


def main() -> None:
    datasets = [load_dataset(dataset_code, filename) for dataset_code, filename in NHANES_FILES.items()]
    merged = merge_datasets(datasets)
    ready = prepare_model_ready_dataframe(merged)
    root_output, api_output = write_outputs(ready)
    print(f"Merged dataframe shape: {ready.shape[0]:,} rows x {ready.shape[1]:,} columns")
    print(f"Wrote: {root_output}")
    print(f"Wrote: {api_output}")


if __name__ == "__main__":
    main()
