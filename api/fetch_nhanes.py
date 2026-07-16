"""
This script is intended to create the first trustworthy NHANES prototype for the KERI
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

import pandas as pd


NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
NHANES_FILES = {
    "DEMO": "DEMO_J.xpt",
    "BMX": "BMX_J.xpt",
    "DIQ": "DIQ_J.xpt",
    "MCQ": "MCQ_J.xpt",
}

MODEL_COLUMNS = {"Diabetes", "Cancer", "Obesity"}


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

    return prepared


def write_outputs(dataframe: pd.DataFrame) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent.parent
    root_output = project_root / "data" / "nhanes_merged.csv"
    api_output = project_root / "api" / "nhanes_data" / "nhanes_merged.csv"

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

    return MODEL_COLUMNS.issubset(set(columns))


def ensure_nhanes_dataset(force: bool = False) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    api_output = project_root / "api" / "nhanes_data" / "nhanes_merged.csv"

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