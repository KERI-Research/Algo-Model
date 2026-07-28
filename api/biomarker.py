from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from chromadb import PersistentClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional at import time during partial installs
    XGBClassifier = None


ARTIFACT_VERSION = "v1"
MODEL_NAME = "diapan_hist_gradient_boosting_v1"
ALTERNATE_MODEL_NAME = "diapan_xgboost_v1"
# Default (NHANES) required/optional field sets. When training on TCGA-CDR (or
# any dataset where these NHANES-specific columns are absent or all-NaN),
# _resolve_field_sets() below narrows both lists to the columns that actually
# carry signal in the given dataframe.
REQUIRED_FIELDS = [
    "Diabetes",
    "DEMO_RIDAGEYR",
    "DEMO_RIAGENDR",
    "BMX_BMXBMI",
]
OPTIONAL_HIGH_IMPACT_FIELDS = [
    "GHB_LBXGH",
    "GLU_LBXGLU",
    "INS_LBXIN",
    "CPEP_LBXCPSI",
]
# NOTE: tcga_followup_days/event and tcga_pfi_days/event are DELIBERATELY
# excluded here — the training labels (5y mortality, 5y progression) are
# derived from them and including them would trivially leak (AUROC ~1.0).
# tcga_treatment_response captures response-to-first-course-of-therapy and is
# measured AFTER initial diagnosis; it is a strong prognostic signal but is
# post-diagnostic, so downstream users should treat model output as
# "prognostic given response so far" rather than "pre-treatment prediction."
TCGA_EXTRA_FEATURES = [
    "tcga_stage_ordinal",
    "tcga_grade_ordinal",
    "tcga_tumor_status",
    "tcga_treatment_response",
]
SENTINEL_VALUES = {7, 9, 77, 99, 777, 999, 6666, 7777, 9999, 99999}
BASE_FEATURES = [
    "Diabetes",
    "Obesity",
    "DEMO_RIDAGEYR",
    "DEMO_RIAGENDR",
    "DEMO_RIDRETH3",
    "BMX_BMXBMI",
    "BMX_BMXWAIST",
    "DIQ_DID040",
    "DIQ_DIQ160",
    "DIQ_DIQ170",
    "DIQ_DIQ180",
    "GHB_LBXGH",
    "GLU_LBXGLU",
    "INS_LBXIN",
    "TRIGLY_LBXTR",
    "TRIGLY_LBDLDL",
    "HDL_LBDHDD",
    "TCHOL_LBXTC",
    "HSCRP_LBXHSCRP",
    "homa_ir",
    "elevated_hba1c",
    "fasting_hyperglycemia",
    "diabetes_duration_years",
    "recent_diabetes_onset",
    "age_bmi_interaction",
    "waist_bmi_interaction",
    # Brief-aligned pancreatic-risk features
    "diabetes_subtype",
    "weight_loss_1yr_lb",
    "significant_weight_loss_flag",
    "weight_loss_10yr_lb",
    # Repeated-cross-sectional trajectory proxies from nhanes_multicycle.csv.
    # These are not within-patient slopes and must not be described as such.
    "survey_cycle_index",
    "hba1c_cycle_age_sex_z",
    "hba1c_age_interaction",
    "hba1c_diabetes_duration_interaction",
    "hba1c_weight_loss_interaction",
]


@dataclass(frozen=True)
class BiomarkerArtifactPaths:
    model_path: Path
    chroma_path: Path


def _coerce_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.where(numeric == 1, 1.0, np.where(numeric == 2, 0.0, np.nan)),
        index=series.index,
    )


def _replace_sentinels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(numeric.isin(SENTINEL_VALUES), np.nan)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    if "Obesity" not in prepared.columns and "BMX_BMXBMI" in prepared.columns:
        bmi = _replace_sentinels(prepared["BMX_BMXBMI"])
        prepared["Obesity"] = np.where(bmi >= 30, 1.0, np.where(bmi.notna(), 0.0, np.nan))

    if "Diabetes" not in prepared.columns and "DIQ_DIQ010" in prepared.columns:
        prepared["Diabetes"] = _coerce_binary(prepared["DIQ_DIQ010"])

    if "Cancer" not in prepared.columns and "MCQ_MCQ220" in prepared.columns:
        prepared["Cancer"] = _coerce_binary(prepared["MCQ_MCQ220"])

    for column in [
        "DEMO_RIDAGEYR",
        "DEMO_RIAGENDR",
        "DEMO_RIDRETH3",
        "BMX_BMXBMI",
        "BMX_BMXWAIST",
        "DIQ_DID040",
        "DIQ_DIQ160",
        "DIQ_DIQ170",
        "DIQ_DIQ180",
    ]:
        if column in prepared.columns:
            prepared[column] = _replace_sentinels(prepared[column])

    if {"DEMO_RIDAGEYR", "DIQ_DID040"}.issubset(prepared.columns):
        onset_age = prepared["DIQ_DID040"]
        current_age = prepared["DEMO_RIDAGEYR"]
        duration = current_age - onset_age
        duration = duration.where((duration >= 0) & (duration <= 80), np.nan)
        prepared["diabetes_duration_years"] = duration
        prepared["recent_diabetes_onset"] = np.where(duration <= 3, 1.0, np.where(duration.notna(), 0.0, np.nan))
    else:
        prepared["diabetes_duration_years"] = np.nan
        prepared["recent_diabetes_onset"] = np.nan

    bmi = prepared.get("BMX_BMXBMI", pd.Series(np.nan, index=prepared.index))
    waist = prepared.get("BMX_BMXWAIST", pd.Series(np.nan, index=prepared.index))
    age = prepared.get("DEMO_RIDAGEYR", pd.Series(np.nan, index=prepared.index))
    prepared["age_bmi_interaction"] = age * bmi
    prepared["waist_bmi_interaction"] = waist * bmi

    return prepared


def _artifact_paths(
    data_path: str | Path,
    target: str = "Cancer",
    cohort_filter: str | None = None,
) -> BiomarkerArtifactPaths:
    dataset_name = Path(data_path).stem
    # Keep default target's artifact path stable for backward compatibility
    suffix = "" if target == "Cancer" else f"_{target.lower()}"
    if cohort_filter:
        suffix += f"__{cohort_filter}"
    base_dir = Path(__file__).resolve().parent / "model_artifacts" / f"{dataset_name}{suffix}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return BiomarkerArtifactPaths(
        model_path=base_dir / f"biomarker_{ARTIFACT_VERSION}.joblib",
        chroma_path=base_dir / "chroma",
    )


def _select_features(prepared: pd.DataFrame) -> list[str]:
    features = [feature for feature in BASE_FEATURES if feature in prepared.columns]
    # Include TCGA-native features when present
    for feature in TCGA_EXTRA_FEATURES:
        if feature in prepared.columns and feature not in features:
            features.append(feature)
    # Include auto-generated TCGA cancer-type one-hot flags
    for column in prepared.columns:
        if column.startswith("tcga_type_") and column not in features:
            features.append(column)
    # Drop features that are entirely NaN for this dataset
    features = [f for f in features if prepared[f].notna().any()]
    return features


def _resolve_field_sets(prepared: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Narrow REQUIRED / OPTIONAL fields to columns that carry data in this frame."""
    required = [f for f in REQUIRED_FIELDS if f in prepared.columns and prepared[f].notna().any()]
    optional = [f for f in OPTIONAL_HIGH_IMPACT_FIELDS if f in prepared.columns and prepared[f].notna().any()]

    # TCGA: promote high-impact optional features into the "optional but
    # tracked" list but do NOT force dropna on them — they are commonly sparse
    # (grade is missing for ~40% of patients, tumor_status ~10%, etc.).
    for feature in ("tcga_stage_ordinal", "tcga_grade_ordinal", "tcga_tumor_status", "tcga_treatment_response"):
        if feature in prepared.columns and prepared[feature].notna().any():
            if feature not in optional:
                optional.append(feature)

    # Ensure at least one required field exists so training does not silently
    # keep NaN-only rows.
    if not required:
        for candidate in ("DEMO_RIDAGEYR", "tcga_stage_ordinal"):
            if candidate in prepared.columns and prepared[candidate].notna().any():
                required = [candidate]
                break
    return required, optional


def _clean_training_frame(prepared: pd.DataFrame, features: list[str], target: str = "Cancer") -> pd.DataFrame:
    model_df = prepared[[target, *features]].copy()
    for column in model_df.columns:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df[(model_df[target] == 0) | (model_df[target] == 1)]
    model_df = model_df.dropna(subset=[target])
    return model_df


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def _stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Any,
    repeats: int = 500,
    seed: int = 42,
) -> list[float]:
    """Deterministic stratified bootstrap 95% CI for rare-event metrics.

    Positive and negative test indices are sampled separately so each bootstrap
    replicate remains evaluable even when the held-out positive count is small.
    This interval reflects test-sample uncertainty only; it does not account for
    model-selection, survey-design or outcome-misclassification uncertainty.
    """
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if len(pos) < 2 or len(neg) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        sampled = np.concatenate(
            [rng.choice(pos, size=len(pos), replace=True),
             rng.choice(neg, size=len(neg), replace=True)]
        )
        values.append(float(metric_fn(y_true[sampled], y_prob[sampled])))
    low, high = np.quantile(values, [0.025, 0.975])
    return [round(float(low), 6), round(float(high), 6)]


def _rank_biomarkers(
    fitted_model: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    importances: Any = permutation_importance(
        fitted_model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="average_precision",
    )

    positive_mask = y_test == 1
    negative_mask = y_test == 0
    rankings: list[dict[str, Any]] = []
    for index, column in enumerate(x_test.columns):
        positive_values = np.asarray(cast(Any, x_test.loc[positive_mask, column]), dtype=float)
        negative_values = np.asarray(cast(Any, x_test.loc[negative_mask, column]), dtype=float)
        positive_mean = float(np.nanmean(positive_values)) if positive_mask.any() else np.nan
        negative_mean = float(np.nanmean(negative_values)) if negative_mask.any() else np.nan
        direction = positive_mean - negative_mean if np.isfinite(positive_mean) and np.isfinite(negative_mean) else 0.0
        rankings.append(
            {
                "feature": column,
                "importance": round(float(importances.importances_mean[index]), 6),
                "stability": round(float(importances.importances_std[index]), 6),
                "direction": "higher_in_positive" if direction > 0 else "lower_in_positive",
                "mean_shift": round(float(direction), 6),
            }
        )

    rankings.sort(key=lambda item: abs(item["importance"]), reverse=True)
    return rankings


def _build_case_document(row: pd.Series, features: list[str], target: str = "Cancer") -> str:
    fragments = [f"SEQN={int(row['SEQN'])}" if pd.notna(row.get("SEQN")) else "SEQN=unknown"]
    for feature in features[:8]:
        value = row.get(feature)
        if pd.notna(value):
            fragments.append(f"{feature}={round(float(value), 3)}")
    fragments.append(f"{target}={int(row[target])}")
    diabetes_value = row.get("Diabetes") if "Diabetes" in row.index else None
    fragments.append(
        f"Diabetes={int(diabetes_value) if diabetes_value is not None and pd.notna(diabetes_value) else 'unknown'}"
    )
    return "; ".join(fragments)


def _store_training_memory(
    prepared: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler,
    chroma_path: Path,
    target: str = "Cancer",
    medians: dict[str, float] | None = None,
) -> dict[str, Any]:
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name="biomarker_cases")

    selected_columns = [*features, target]
    if "Diabetes" in prepared.columns and "Diabetes" not in features and "Diabetes" != target:
        selected_columns.append("Diabetes")
    memory_df = prepared[selected_columns].copy()
    if "SEQN" in prepared.columns:
        memory_df.insert(0, "SEQN", prepared["SEQN"])
    memory_df = memory_df.dropna(subset=[target])
    if medians:
        memory_df = memory_df.fillna(pd.Series(medians))
    memory_df = memory_df.fillna(memory_df.median(numeric_only=True))
    memory_df = memory_df.head(600)

    feature_matrix = memory_df[features].astype(float)
    embeddings = scaler.transform(feature_matrix).tolist()
    ids = [f"case-{index}" for index in range(len(memory_df))]
    documents = [_build_case_document(row, features, target=target) for _, row in memory_df.iterrows()]
    metadatas = [
        {
            "target_name": target,
            "target_value": int(row[target]),
            "diabetes": (
                int(row["Diabetes"])
                if "Diabetes" in memory_df.columns and pd.notna(row.get("Diabetes"))
                else -1
            ),
        }
        for _, row in memory_df.iterrows()
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return {"collection": "biomarker_cases", "stored_cases": len(memory_df)}


def _build_candidate_models(positive_rate: float = 0.5) -> list[tuple[str, Any]]:
    """Build model candidates. When positive class is rare (positive_rate < 0.1),
    apply class-weight balancing so models don't collapse to the majority class."""
    is_rare = positive_rate < 0.1 and positive_rate > 0.0
    class_weight = "balanced" if is_rare else None
    scale_pos_weight = (1.0 - positive_rate) / positive_rate if is_rare else 1.0

    candidates: list[tuple[str, Any]] = [
        (
            MODEL_NAME,
            HistGradientBoostingClassifier(
                random_state=42,
                max_depth=4,
                max_iter=240,
                learning_rate=0.05,
                min_samples_leaf=20,
                class_weight=class_weight,
            ),
        )
    ]

    if XGBClassifier is not None:
        candidates.append(
            (
                ALTERNATE_MODEL_NAME,
                XGBClassifier(
                    n_estimators=260,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                ),
            )
        )

    return candidates


def _fit_and_score_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    positive_rate: float = 0.5,
) -> tuple[str, Any, dict[str, dict[str, float]], np.ndarray]:
    benchmark_metrics: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model: Any = None
    best_probabilities: np.ndarray | None = None
    best_score = (-1.0, -1.0)

    for model_name, model in _build_candidate_models(positive_rate=positive_rate):
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = {
            "auroc": round(_safe_auc(y_test, probabilities), 6),
            "auprc": round(_safe_auprc(y_test, probabilities), 6),
        }
        benchmark_metrics[model_name] = metrics
        score = (metrics["auprc"], metrics["auroc"])
        if score > best_score:
            best_score = score
            best_name = model_name
            best_model = model
            best_probabilities = probabilities

    if best_model is None or best_probabilities is None:
        raise ValueError("Unable to fit any biomarker candidate models.")

    return best_name, best_model, benchmark_metrics, best_probabilities


def train_biomarker_model(
    data_path: str,
    force: bool = False,
    target: str = "Cancer",
    cohort_filter: str | None = None,
) -> dict[str, Any]:
    """
    Train a biomarker classifier for ``target`` on the dataset at ``data_path``.

    ``cohort_filter`` optionally restricts the training rows before splitting.
    Currently supported: 'diabetics_only' — keep only rows where Diabetes == 1.
    This is used for the pancreatic-in-diabetics risk-stratification model per
    the DiaPan brief.
    """
    paths = _artifact_paths(data_path, target=target, cohort_filter=cohort_filter)
    if paths.model_path.exists() and not force:
        return joblib.load(paths.model_path)

    df = pd.read_csv(data_path)
    prepared = _prepare_dataframe(df)
    if cohort_filter == "diabetics_only":
        if "Diabetes" not in prepared.columns:
            raise ValueError("'diabetics_only' cohort filter requires a Diabetes column.")
        before = len(prepared)
        prepared = prepared[pd.to_numeric(prepared["Diabetes"], errors="coerce") == 1].copy()
        print(f"cohort_filter=diabetics_only reduced {before} -> {len(prepared)} rows")
    features = _select_features(prepared)
    # Never let the target leak into itself as a feature
    features = [f for f in features if f != target]
    if not features:
        raise ValueError("No biomarker features were available in the dataset.")
    if target not in prepared.columns or not prepared[target].notna().any():
        raise ValueError(f"Target column '{target}' is unavailable or all-NaN in this dataset.")

    required_fields, optional_fields = _resolve_field_sets(prepared)
    model_df = _clean_training_frame(prepared, features, target=target)
    if required_fields:
        model_df = model_df.dropna(subset=required_fields)
    if len(model_df) < 120:
        raise ValueError(
            f"Insufficient rows available for biomarker training after cleaning "
            f"(got {len(model_df)} rows, required_fields={required_fields})."
        )

    x = model_df[features].copy()
    y = model_df[target].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    # Learn imputation statistics on training rows only. Computing medians
    # before the split would expose the held-out feature distribution.
    numeric_medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(numeric_medians)
    x_test = x_test.fillna(numeric_medians)

    positive_rate = float(y.mean())
    model_name, model, benchmark_metrics, test_prob = _fit_and_score_models(
        x_train,
        y_train,
        x_test,
        y_test,
        positive_rate=positive_rate,
    )
    biomarker_ranking = _rank_biomarkers(model, x_test, y_test)

    scaler = StandardScaler()
    scaler.fit(x_train[features])
    memory_summary = _store_training_memory(
        model_df,
        features,
        scaler,
        paths.chroma_path,
        target=target,
        medians=numeric_medians.to_dict(),
    )

    artifact = {
        "model": model,
        "model_name": model_name,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "features": features,
        "required_fields": required_fields,
        "optional_high_impact_fields": optional_fields,
        "medians": numeric_medians.to_dict(),
        "retrieval_scaler": scaler,
        "metrics": {
            "auroc": round(_safe_auc(y_test, test_prob), 6),
            "auprc": round(_safe_auprc(y_test, test_prob), 6),
            "auroc_ci_95": _stratified_bootstrap_ci(
                y_test, test_prob, roc_auc_score
            ),
            "auprc_ci_95": _stratified_bootstrap_ci(
                y_test, test_prob, average_precision_score
            ),
            "brier_score": round(float(brier_score_loss(y_test, test_prob)), 6),
            "positive_rate": round(float(y.mean()), 6),
            "test_positive_rate": round(float(y_test.mean()), 6),
            "auprc_lift_over_prevalence": round(
                float(_safe_auprc(y_test, test_prob) / max(y_test.mean(), 1e-12)), 6
            ),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "test_positives": int((y_test == 1).sum()),
            "test_negatives": int((y_test == 0).sum()),
        },
        "benchmarks": benchmark_metrics,
        "biomarker_ranking": biomarker_ranking,
        "memory_summary": memory_summary,
        "target": target,
        "cohort_filter": cohort_filter,
        "cohort_summary": {
            "rows_used": int(len(model_df)),
            "target_positive_rate": round(float(model_df[target].mean()), 6),
            "diabetes_rate": (
                round(float(model_df["Diabetes"].mean()), 6)
                if "Diabetes" in model_df.columns and model_df["Diabetes"].notna().any()
                else None
            ),
            "available_features": len(features),
        },
    }
    joblib.dump(artifact, paths.model_path)
    return artifact


def load_biomarker_model(data_path: str, target: str = "Cancer", cohort_filter: str | None = None) -> dict[str, Any]:
    paths = _artifact_paths(data_path, target=target, cohort_filter=cohort_filter)
    if not paths.model_path.exists():
        return train_biomarker_model(data_path, force=True, target=target, cohort_filter=cohort_filter)
    return joblib.load(paths.model_path)


def _normalize_patient_record(record: dict[str, Any], artifact: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    row = {feature: np.nan for feature in artifact["features"]}
    row.update(record)
    frame = pd.DataFrame([row])
    prepared = _prepare_dataframe(frame)
    features = artifact["features"]

    missing_required = [field for field in artifact["required_fields"] if pd.isna(prepared.at[0, field])]
    missing_optional = [field for field in artifact["optional_high_impact_fields"] if pd.isna(prepared.at[0, field])]

    for feature, median in artifact["medians"].items():
        if feature in prepared.columns:
            prepared[feature] = prepared[feature].fillna(median)

    return prepared[features], {
        "missing_required_fields": missing_required,
        "missing_optional_fields": missing_optional,
    }


def _query_similar_cases(data_path: str, artifact: dict[str, Any], patient_features: pd.DataFrame) -> list[dict[str, Any]]:
    paths = _artifact_paths(data_path)
    if not paths.chroma_path.exists():
        return []
    client = PersistentClient(path=str(paths.chroma_path))
    try:
        collection = client.get_collection(name="biomarker_cases")
    except Exception:
        return []
    embedding = artifact["retrieval_scaler"].transform(patient_features.astype(float)).tolist()[0]
    result = collection.query(query_embeddings=[embedding], n_results=3)
    matches: list[dict[str, Any]] = []
    for index, document in enumerate(result.get("documents", [[]])[0]):
        matches.append(
            {
                "document": document,
                "distance": round(float(result.get("distances", [[]])[0][index]), 6),
                "metadata": result.get("metadatas", [[]])[0][index],
            }
        )
    return matches


def _build_follow_up_questions(missing_fields: list[str], optional: bool) -> list[str]:
    prefix = "Required" if not optional else "Optional"
    return [
        f"{prefix} field needed: can you provide {field}?"
        for field in missing_fields
    ]


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.78:
        return "high"
    if confidence >= 0.62:
        return "moderate"
    return "low"


def _build_patient_assessment(data_path: str, artifact: dict[str, Any], patient_record: dict[str, Any]) -> dict[str, Any]:
    patient_features, missingness = _normalize_patient_record(patient_record, artifact)
    missing_required = missingness["missing_required_fields"]
    missing_optional = missingness["missing_optional_fields"]

    if missing_required:
        return {
            "status": "needs_required_fields",
            "missing_required_fields": missing_required,
            "follow_up_questions": _build_follow_up_questions(missing_required, optional=False),
            "confidence": 0.0,
            "confidence_label": "insufficient",
        }

    probability = float(artifact["model"].predict_proba(patient_features)[:, 1][0])
    completeness = 1.0 - (len(missing_optional) / max(len(artifact["optional_high_impact_fields"]), 1))
    certainty = abs(probability - 0.5) * 2.0
    confidence = round(float((0.6 * certainty) + (0.4 * completeness)), 6)
    similar_cases = _query_similar_cases(data_path, artifact, patient_features)

    optional_follow_ups: list[str] = []
    status = "scored"
    if confidence < 0.62 and missing_optional:
        optional_follow_ups = missing_optional[:3]
        status = "needs_more_context"

    top_biomarkers = artifact["biomarker_ranking"][:5]
    biomarker_names = ", ".join(item["feature"] for item in top_biomarkers[:3])
    explanation = (
        f"The record was scored against the NHANES-trained biomarker model. "
        f"Current cancer-linked risk probability is {probability:.3f}, with { _confidence_label(confidence) } confidence. "
        f"The strongest cohort biomarkers in this model are {biomarker_names}."
    )

    return {
        "status": status,
        "cancer_risk_probability": round(probability, 6),
        "diabetes_cancer_link_score": round(float((probability + patient_features.iloc[0].get("Diabetes", 0.0)) / 2.0), 6),
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "missing_required_fields": [],
        "missing_optional_fields": missing_optional,
        "recommended_follow_up_fields": optional_follow_ups,
        "follow_up_questions": _build_follow_up_questions(optional_follow_ups, optional=True),
        "similar_cases": similar_cases,
        "explanation": explanation,
    }


def execute_biomarker_discovery(
    data_path: str,
    patient_record: dict[str, Any] | None = None,
    top_k: int = 8,
    force_retrain: bool = False,
    target: str = "Cancer",
    cohort_filter: str | None = None,
) -> dict[str, Any]:
    if force_retrain:
        artifact = train_biomarker_model(data_path, force=True, target=target, cohort_filter=cohort_filter)
    else:
        artifact = load_biomarker_model(data_path, target=target, cohort_filter=cohort_filter)
    paths = _artifact_paths(data_path, target=target, cohort_filter=cohort_filter)

    response: dict[str, Any] = {
        "model": artifact["model_name"],
        "artifact_version": artifact["artifact_version"],
        "created_at": artifact["created_at"],
        "required_fields": artifact["required_fields"],
        "optional_high_impact_fields": artifact["optional_high_impact_fields"],
        "metrics": artifact["metrics"],
        "benchmarks": artifact.get("benchmarks", {}),
        "cohort_summary": artifact["cohort_summary"],
        "biomarker_ranking": artifact["biomarker_ranking"][:top_k],
        "memory": {
            **artifact["memory_summary"],
            "path": str(paths.chroma_path),
        },
        "notes": [
            "NHANES biomarker workflow now includes lab features such as HbA1c, fasting glucose, insulin, lipids, and hs-CRP when present in the 2017-2018 public files.",
            "The training pipeline benchmarks HistGradientBoosting against XGBoost and preserves the same artifact contract while selecting the stronger model by AUPRC, then AUROC.",
            "A small local LLM can be layered on top of this output for explanation and question phrasing, but the predictor itself is tabular-first.",
        ],
    }
    if patient_record is not None:
        response["patient_assessment"] = _build_patient_assessment(data_path, artifact, patient_record)

    return response
