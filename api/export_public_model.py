"""Export DiaPan's final research model as a public Hugging Face-ready artifact."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from biomarker import _build_candidate_models, _prepare_dataframe, _select_features
from validate_cycle_holdout import TEMPORAL_PROXY_FEATURES


FEATURE_DESCRIPTIONS = {
    "Obesity": ("Derived BMI >= 30 flag", "1 indicates obesity; not inherently better/worse"),
    "DEMO_RIDAGEYR": ("Age in years", "Higher age was associated with higher risk in this cohort"),
    "DEMO_RIAGENDR": ("NHANES sex code: 1 male, 2 female", "Categorical; higher is not better"),
    "DEMO_RIDRETH3": ("NHANES race/ethnicity category code", "Categorical; higher is not better"),
    "BMX_BMXBMI": ("Body mass index, kg/m2", "Higher is not automatically better; nonlinear risk"),
    "BMX_BMXWAIST": ("Waist circumference, cm", "Higher generally indicates central adiposity"),
    "DIQ_DID040": ("Age when diabetes was diagnosed", "Lower can indicate earlier onset"),
    "DIQ_DIQ160": ("Ever told prediabetes, binary questionnaire code", "Categorical questionnaire code"),
    "DIQ_DIQ170": ("Diabetes risk questionnaire item", "Categorical questionnaire code"),
    "DIQ_DIQ180": ("Diabetes risk questionnaire item", "Categorical questionnaire code"),
    "GHB_LBXGH": ("Measured HbA1c, percent", "Higher means poorer glycaemic control; model effect is contextual"),
    "GLU_LBXGLU": ("Fasting plasma glucose, mg/dL", "Higher means higher fasting glucose"),
    "INS_LBXIN": ("Fasting insulin, uU/mL", "Higher is not automatically better; interpret with glucose"),
    "CPEP_LBXCPSI": ("Fasting C-peptide, SI units; early cycles only", "Higher indicates greater endogenous insulin secretion"),
    "TRIGLY_LBXTR": ("Triglycerides, mg/dL", "Higher is generally metabolically adverse"),
    "TRIGLY_LBDLDL": ("LDL cholesterol, mg/dL", "Higher is generally metabolically adverse"),
    "HDL_LBDHDD": ("HDL cholesterol, mg/dL", "Higher is generally favourable metabolically"),
    "TCHOL_LBXTC": ("Total cholesterol, mg/dL", "Neither universally better nor worse in this research model"),
    "HSCRP_LBXHSCRP": ("High-sensitivity C-reactive protein", "Higher indicates inflammation; evidence for PDAC is weak"),
    "homa_ir": ("HOMA-IR proxy: glucose x insulin / 405", "Higher indicates greater insulin resistance"),
    "elevated_hba1c": ("HbA1c >= 6.5 flag", "1 indicates diabetic-range HbA1c"),
    "fasting_hyperglycemia": ("Fasting glucose >= 126 flag", "1 indicates diabetic-range fasting glucose"),
    "diabetes_duration_years": ("Current age minus reported diagnosis age", "Risk is duration-dependent, not simply monotonic"),
    "recent_diabetes_onset": ("Diabetes duration <= 3 years", "1 is a recognised risk-enrichment signal"),
    "age_bmi_interaction": ("Age multiplied by BMI", "Interaction; no standalone clinical unit"),
    "waist_bmi_interaction": ("Waist circumference multiplied by BMI", "Interaction; higher reflects combined adiposity"),
    "diabetes_subtype": ("Proxy: 0 non-diabetic, 1 Type-1-like, 2 Type-2-like", "Unvalidated heuristic; higher is not better"),
    "weight_loss_1yr_lb": ("Weight one year ago minus current weight, lb", "Positive means weight loss"),
    "significant_weight_loss_flag": ("Weight loss >= 10 lb", "1 means significant recent weight loss"),
    "weight_loss_10yr_lb": ("Weight ten years ago minus current weight, lb", "Positive means long-term weight loss"),
    "hba1c_age_interaction": ("HbA1c multiplied by age", "Interaction; no standalone clinical unit"),
    "hba1c_diabetes_duration_interaction": ("HbA1c multiplied by diabetes duration", "Interaction"),
    "hba1c_weight_loss_interaction": ("HbA1c multiplied by one-year weight loss", "Interaction"),
}


INFERENCE_SOURCE = '''"""Inference helper for the DiaPan pancreatic-risk research model."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd


class DiaPanRiskModel:
    def __init__(self, artifact_dir: str | Path | None = None):
        root = Path(artifact_dir or Path(__file__).resolve().parent)
        self.artifact = joblib.load(root / "model.joblib")

    def predict(self, records: list[dict]) -> list[dict]:
        frame = pd.DataFrame(records)
        for feature in self.artifact["features"]:
            if feature not in frame:
                frame[feature] = float("nan")
        x = frame[self.artifact["features"]].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(pd.Series(self.artifact["medians"]))
        probability = self.artifact["model"].predict_proba(x)[:, 1]
        return [
            {
                "research_risk_score": round(float(value), 8),
                "interpretation": (
                    "Higher means higher relative model-assigned risk. "
                    "This is not a calibrated clinical probability."
                ),
            }
            for value in probability
        ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON file containing one object or a list")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    records = data if isinstance(data, list) else [data]
    print(json.dumps(DiaPanRiskModel().predict(records), indent=2))
'''


BENCHMARK_SOURCE = '''"""Evaluate prediction CSVs from any external/Hugging Face model.

Required columns: global_participant_id, probability. The command joins these
against DiaPan's labelled diabetic cohort and reports model-comparable metrics.
"""
from __future__ import annotations
import argparse, json
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True)
parser.add_argument("--labels", required=True, help="Path to nhanes_multicycle.csv")
args = parser.parse_args()
pred = pd.read_csv(args.predictions)
labels = pd.read_csv(args.labels, low_memory=False)
labels = labels[(labels["Diabetes"] == 1) & labels["PancreaticCancer"].notna()]
data = labels[["global_participant_id", "PancreaticCancer"]].merge(
    pred[["global_participant_id", "probability"]], on="global_participant_id", how="inner"
)
y = data["PancreaticCancer"].astype(int).to_numpy()
p = data["probability"].astype(float).to_numpy()
result = {
    "rows": len(data),
    "positives": int(y.sum()),
    "auroc": float(roc_auc_score(y, p)),
    "auprc": float(average_precision_score(y, p)),
    "auprc_lift_over_prevalence": float(average_precision_score(y, p) / y.mean()),
    "brier_score": float(brier_score_loss(y, p)),
}
print(json.dumps(result, indent=2))
'''


def export(dataset: Path, benchmark_report: Path, output: Path) -> Path:
    frame = _prepare_dataframe(pd.read_csv(dataset, low_memory=False))
    frame = frame[(pd.to_numeric(frame["Diabetes"], errors="coerce") == 1)]
    frame = frame[frame["PancreaticCancer"].isin([0, 1])].copy()
    features = [
        feature for feature in _select_features(frame)
        if feature not in TEMPORAL_PROXY_FEATURES and feature != "PancreaticCancer"
    ]
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    y = frame["PancreaticCancer"].astype(int).to_numpy()
    medians = x.median(numeric_only=True)
    x = x.fillna(medians)
    positive_rate = float(y.mean())
    models = dict(_build_candidate_models(positive_rate=positive_rate))
    model_name = "diapan_xgboost_v1"
    model = models[model_name]
    model.fit(x, y)

    benchmark = json.loads(benchmark_report.read_text())
    metrics = benchmark["pooled_out_of_cycle"]["clinical_only"][model_name]
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_name": model_name,
        "target": "PancreaticCancer",
        "cohort_filter": "diabetics_only",
        "features": features,
        "medians": medians.to_dict(),
        "training_rows": len(frame),
        "training_positives": int(y.sum()),
        "score_semantics": (
            "Higher is higher relative model-assigned risk. Scores are not "
            "calibrated absolute clinical probabilities."
        ),
        "temporal_validation": metrics,
    }
    joblib.dump(artifact, output / "model.joblib")

    schema = {
        "target": {
            "name": "PancreaticCancer",
            "positive": "Self-reported pancreatic cancer, MCQ230A-D code 39",
            "negative": "No pancreatic site reported",
        },
        "features": {
            feature: {
                "description": FEATURE_DESCRIPTIONS.get(feature, ("Derived model feature", ""))[0],
                "direction": FEATURE_DESCRIPTIONS.get(feature, ("", "No simple direction"))[1],
                "missing_value_handling": f"Imputed with training median {medians.get(feature)}",
            }
            for feature in features
        },
    }
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    (output / "config.json").write_text(json.dumps({
        "pipeline_tag": "tabular-classification",
        "library_name": "xgboost",
        "license": "cc-by-4.0",
        "model_name": "DiaPan-XGB v1",
        "intended_use": "Research benchmarking and hypothesis generation only",
        "temporal_validation": metrics,
    }, indent=2))
    (output / "inference.py").write_text(INFERENCE_SOURCE)
    (output / "benchmark_predictions.py").write_text(BENCHMARK_SOURCE)
    (output / "requirements.txt").write_text(
        "pandas>=2.2\nscikit-learn>=1.5\nxgboost>=2.1\njoblib>=1.4\n"
    )
    sample = {
        "Diabetes": 1,
        "DEMO_RIDAGEYR": 62,
        "DEMO_RIAGENDR": 1,
        "BMX_BMXBMI": 29.4,
        "BMX_BMXWAIST": 102.0,
        "GHB_LBXGH": 7.1,
        "GLU_LBXGLU": 135.0,
        "INS_LBXIN": 14.0,
        "diabetes_duration_years": 2.0,
        "recent_diabetes_onset": 1,
        "weight_loss_1yr_lb": 12.0,
        "significant_weight_loss_flag": 1,
    }
    (output / "sample_input.json").write_text(json.dumps(sample, indent=2))
    shutil.copy(benchmark_report, output / "benchmark_results.json")
    model_card = f"""---
license: cc-by-4.0
library_name: xgboost
pipeline_tag: tabular-classification
tags:
  - medical
  - tabular-classification
  - pancreatic-cancer
  - diabetes
  - nhanes
  - research
---

# DiaPan-XGB v1

## Model summary

This research model ranks **self-reported prevalent pancreatic-cancer risk
inside the NHANES diabetic cohort** using metabolic, demographic and
behavioural features. It is exported from the DiaPan MSc research project for
reproducible benchmarking.

**This is not a diagnostic device. A higher score is not a validated absolute
probability of pancreatic cancer.**

## Score meaning

- **Higher `research_risk_score`:** the model assigns greater relative risk.
- **Lower score:** the model assigns lower relative risk.
- There is no clinically validated threshold.
- Class weighting means raw probabilities are not calibrated incidence risks.

Column meanings and expected direction are provided in `feature_schema.json`.
Categorical codes and interaction terms do not have a meaningful "higher is
better" interpretation.

## Training cohort

- Source: pooled NHANES 1999-March 2020 repeated cross-sections
- Filter: `Diabetes == 1`
- Target: self-reported pancreatic cancer, `MCQ230A-D == 39`
- Rows used for final fit: {len(frame):,}
- Positive cases: {int(y.sum())}
- Features: {len(features)}

## Temporal validation

Leave-one-survey-cycle-out validation was used. Every test participant was
scored by a model that did not train on their survey cycle.

| Metric | Result | Is higher better? |
|---|---:|---|
| AUROC | {metrics['auroc']:.3f} ({metrics['auroc_ci_95'][0]:.3f}-{metrics['auroc_ci_95'][1]:.3f}) | Yes; 0.5 is random ranking |
| AUPRC | {metrics['auprc']:.4f} ({metrics['auprc_ci_95'][0]:.4f}-{metrics['auprc_ci_95'][1]:.4f}) | Yes, but compare with prevalence |
| AUPRC lift | {metrics['auprc_lift']:.2f}x | Yes; 1x is prevalence baseline |
| Brier score | {metrics['brier_score']:.4f} | **Lower is better** |

Only {metrics['positives']} positive cases contribute to pooled out-of-cycle
validation. Results are exploratory and uncertain.

## Usage

```bash
pip install -r requirements.txt
python inference.py --input sample_input.json
```

Python:

```python
from inference import DiaPanRiskModel

model = DiaPanRiskModel()
result = model.predict([{{"Diabetes": 1, "DEMO_RIDAGEYR": 62,
                         "GHB_LBXGH": 7.1, "weight_loss_1yr_lb": 12}}])
```

## Benchmark contract

External models should use identical leave-one-cycle-out folds and return:

```text
global_participant_id,probability
1999-2000:12345,0.018
```

Evaluate with:

```bash
python benchmark_predictions.py \
  --predictions external_predictions.csv \
  --labels /path/to/nhanes_multicycle.csv
```

Compatible generic Hugging Face candidates include
[Prior-Labs TabPFN v2](https://hf.co/Prior-Labs/TabPFN-v2-clf) and
[AutoGluon TabPFN Mix](https://hf.co/autogluon/tabpfn-mix-1.0-classifier).
They must be retrained on the same DiaPan folds; their existing Hub metrics are
not directly comparable. Prior-Labs models use a custom license, while the
AutoGluon model is Apache-2.0.

## Limitations

- NHANES is repeated cross-sectional, not longitudinal.
- The target is self-reported prevalent disease, not future incident cancer.
- Positive cases are rare and confidence intervals are wide.
- Missing biomarkers are median-imputed; C-peptide is available only in early
  cycles.
- CA19-9 is unavailable.
- Diabetes subtype is an unvalidated proxy.
- Survey weights are not used for predictive fitting.

## Ethical use

Do not use this model to diagnose, exclude, prioritise treatment, or reassure
individual patients. Intended use is research benchmarking, reproducibility and
hypothesis generation.
"""
    (output / "README.md").write_text(model_card)
    project_root = dataset.parent.parent
    license_path = project_root / "LICENSE"
    if license_path.exists():
        shutil.copy(license_path, output / "LICENSE")
    (output / ".gitattributes").write_text(
        "*.joblib filter=lfs diff=lfs merge=lfs -text\n"
    )
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    path = export(
        root / "data" / "nhanes_multicycle.csv",
        root / "data" / "cycle_holdout_validation.json",
        root / "model_artifacts" / "huggingface" / "diapan-risk-xgboost",
    )
    print(f"Exported {path}")
