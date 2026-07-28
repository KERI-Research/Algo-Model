# Hugging Face Benchmarking Guide

## What constitutes a fair comparison

A Hugging Face model is comparable to DiaPan only if it:

1. Uses the same diabetic cohort and `PancreaticCancer` target.
2. Uses the same clinical-only input columns.
3. Learns preprocessing from training cycles only.
4. Uses identical leave-one-survey-cycle-out folds.
5. Reports AUROC, AUPRC, prevalence-relative lift and Brier score.

Hub download counts or metrics from unrelated datasets are not comparable.

## Candidate Hugging Face models

| Candidate | Hub task | License | Suitability |
| --- | --- | --- | --- |
| [Prior-Labs TabPFN 3](https://hf.co/Prior-Labs/tabpfn_3) | Tabular classification | Custom/other | Technically relevant; license must be reviewed before redistribution |
| [Prior-Labs TabPFN 2.5](https://hf.co/Prior-Labs/tabpfn_2_5) | Tabular classification | Custom/other | Technically relevant; requires identical DiaPan fold retraining |
| [Prior-Labs TabPFN v2 classifier](https://hf.co/Prior-Labs/TabPFN-v2-clf) | Tabular classification | Custom/other | Compatible architecture; not a pancreatic-specific model |
| [AutoGluon TabPFN Mix](https://hf.co/autogluon/tabpfn-mix-1.0-classifier) | Tabular classification | Apache-2.0 | Most straightforward public benchmark candidate |
| [TabPFN Oncology UQ](https://hf.co/RyeCatcher/tabpfn-oncology-uq) | Unspecified | Unspecified | Oncology label alone is insufficient; no direct comparability established |

No public Hub model found in the search was already trained for the exact DiaPan
task and feature schema. Generic models must therefore be fitted on DiaPan data,
not evaluated zero-shot using unrelated heads or output labels.

## DiaPan public-ready artifact

Path:

```text
model_artifacts/huggingface/diapan-risk-xgboost/
```

Contents:

| File | Purpose |
| --- | --- |
| `model.joblib` | Final clinical-only XGBoost model fitted on all labelled diabetics |
| `README.md` | Hugging Face model card |
| `config.json` | Task, license and benchmark metadata |
| `feature_schema.json` | Input names, meanings, direction and imputation |
| `inference.py` | Local prediction helper |
| `sample_input.json` | Example input |
| `benchmark_results.json` | Complete temporal fold report |
| `benchmark_predictions.py` | Model-agnostic metric evaluator |
| `requirements.txt` | Runtime dependencies |

The artifact is public-ready under the project's CC BY 4.0 license, but has not
been uploaded. Publication requires an explicit confirmation because it creates
public content.

## External-model prediction contract

An external model should produce:

```csv
global_participant_id,probability
1999-2000:12345,0.0184
2001-2002:45678,0.0021
```

The probability column may contain an uncalibrated ranking score as long as it
is numeric and higher means greater predicted risk. Brier score should only be
interpreted as calibration error when outputs are probability-like.

Evaluate:

```bash
cd model_artifacts/huggingface/diapan-risk-xgboost

python benchmark_predictions.py \
  --predictions external_predictions.csv \
  --labels ../../../data/nhanes_multicycle.csv
```

## Current local benchmark

| Model | AUROC | AUPRC | Lift | Brier |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | **0.643** | 0.0128 | 2.03× | 0.0123 |
| HistGradientBoosting | 0.635 | 0.0119 | 1.88× | 0.0159 |
| Random Forest | 0.629 | 0.0184 | 2.91× | **0.0073** |
| Balanced Logistic Regression | 0.601 | **0.0347** | **5.49×** | 0.1996 |

XGBoost is exported because it has the strongest temporal AUROC. Random Forest
is a credible secondary benchmark. Logistic Regression's poor Brier score
shows that its high AUPRC does not translate into trustworthy probabilities.

## Recommended Hugging Face experiment

1. Start with AutoGluon TabPFN Mix because its Apache-2.0 license is explicit.
2. Retrain separately inside each DiaPan temporal fold.
3. Export one prediction per held-out participant.
4. Evaluate with `benchmark_predictions.py`.
5. Compare confidence intervals and per-cycle variation, not only pooled
   metrics.
6. Document computational limits and any reduced training subset. TabPFN-style
   models may impose sample/feature constraints that change comparability.
