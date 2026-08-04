# MetaboGuard

**Self-supervised metabolic early-warning research for cancer and diabetes
prevention.**

Developed within the **KERI department**.

## Intended use

MetaboGuard learns common metabolic patterns from clinical and behavioural
data, then identifies patient profiles that differ from the training reference.
Its intended role is to support clinician-reviewed monitoring and prevention
research.

MetaboGuard is intended to help answer:

- Is this metabolic profile unusually different from comparable records?
- Which measurements contribute most to that deviation?
- Does the learned representation contain cross-sectional signal associated
  with cancer or diabetes?
- Which follow-up measurements might a clinician consider reviewing?

## Not intended for

MetaboGuard must not:

- diagnose cancer or diabetes;
- estimate a future disease probability from cross-sectional NHANES data;
- recommend treatment;
- reassure a patient that disease is absent;
- present Type 1 diabetes risk without validated autoimmune biomarkers;
- replace clinical judgement, screening guidelines or confirmatory testing.

## Architecture

### Self-supervised representation

`metaboguard_ssl_v1` is a denoising tabular autoencoder trained without disease
labels. It uses masked-feature reconstruction to learn a 16-dimensional latent
representation from metabolic, demographic, behavioural, CBC and biochemistry
variables.

Outputs:

- `metabolic_deviation_score`
- reference percentile
- 16-dimensional latent representation
- five highest reconstruction-deviation features

A higher deviation score means “more unusual relative to the training
reference.” It does not mean higher validated cancer or diabetes probability.

### Post-hoc research heads

Frozen embeddings are evaluated against:

- any-cancer prevalence;
- Type 2 diabetes proxy;
- Type 1 diabetes proxy, research-only.

Labels are used only after representation training. Current checks are
cross-sectional association tests and do not measure future disease
development.

### Future longitudinal risk heads

When patient-level longitudinal outcomes become available, MetaboGuard will
select supported horizons based on available event counts. Candidate horizons
are 1, 3 and 5 years, but a horizon is enabled only when it contains at least
50 events and 50 eligible non-events.

## Current artifact

| Property | Value |
| --- | ---: |
| Dataset | `nhanes_multicycle_v2.csv` |
| Adult rows available | 89,472 |
| Unlabelled training rows | 50,000 |
| Raw input features | 25 |
| Transformed dimensions | 55 |
| Latent dimensions | 16 |
| Training epochs | 25 |
| Validation reconstruction loss | 0.0565 |

Cross-sectional association checks:

| Check | Positives | AUROC | AUPRC | Warning |
| --- | ---: | ---: | ---: | --- |
| Any-cancer prevalence | 5,584 | 0.699 | 0.169 | Not future cancer development |
| Type 2 diabetes proxy | 7,067 | 0.923 | 0.675 | Not future diabetes development |
| Type 1 proxy | 177 | 0.909 | 0.135 | Unvalidated, research-only proxy |

Diagnosis age and insulin-use variables that define the Type 1 proxy were
excluded from the encoder to prevent direct label leakage.

## Dataset roles

### NHANES v2

Used for self-supervised metabolic representation and cross-sectional
association analysis. The pooled dataset includes:

- glucose, HbA1c, insulin and partial-cycle C-peptide;
- lipids and hs-CRP;
- BMI, waist and weight-change proxies;
- smoking and alcohol;
- haemoglobin and platelets;
- ALT, alkaline phosphatase and creatinine.

NHANES is repeated cross-sectional data. It cannot establish individual
biomarker trajectories or future disease development.

### TCGA-CDR

Used only for cancer-context and prognosis research among already diagnosed
patients. Tumour stage, treatment response and survival fields are prohibited
from preventive early-warning scoring.

### Future linked cohort

Development-risk validation requires:

- repeated patient measurements;
- exact diabetes and cancer diagnosis dates;
- incident disease outcomes;
- sufficient events at each proposed horizon;
- Type 1 autoantibodies and C-peptide;
- genetics only after appropriate approval.

## Type 1 diabetes policy

Type 1 output is research-only until the project has:

- islet autoantibodies such as GAD, IA-2, ZnT8 or IAA;
- clinically appropriate C-peptide;
- family history;
- longitudinal glucose/HbA1c;
- approved genetic-risk inputs where permitted.

The current age-at-diagnosis/insulin-use subtype is an exploratory proxy and
must never be shown as a patient-facing Type 1 warning.

## Corrected cancer outcome

Official NHANES MCQ230 coding defines:

- code 29: Pancreas
- code 39: Other

The corrected pancreatic cohort contains only 19 cases overall, 7 among
diabetics and 2 meeting a three-year NODM-PC definition. Supervised
pancreatic-risk training is blocked below 20 usable positives. The earlier
DiaPan-XGB artifact is invalidated.

## Installation

Core API and NumPy inference:

```bash
pip install -r requirements.txt
```

Self-supervised training:

```bash
pip install -r requirements-ssl.txt
```

## Training

```bash
cd api

python train_self_supervised.py \
  --dataset ../data/nhanes_multicycle_v2.csv \
  --epochs 25 \
  --latent-dim 16
```

## Scoring

```bash
python score_prevention_record.py \
  --artifact ../model_artifacts/metaboguard_ssl/nhanes_multicycle_v2 \
  --input ../metaboguard_sample_input.json
```

## API

```bash
uvicorn main:app --reload --port 8000
```

Relevant endpoints:

- `GET /api/v1/datasets`
- `POST /api/v1/prevention-capabilities`
- `POST /api/v1/prevention-score`
- `POST /api/v1/analyze`
- `POST /api/v1/predictive-baseline`
- `POST /api/v1/biomarker-discovery`

The prevention endpoint returns a deviation score and clinical warning. It does
not provide a diagnosis or future-risk claim.

## Research safeguards

- Prevention-safe feature allowlist
- Explicit post-diagnosis feature denylist
- Dataset capability detection
- Adaptive horizon eligibility checks
- Training-only preprocessing for supervised validation
- Minimum event thresholds
- Dataset fingerprints for supervised artifacts
- Public-export blocking for invalid/underpowered models
- Research-only Type 1 proxy

## Documentation

- [Documentation Hub](index.html)
- [PREVENTION_MODEL_SPEC](PREVENTION_MODEL_SPEC.html)
- [RESULTS_GUIDE](RESULTS_GUIDE.html)
- [COLUMN_DICTIONARY](COLUMN_DICTIONARY.html)
- [METHODOLOGY](METHODOLOGY.html)
- [PAPER_VARIABLE_CATALOGUE](PAPER_VARIABLE_CATALOGUE.html)
- [TRAINING](TRAINING.html)
- [BENCHMARKS](BENCHMARKS.html)
- [MODEL_CARD](MODEL_CARD.html)
- [RESEARCH_EVIDENCE](RESEARCH_EVIDENCE.html)
- [HUGGINGFACE_BENCHMARKING](HUGGINGFACE_BENCHMARKING.html)
- [CYCLE_HOLDOUT_VALIDATION](CYCLE_HOLDOUT_VALIDATION.html)

## Research sources

- [Zhou et al. 2025: genetic and clinical NODM-PC model](https://doi.org/10.1186/s12916-025-04048-4)
- [Yang et al. 2026: clinical CatBoost and multi-omics integration](https://doi.org/10.1186/s12967-026-07767-1)
