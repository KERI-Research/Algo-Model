# DiaPan

**AI-Driven Discovery of Metabolic Biomarkers Linking Diabetes and Pancreatic
Cancer for Early Detection.**

Developed within the **KERI department**.

DiaPan investigates whether combinations of clinical, biochemical and behavioural
data can surface early warning patterns for pancreatic cancer inside the large
population of patients with diabetes. Rather than relying on any single
biomarker, the pipeline builds risk-stratification models across multiple
variables so that subtle metabolic trajectories become tractable.

The project has a Python FastAPI backend and a React frontend, and is backed
by two open real-world datasets:

- **NHANES 1999–March 2020** (pooled population surveys — diabetes prevalence, HbA1c,
  glucose, insulin, lipids, hs-CRP, self-reported cancer history including
  pancreatic-cancer site code, self-reported weight history)
- **TCGA-CDR** (11,160 real cancer patients across 33 types, including 133
  pancreatic adenocarcinoma cases)

## Research goals from the brief

1. **Detect subtle metabolic signals** of pancreatic cancer that emerge from
   combinations of variables, not isolated markers.
2. **Stratify risk within diabetics** — which diabetic patients are the
   highest-risk for pancreatic cancer?
3. **Turn insights into practical tooling** — an API endpoint that scores a
   patient record and returns explainable biomarker rankings.

## Variables covered per the brief

| Brief-required variable | Where in DiaPan | Notes |
|---|---|---|
| Type of diabetes (T1 / T2 / gestational) | `diabetes_subtype` (derived) | Proxy from age-at-diagnosis and insulin use; NHANES does not directly ask T1 vs T2 |
| Timing of onset (recent vs long-standing) | `recent_diabetes_onset` | 1 if diagnosed within last 3 years |
| Duration since diagnosis | `diabetes_duration_years` | Current age − DIQ_DID040 |
| HbA1c levels | `GHB_LBXGH`, `elevated_hba1c` | HbA1c ≥ 6.5 flagged |
| C-peptide | `CPEP_LBXCPSI` | Measured in 1999-2004 fasting files only; 9,501 pooled non-null records |
| Insulin | `INS_LBXIN`, `homa_ir` | HOMA-IR = glucose × insulin / 405 |
| CA 19-9 | **Not available** | NHANES does not carry CA 19-9; would require MIMIC/UK Biobank |
| Age | `DEMO_RIDAGEYR` | |
| Sex | `DEMO_RIAGENDR` | |
| Weight loss | `weight_loss_1yr_lb`, `significant_weight_loss_flag` (≥10 lb), `weight_loss_10yr_lb` | Derived from NHANES WHQ WHD020 / WHD050 / WHD140 |
| Obesity | `Obesity`, `BMX_BMXBMI`, `BMX_BMXWAIST` | BMI ≥ 30 |

## What This Project Does

1. **Dataset discovery**: list available CSV datasets.
2. **Dataset preview**: inspect columns and sample rows before analysis.
3. **Causal analysis** (`/api/v1/analyze`): estimate directional effects (e.g.
   Diabetes → Cancer) with DoWhy refutation checks.
4. **Predictive baseline** (`/api/v1/predictive-baseline`): logistic-regression
   baseline with imbalance-aware metrics.
5. **Biomarker discovery** (`/api/v1/biomarker-discovery`): train a local
   HistGradientBoosting / XGBoost model against a chosen target (`Cancer`,
   `PancreaticCancer`, or TCGA `Progression`), optionally restrict to a
   sub-cohort (`cohort_filter=diabetics_only`), rank cohort-level biomarkers,
   score patient records, and request missing fields when confidence is low.

## Pancreatic-cancer-in-diabetics risk stratification (brief’s core question)

The target `PancreaticCancer` is derived from NHANES `MCQ_MCQ230A/B/C/D`
(self-reported cancer site, code 39 = pancreas). Two models are supported:

```bash
cd api
# 1) Screening in the general population — baseline, expected to be weak
python train_biomarker_model.py --dataset nhanes_multicycle.csv --force \
    --target PancreaticCancer

# 2) Risk stratification within diabetic patients (the brief's key ask)
python train_biomarker_model.py --dataset nhanes_multicycle.csv --force \
    --target PancreaticCancer --cohort-filter diabetics_only
```

Current pooled NHANES 1999-March 2020 benchmarks:

| Cohort | Rows used | Test positives | AUROC (95% CI) | AUPRC (95% CI) | AUPRC lift |
|---|---:|---:|---:|---:|---:|
| General population | 52,891 | 55 | 0.679 (0.598-0.748) | 0.011 (0.008-0.020) | 2.14x |
| **Diabetics only** | **6,473** | **8** | **0.641 (0.419-0.830)** | **0.135 (0.007-0.386)** | **21.81x** |

Pooling expands the cohort to 107,622 participants, 318 pancreatic-cancer
positives overall and 50 among labelled diabetics. The diabetics-only model
shows a 21.81x AUPRC lift over prevalence, but its confidence intervals remain
wide because the held-out fold contains only eight positives. This is promising
enrichment with high uncertainty, not validated early detection.

### Temporal generalisation

Random splits mix participants from every survey era and can overstate
performance. `api/validate_cycle_holdout.py` therefore holds out each NHANES
cycle in turn and learns imputation statistics only from the remaining cycles.

The preferred clinical-only XGBoost benchmark achieves out-of-cycle AUROC
**0.643** (cycle-bootstrap 95% CI 0.561-0.703), AUPRC **0.0128**
(0.0084-0.0289) and 2.03x lift over prevalence. This is substantially weaker
than the random-split AUPRC of 0.135, so the random-split result is treated as
optimistic. See `docs/CYCLE_HOLDOUT_VALIDATION.md` for every fold.

Top biomarkers in the diabetics-only model (per permutation importance,
five-figure precision):

- `waist_bmi_interaction` (central adiposity)
- `weight_loss_1yr_lb` (lower in positives — hint of the brief's weight-loss
  red flag)
- `TCHOL_LBXTC` (total cholesterol)
- `DEMO_RIDAGEYR` (age)
- `DEMO_RIDRETH3` (ethnicity)

## Prognosis in confirmed pancreatic cancer (TCGA-PAAD)

TCGA-CDR contains 133 pancreatic adenocarcinoma (PAAD) patients. 5-year
mortality is 92 % — too imbalanced for a standalone PAAD-only classifier
with honest metrics. PAAD is instead included in the multi-cancer TCGA
models (`--target Cancer` and `--target Progression`), where the
`tcga_type_PAAD` one-hot flag ranks among the top biomarkers.

## Honest limitations

- **No true patient trajectories.** Multi-cycle NHANES is implemented, but it
  remains repeated cross-sectional data. Engineered HbA1c/cycle interactions
  are cohort-level proxies, not within-patient changes.
- **CA 19-9 is absent.** C-peptide is available only in 1999-2004 fasting
  files. A longitudinal clinical cohort is still required for serial CA19-9
  and metabolic lead-time curves.
- **Diabetes subtype is a proxy.** NHANES doesn't ask T1 vs T2 directly.
- **Rare positive class.** Pooling yields 318 pancreatic-cancer positives, but
  only eight positives appear in the diabetics-only held-out fold.

Full evidence and decision rationale:

- [`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md) — plain-English metric
  meanings, “higher/lower is better”, diagrams and model comparison
- [`docs/COLUMN_DICTIONARY.md`](docs/COLUMN_DICTIONARY.md) — raw and derived
  column meanings, units and direction
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — implemented design, weighting,
  harmonisation, evaluation and results
- [`docs/RESEARCH_EVIDENCE.md`](docs/RESEARCH_EVIDENCE.md) — source-cited
  literature review and decision-to-evidence table
- [`docs/CYCLE_HOLDOUT_VALIDATION.md`](docs/CYCLE_HOLDOUT_VALIDATION.md) —
  temporal transportability results for every survey cycle
- [`docs/HUGGINGFACE_BENCHMARKING.md`](docs/HUGGINGFACE_BENCHMARKING.md) —
  compatible Hub candidates and fair-comparison contract
- `data/nhanes_multicycle_build_report.json` — machine-readable source manifest,
  coverage audit and weight formula
- `data/cycle_holdout_validation.json` — machine-readable fold predictions and
  pooled out-of-cycle metrics

Public-ready model artifact:

```text
model_artifacts/huggingface/diapan-risk-xgboost/
```

The folder contains the final XGBoost model, model card, feature schema,
inference helper, sample input and model-agnostic benchmarking script. It is
ready to upload to Hugging Face but has not been published.

## Project Structure

```text
DiaPan/
 api/
  main.py                # FastAPI routes
  engine.py              # DAG + causal inference execution
  predictive.py          # Predictive baseline model
  biomarker.py           # Biomarker discovery, local artifact loading, ChromaDB retrieval
  train_biomarker_model.py # Offline training CLI for the biomarker model
  fetch_nhanes.py        # NHANES download/merge/build pipeline
  fetch_tcga.py          # TCGA-CDR download and harmonization
  nhanes_data/
   nhanes_merged.csv
   tcga_cdr.csv
 data/
  nhanes_merged.csv
  tcga_cdr.csv
 frontend/
  src/
  package.json
 resources/
  Model Understanding.tex
 requirements.txt
```

## Data Pipeline (TCGA-CDR)

The TCGA-CDR builder pulls the TCGA Pan-Cancer Clinical Data Resource
(Liu et al., *Cell* 2018) supplemental table from the NCI GDC and reshapes it
into a DiaPan-compatible CSV keyed on a synthetic SEQN column.

Source: `TCGA-CDR-SupplementalTableS1.xlsx` from
`https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81` — 11,160
patients across 33 cancer types, open access, no login or DUA required.

Two prediction targets are derived from the CDR survival columns:

- `Cancer` → **5-year all-cause mortality**: `1` if the patient died within 5
  years of initial pathologic diagnosis, `0` if alive with at least 5 years of
  follow-up, dropped otherwise. 4,996 labelled patients (3,193 events / 1,803
  survivors).
- `Progression` → **5-year progression-free-interval event** (built from
  `PFI` and `PFI.time` in the same way). 5,120 labelled patients
  (3,779 events / 1,341 progression-free).

Harmonized feature columns (TCGA → NHANES-shaped where possible):

- `age_at_initial_pathologic_diagnosis` → `DEMO_RIDAGEYR`
- `gender` (MALE/FEMALE) → `DEMO_RIAGENDR` (1/2)
- `race` → `DEMO_RIDRETH3` (integer coded)
- `ajcc_pathologic_tumor_stage` → `tcga_stage_ordinal` (0..4)
- `histological_grade` → `tcga_grade_ordinal` (1..4, high/low collapsed)
- `tumor_status` → `tcga_tumor_status` (1 with tumor, 0 tumor free)
- `treatment_outcome_first_course` → `tcga_treatment_response`
  (0 progressive … 3 complete remission) — **post-treatment**, so downstream
  users should treat model output as "prognostic given first-line response"
  rather than pre-treatment prediction
- `type` (cancer type) → `tcga_cancer_type` + 33 one-hot `tcga_type_*` flags
- `OS`/`OS.time`/`PFI`/`PFI.time` → `tcga_event`, `tcga_followup_days`,
  `tcga_pfi_event`, `tcga_pfi_days` **(metadata only, deliberately excluded
  from model features to prevent label leakage)**

Build the dataset:

```bash
python api/fetch_tcga.py
```

Train the biomarker model:

```bash
cd api
# 5-year mortality (default target)
python train_biomarker_model.py --dataset tcga_cdr.csv --force
# 5-year disease progression
python train_biomarker_model.py --dataset tcga_cdr.csv --force --target Progression
```

Artifacts land in `api/model_artifacts/tcga_cdr/` (mortality) and
`api/model_artifacts/tcga_cdr_progression/` (progression) so both can coexist.

Current TCGA-CDR benchmarks (XGBoost vs HistGradientBoosting):

| Target | Rows | Positive rate | AUROC | AUPRC |
|---|---|---|---|---|
| 5-year all-cause mortality (`Cancer`) | 4,887 | 63.9 % | **0.894** | **0.936** |
| 5-year progression (`Progression`) | 5,025 | 73.9 % | **0.912** | **0.966** |

Top biomarkers on both targets: `tcga_tumor_status`, `tcga_treatment_response`,
`tcga_type_GBM` (glioblastoma), `DEMO_RIDAGEYR` (age), `tcga_stage_ordinal`.

## Data Pipeline (NHANES)

The NHANES builder script pulls 2017-2018 public files and merges them by participant key SEQN:

- DEMO (demographics)
- BMX (body measurements)
- DIQ (diabetes questionnaire)
- MCQ (medical conditions questionnaire)
- GHB (glycohemoglobin / HbA1c)
- GLU (fasting glucose)
- INS (fasting insulin)
- TRIGLY (triglycerides and LDL)
- HDL (HDL cholesterol)
- TCHOL (total cholesterol)
- HSCRP (high-sensitivity C-reactive protein)

The merged file is written to both:

- data/nhanes_merged.csv
- api/nhanes_data/nhanes_merged.csv

The pipeline also derives model-ready binary columns when possible:

- Obesity from BMX_BMXBMI (BMI >= 30)
- Diabetes from DIQ_DIQ010
- Cancer from MCQ_MCQ220
- HOMA-IR proxy from fasting glucose and insulin
- Elevated HbA1c flag from GHB_LBXGH
- Fasting hyperglycemia flag from GLU_LBXGLU

## Causal Model Summary

The causal engine is designed for the NHANES dataset. It prepares a minimal
model dataframe with:

- Obesity
- Diabetes
- Cancer

On TCGA-CDR the causal analysis is not applicable (no diabetes labels); the
`/api/v1/analyze` endpoint will return a clear error if invoked against
`tcga_cdr.csv`. Use TCGA-CDR with `/api/v1/biomarker-discovery` and
`/api/v1/predictive-baseline` instead.

It then evaluates a DAG with these structural assumptions:

- Obesity -> Diabetes
- Obesity -> Cancer
- treatment -> outcome (dynamic, based on request)

Default request direction is Diabetes -> Cancer.

Estimation path:

1. identify effect using DoWhy
2. estimate with backdoor linear regression
3. run refutations:

- random common cause
- placebo treatment refuter
- data subset refuter

If DoWhy runtime compatibility fails and fallback is allowed, the API returns an association-style fallback estimate with warnings.

## Predictive Baseline Summary

The predictive module builds two binary classifiers (one for Diabetes, one for Cancer) using a lightweight custom logistic regression implementation.

It uses:

- fixed stratified 80/20 split
- standardization based on training set
- gradient descent optimization with L2 regularization

Reported metrics include:

- AUROC
- AUPRC
- recall
- precision
- F1
- balanced accuracy
- specificity
- Brier score

This predictive baseline is explicitly non-causal and is intended as a benchmark.

## Backend Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run backend API:

```bash
cd api
uvicorn main:app --reload
```

Backend default URL:

- <http://localhost:8000>

## Frontend Setup

In a second terminal:

```bash
cd frontend
npm install
npm start
```

Frontend default URL (React dev server):

- <http://localhost:3000>

Optional API URL override for frontend:

```bash
export REACT_APP_API_URL=http://localhost:8000
```

## API Endpoints

### GET /api/v1/datasets

Lists discovered CSV datasets.

### POST /api/v1/dataset-preview

Request body:

```json
{
 "dataset": "nhanes_merged.csv"
}
```

Returns dataset name, column list, and a sample preview.

### POST /api/v1/analyze

Request body:

```json
{
 "dataset": "nhanes_merged.csv",
 "treatment": "Diabetes",
 "outcome": "Cancer",
 "allow_fallback": true
}
```

Returns:

- estimate
- refutation estimates
- execution mode (dowhy_causal or fallback_association)
- warnings and error context (if any)

### POST /api/v1/predictive-baseline

Request body:

```json
{
 "dataset": "nhanes_merged.csv"
}
```

Returns per-target baseline metrics for Diabetes and Cancer.

### POST /api/v1/biomarker-discovery

Request body:

```json
{
 "dataset": "nhanes_merged.csv",
 "patient_record": {
  "Diabetes": 1,
  "DEMO_RIDAGEYR": 62,
  "DEMO_RIAGENDR": 2,
  "BMX_BMXBMI": 31.4,
  "BMX_BMXWAIST": 101.2,
  "DIQ_DID040": 56
 },
 "top_k": 8,
 "force_retrain": false
}
```

Returns:

- cohort-level biomarker ranking
- biomarker model metrics and artifact metadata
- ChromaDB memory summary
- patient assessment with confidence and follow-up questions when required fields are missing or confidence is low

## Rebuild NHANES Data

To regenerate the merged dataset manually:

```bash
source .venv/bin/activate
python api/fetch_nhanes.py
```

The backend can also auto-build nhanes_merged.csv if it is missing.

## Train The Biomarker Model

To create or refresh the local biomarker artifact:

```bash
source .venv/bin/activate
python api/train_biomarker_model.py --force
```

This command:

- loads NHANES
- applies the same feature and cleaning logic used at runtime
- benchmarks HistGradientBoosting against XGBoost while preserving the artifact contract
- trains the winning local tabular biomarker model
- writes a versioned artifact plus ChromaDB retrieval memory under api/model_artifacts/

The command output now includes both the selected model metrics and the benchmark summary.

## Common Workflow

1. Start backend.
2. Start frontend.
3. Open the UI.
4. Choose dataset and preview rows.
5. Run causal analysis and review estimate plus refutations.
6. Run the biomarker model with a patient-style record and review its biomarker ranking, confidence, and follow-up questions.
7. Compare with predictive baseline metrics.

## Notes and Interpretation

- Causal output quality depends on the DAG assumptions and data quality.
- Fallback mode is useful for availability but should be interpreted as descriptive association, not a fully identified causal effect.
- Predictive performance does not imply causality.

## Troubleshooting

Backend import/runtime issues:

- Ensure the virtual environment is active.
- Reinstall requirements:

```bash
pip install -r requirements.txt --upgrade
```

Frontend cannot reach backend:

- Confirm backend is running on port 8000.
- Set REACT_APP_API_URL explicitly before npm start.

Missing dataset errors:

- Ensure nhanes_merged.csv exists under data/ or api/nhanes_data/.
- Rebuild with python api/fetch_nhanes.py.
