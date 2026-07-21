# KERI

KERI is a research-oriented web application for exploring the diabetes-cancer relationship using:

- a causal pipeline based on a directed acyclic graph (DAG) and DoWhy
- a predictive baseline model for comparison
- NHANES merged tabular data

The project has a Python FastAPI backend and a React frontend.

## What This Project Does

KERI provides four core capabilities:

1. Dataset discovery: list available CSV datasets.
2. Dataset preview: inspect columns and sample rows before analysis.
3. Causal analysis: estimate directional effects (for example Diabetes -> Cancer) with refutation checks.
4. Predictive baseline: fit a simple logistic baseline and report imbalance-aware metrics.
5. Biomarker discovery: train a local tabular model, rank cohort-level biomarkers, benchmark HistGradientBoosting against XGBoost, score patient-style records, and ask for missing required fields when confidence is too low or records are incomplete.

## Project Structure

```text
KERI/
 api/
  main.py                # FastAPI routes
  engine.py              # DAG + causal inference execution
  predictive.py          # Predictive baseline model
  biomarker.py           # Biomarker discovery, local artifact loading, ChromaDB retrieval
  train_biomarker_model.py # Offline training CLI for the biomarker model
  fetch_nhanes.py        # NHANES download/merge/build pipeline
  nhanes_data/
   nhanes_merged.csv
 data/
  nhanes_merged.csv
 frontend/
  src/
  package.json
 resources/
  Model Understanding.tex
 requirements.txt
```

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

The causal engine prepares a minimal model dataframe with:

- Obesity
- Diabetes
- Cancer

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
