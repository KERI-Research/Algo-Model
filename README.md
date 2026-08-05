# MetaboGuard

**Self-supervised metabolic early-warning research for cancer and diabetes
prevention.**

Developed within the **KERI department**.

## Quick start (validated 2026-08-04)

```bash
cd api
python data_integrity.py --dataset ../data/nhanes_multicycle_v2.csv   # fail-closed validation
python run_meeting_demo.py                                            # validate + smoke train + baselines (~25s)
python run_meeting_demo.py --full                                     # full current SSL configuration (~30s)
python -m unittest discover -p "test_*.py"                            # safety + pipeline tests
cd ../frontend && npm run build                                       # production build
```

See [`TODAY_MEETING_DEMO.md`](TODAY_MEETING_DEMO.md) for the demonstration runbook,
verified numbers and the sentences that are safe to say about them.

A complete verified full run (40 epochs, NumPy backend) is stored at
[`model_artifacts/metaboguard_ssl/meeting_2026-08-04/`](model_artifacts/metaboguard_ssl/meeting_2026-08-04/)
with weights, training-only preprocessor, split indices, run manifest, metrics, model card
and benchmark report. It is deliberately **not promoted** for API serving and is not a
clinical or future-risk model. PyTorch is not installed, so torch-backend parity is
unverified.

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

Every entry point validates the dataset first and refuses invalidated files,
denylisted inputs and ungated future-risk heads. Full guide:
[`docs/TRAINING.md`](docs/TRAINING.md).

```bash
cd api

# bounded smoke run (minutes, CPU, PyTorch optional)
python train_self_supervised.py --smoke

# full run from the committed configuration
python train_self_supervised.py --config configs/ssl_full.json

# promote a full run for API serving (smoke runs are refused)
python train_self_supervised.py --config configs/ssl_full.json --promote
```

Artifacts are written to
`model_artifacts/metaboguard_ssl/runs/<dataset>__<run_label>__<UTC timestamp>/`
with weights, the training-only preprocessor, `metadata.json` (config, split policy,
run manifest with seed/backend/device/package versions/timings), a generated
`MODEL_CARD.md`, `splits.npz` and a resumable checkpoint.

Backends: `torch` when installed, otherwise a deterministic NumPy implementation of
the same architecture (`--backend numpy`). Default device is CPU for reproducibility;
`--device mps` is available on the torch backend and is not bit-for-bit reproducible.

## Research pass: reliability, phenotypes, evidence

Following the 2026-08-04 supervisor feedback
([decision record](docs/decisions/2026-08-04-professor-feedback.md)):

```bash
cd api

# everything at once: integrity -> reliability -> evidence -> clustering x2 -> charts
../.venv/bin/python run_research_pass.py            # ~1-3 min
../.venv/bin/python run_research_pass.py --quick     # smaller grid, ~1 min

# individual steps
../.venv/bin/python data_reliability.py --output ../model_artifacts/reports/data_reliability.json
../.venv/bin/python evidence_catalogue.py --strict
../.venv/bin/python clustering.py --complete-cases-only
```

- **Data reliability** (`api/data_reliability.py`) grades every model input into
  `usable_now`, `qualified_use`, `unavailable` or `prohibited`, and fails closed on unit,
  leakage or schema violations. Current file: 17 usable now, 8 qualified use, 16 prohibited.
- **Exploratory phenotype clustering** (`api/clustering.py`) is label-free in fit and
  selection, gated on stability, a permuted null, outlier sensitivity and mandatory negative
  controls, and **abstains** when nothing passes. Current result: `no_stable_clusters`
  because the recoverable structure is dominated by survey cycle. See
  [`docs/CLUSTERING.md`](docs/CLUSTERING.md).
- **Evidence catalogue** (`data/evidence/biomarker_evidence.json`) records 20 rows with
  mandatory provenance, plus the allowed/denied statement lists and the claims contract
  (PRoBE, TRIPOD+AI, PROBAST+AI, STARD). See
  [`docs/EVIDENCE_AND_CLAIMS.md`](docs/EVIDENCE_AND_CLAIMS.md).

Clusters are patient/metabolic phenotypes — never a cancer diagnosis, subtype or site.
Early detection is treated as a **panel and feature-interaction** problem; the claim that
cancers have no specific biomarkers is false and is not made anywhere in this project.

## Benchmarks

```bash
cd api
python baselines.py --dataset ../data/nhanes_multicycle_v2.csv \
  --ssl-artifact ../model_artifacts/metaboguard_ssl/nhanes_multicycle_v2
```

PCA reconstruction (components matched to the latent dimension) and Isolation Forest,
evaluated under the identical preprocessing and split boundaries. Unsupervised
deviation only — no disease prediction. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

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
- `POST /api/v1/prevention-capabilities` — capability + horizon gate report
- `POST /api/v1/prevention-score` — deviation score, percentile, latent representation
- `POST /api/v1/data-integrity` — coding, leakage, duplicate, missingness and split report
- `POST /api/v1/prevention-future-risk` — intentionally fail-closed (HTTP 409) until gates pass
- `GET /api/v1/future-risk-capability` — what future-risk output is possible right now and why:
  capability state, the 50-event gate, permanently disabled outcomes, and whether a
  simulation-only artefact exists. Real patient future risk is always reported as disabled.
- `POST /api/v1/simulation/future-risk-score` — **simulation only**. Requires an explicit
  `simulation_mode=true` flag plus a simulation-only artefact, refuses cross-sectional payloads
  (HTTP 422) and single-visit histories, and returns raw plus calibrated cumulative incidence per
  horizon with an abstention wherever a horizon failed its gate. See
  `docs/FUTURE_RISK_PROTOCOL.md`, `docs/LONGITUDINAL_SCHEMA.md`, `docs/FUTURE_RISK_EVALUATION.md`
  and `docs/SYNTHEA_GENERATION.md`.
 — what future-risk output is possible right now and why:
  capability state, the 50-event gate, permanently disabled outcomes, and whether a
  simulation-only artefact exists. Real patient future risk is always reported as disabled.
- `POST /api/v1/simulation/future-risk-score` — **simulation only**. Requires an explicit
  `simulation_mode=true` flag plus a simulation-only artefact, refuses cross-sectional payloads
  (HTTP 422) and single-visit histories, and returns raw plus calibrated cumulative incidence per
  horizon with an abstention wherever a horizon failed its gate. See
  `docs/FUTURE_RISK_PROTOCOL.md`, `docs/LONGITUDINAL_SCHEMA.md`, `docs/FUTURE_RISK_EVALUATION.md`
  and `docs/SYNTHEA_GENERATION.md`.
 — what future-risk output is possible right now and why:
  capability state, the 50-event gate, permanently disabled outcomes, and whether a
  simulation-only artefact exists. Real patient future risk is always reported as disabled.
- `POST /api/v1/simulation/future-risk-score` — **simulation only**. Requires an explicit
  `simulation_mode=true` flag plus a simulation-only artefact, refuses cross-sectional payloads
  (HTTP 422) and single-visit histories, and returns raw plus calibrated cumulative incidence per
  horizon with an abstention wherever a horizon failed its gate. See
  `docs/FUTURE_RISK_PROTOCOL.md`, `docs/LONGITUDINAL_SCHEMA.md`, `docs/FUTURE_RISK_EVALUATION.md`
  and `docs/SYNTHEA_GENERATION.md`.
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
- Fail-closed validation on every training and scoring entry point (`api/data_integrity.py`)
- Invalidated datasets and pancreatic-cancer targets blocked on both train and load paths
- Participant-grouped, seeded splits with training-only preprocessing and deviation reference
- Run manifests (seed, backend, device, package versions, timings) and generated model cards
- Automated tests for coding, leakage, gates, split boundaries and API terminology

## Documentation

- [`TODAY_MEETING_DEMO.md`](TODAY_MEETING_DEMO.md) — demonstration runbook and verified numbers
- [`docs/TRAINING.md`](docs/TRAINING.md) — training, configs, backends, reproducibility rules
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — PCA / Isolation Forest baselines and deferred comparisons
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — safety card, gates, limitations, blocker
- [`docs/CLUSTERING.md`](docs/CLUSTERING.md) — exploratory phenotypes, gates, negative controls, abstain
- [`docs/EVIDENCE_AND_CLAIMS.md`](docs/EVIDENCE_AND_CLAIMS.md) — evidence provenance, allowed/denied statements, claims contract
- [`docs/decisions/2026-08-04-professor-feedback.md`](docs/decisions/2026-08-04-professor-feedback.md) — dated decision record (recollected notes)
- [`docs/PREVENTION_MODEL_SPEC.md`](docs/PREVENTION_MODEL_SPEC.md)
- [`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md)
- [`docs/COLUMN_DICTIONARY.md`](docs/COLUMN_DICTIONARY.md)
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- [`docs/PAPER_VARIABLE_CATALOGUE.md`](docs/PAPER_VARIABLE_CATALOGUE.md)
- [`docs/RESEARCH_EVIDENCE.md`](docs/RESEARCH_EVIDENCE.md)
- [`docs/HUGGINGFACE_BENCHMARKING.md`](docs/HUGGINGFACE_BENCHMARKING.md)
- [`docs/CYCLE_HOLDOUT_VALIDATION.md`](docs/CYCLE_HOLDOUT_VALIDATION.md)

## Research sources

- [Zhou et al. 2025: genetic and clinical NODM-PC model](https://doi.org/10.1186/s12916-025-04048-4)
- [Yang et al. 2026: clinical CatBoost and multi-omics integration](https://doi.org/10.1186/s12967-026-07767-1)
