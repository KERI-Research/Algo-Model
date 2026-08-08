# MetaboGuard Training Guide

All commands are run from the `api/` directory. `../.venv/bin/python` is the
project interpreter on the development laptop.

## 0. Dependencies

```bash
pip install -r requirements.txt        # API + NumPy inference + baselines
pip install -r requirements-ssl.txt    # adds PyTorch for the torch backend
pip install -r requirements-dev.txt    # optional: pytest, ruff, mypy
```

PyTorch is **optional**. `self_supervised.py` ships two interchangeable backends:

| Backend | When it is used | Reproducibility |
| --- | --- | --- |
| `torch` | selected automatically when PyTorch imports | seeded generators, CPU deterministic |
| `numpy` | fallback when PyTorch is absent, or `--backend numpy` | fully deterministic Adam implementation |

Both backends export the same weight names (`enc1.weight`, …, `dec_out.bias`), so
`NumpyAutoencoder` scores artifacts from either one. Apple Silicon note: the
default device is **cpu** for reproducibility. `--device mps` is accepted on the
torch backend and falls back to CPU when MPS is unavailable; MPS results are not
bit-for-bit reproducible ([PyTorch randomness notes](https://pytorch.org/docs/stable/notes/randomness.html)).

## 1. Validate the data first (fail-closed)

```bash
python data_integrity.py --dataset ../data/nhanes_multicycle_v2.csv \
  --output ../model_artifacts/reports/data_integrity_nhanes_multicycle_v2.json
```

The validator exits non-zero on blocking findings, and every training entry point
calls it internally, so invalidated files (`nhanes_merged.csv`,
`nhanes_multicycle.csv`) and denylist leaks cannot reach a training run.

## 2. Train

Bounded smoke run (minutes, safe for demonstrations):

```bash
python train_self_supervised.py --smoke
```

Full run with the committed configuration:

```bash
python train_self_supervised.py --config configs/ssl_full.json
```

Selected wider candidate (same label-free objective and 16-dimensional public
representation):

```bash
python train_self_supervised.py \
   --config configs/ssl_retrain_wide.json \
   --output-dir ../model_artifacts/metaboguard_ssl/runs/nhanes_multicycle_v2__retrain_wide__20260808

python promote_ssl_candidate.py \
   --candidate ../model_artifacts/metaboguard_ssl/runs/nhanes_multicycle_v2__retrain_wide__20260808 \
   --baseline ../model_artifacts/metaboguard_ssl/runs/nhanes_multicycle_v2__full__20260804T225555Z \
   --promote
```

The promotion utility requires improved validation loss, at least 1% lower
holdout reconstruction MSE, identical features/output/latent dimension and
holdout deviation-rank Spearman >= 0.80. It writes `promotion_report.json` and
updates `CURRENT.json` only when every gate passes.

The selected 2026-08-08 run uses hidden layers 128 -> 64 -> latent 16. Against
the frozen 96 -> 48 -> latent 16 baseline, holdout reconstruction MSE improved
from 0.040524 to 0.034867 (13.96%), with deviation-rank Spearman 0.911 and
top-5% flag Jaccard 0.516. PCA holdout reconstruction MSE was 0.046048. These
are label-free reconstruction/deviation metrics, not disease-prediction metrics.

Config-driven: every hyperparameter lives in `configs/*.json` and maps 1:1 to
`SSLConfig`. CLI flags override the file. Unknown keys raise, so a typo cannot
silently train a different model.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--backend {auto,torch,numpy}` | force a backend (default `auto`) |
| `--device {cpu,mps,auto}` | torch device; `cpu` default |
| `--epochs / --batch-size / --latent-dim / --max-train-rows` | override config |
| `--random-seed` | seed for splits, masking, noise and init |
| `--resume` | continue from `checkpoint_state.json` in the output directory |
| `--promote` | point `model_artifacts/metaboguard_ssl/CURRENT.json` at this run (refused for smoke runs) |
| `--output-dir` | explicit artifact directory |

Default output directory:
`model_artifacts/metaboguard_ssl/runs/<dataset>__<run_label>__<UTC timestamp>/`.
Nothing is overwritten, and the API only serves an artifact you explicitly name or
promote — so a smoke run can never become the default.

## 3. One-command demonstration

```bash
python run_meeting_demo.py          # smoke
python run_meeting_demo.py --full   # full configuration
```

Writes to `model_artifacts/demo_runs/<label>__<timestamp>/`:
`data_integrity_report.json`, `ssl_artifact/` (weights, preprocessor, metadata,
model card, splits, checkpoint), `baselines/baseline_report.json`,
`run_summary.json` and a human-readable `RUN_SUMMARY.md`.

## 4. Artifact contents and versioning

| File | Purpose |
| --- | --- |
| `autoencoder_weights.npz` | best-validation weights, backend-independent |
| `preprocessor.joblib` | ColumnTransformer fit on the **training partition only** |
| `metadata.json` | features, split policy, config, history, capabilities, deviation reference, run manifest |
| `MODEL_CARD.md` | generated card stating scope, limits and the longitudinal blocker |
| `splits.npz` | exact train/validation/holdout row positions |
| `checkpoint_weights.npz`, `checkpoint_state.json` | resumable checkpoint |
| `data_integrity_report.json` | the validation report for the exact input bytes |
| `promotion_report.json` | candidate-vs-baseline gates, metrics and scope statement |

`metadata.json.run_manifest` records the seed, backend, device, package versions
(python, numpy, pandas, scikit-learn, torch), epochs completed, wall-clock training
time and the checkpoint policy — enough to reproduce or audit a run.

## 5. Reproducibility rules implemented in code

1. **Deterministic seeding.** One seed drives the split, feature masking, Gaussian
   noise, weight initialisation and the training permutation, in both backends.
2. **Participant-grouped splits.** `data_integrity.group_split_indices` groups rows
   by `global_participant_id` (falling back to other identifiers) and asserts that
   participants never appear in two partitions.
3. **Train-only preprocessing.** Imputation medians, robust scaling and one-hot
   categories are fit on training rows only, per
   [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
   A unit test injects extreme values into the holdout and asserts the training
   scaler is unchanged.
4. **Train-only deviation reference.** Percentiles are computed against the training
   score distribution, so a percentile has a stable meaning.
5. **Label-free encoder.** Outcome columns are denylisted, the feature list is
   asserted disjoint from them, and the labels that exist in the file are recorded
   in the artifact as *present but unused*.
6. **Fail-closed gates.** Invalidated datasets/targets raise; longitudinal heads
   raise unless a horizon has ≥50 events and ≥50 non-events.

## 5b. Research pass (reliability, phenotypes, evidence)

```bash
cd api
../.venv/bin/python data_reliability.py --output ../model_artifacts/reports/data_reliability.json
../.venv/bin/python evidence_catalogue.py --strict
../.venv/bin/python clustering.py --complete-cases-only
../.venv/bin/python run_research_pass.py           # all of the above plus SVG charts
```

`run_research_pass.py` writes `model_artifacts/research_runs/research__<UTC>/` containing
`data_integrity_report.json`, `data_reliability_report.json`,
`evidence_catalogue_report.json`, `clustering_all_adults/`, `clustering_complete_cases/`,
`charts/*.svg` with the CSV behind each chart, `research_pass.json` and
`RESEARCH_SUMMARY.md`. Clustering consumes the **frozen** encoder artifact and never
retrains it. See [`CLUSTERING.md`](CLUSTERING.md) and
[`EVIDENCE_AND_CLAIMS.md`](EVIDENCE_AND_CLAIMS.md).

## 6. Expected console noise (all non-fatal)

| Message | Source | Handling |
| --- | --- | --- |
| `PydanticDeprecatedSince211: Accessing the 'model_fields' attribute on the instance is deprecated` (`chromadb/types.py:144`) | third-party: Chroma 0.6.3 with Pydantic ≥2.11 | **Left visible.** It is Chroma's code, not ours, and suppressing deprecation warnings globally would hide real ones. It disappears when Chroma is upgraded. |
| `Failed to send telemetry event ...: capture() takes 1 positional argument but 3 were given` | Chroma product telemetry vs the installed posthog version | **Silenced, narrowly.** `api/biomarker.py` sets `anonymized_telemetry=False` and raises only the `chromadb.telemetry.product.posthog` logger to CRITICAL. No data ever left the machine (the call fails before any request), and every other logger is untouched. |
| `ResourceWarning: unclosed database in <sqlite3.Connection ...>` | Chroma's persistent SQLite connection, opened by project code | **Fixed.** `_chroma_client()` is now a context manager that stops the `SqliteDB` component and clears the system cache, so the connection is closed on every path. |
| `UserWarning: [Errno 1] Operation not permitted. joblib will operate in serial mode` | joblib probing shared memory in a restricted shell | Environment-specific, harmless; does not appear in a normal local terminal. |

If a *new* warning appears, treat it as real: nothing in this project suppresses warnings
globally.

## 7. Scoring

```bash
python score_prevention_record.py \
  --artifact ../model_artifacts/metaboguard_ssl/nhanes_multicycle_v2 \
  --input ../metaboguard_sample_input.json
```

Output fields are `metabolic_deviation_score`, `reference_percentile`,
`latent_representation`, `top_deviation_features` — never a disease probability.

## 7b. Realistic synthetic Patient Probe records

The Patient Probe generator is separate from model training. Refresh its
aggregate profile model after the NHANES source or split policy changes:

```bash
python export_synthetic_profile_model.py
python ../deploy/professor/prepare_assets.py
```

`synthetic_profile_model.json` contains 1st-99th empirical quantiles, category
frequencies and regularized rank-correlation factors from the same
participant-grouped training partition. The browser uses a seeded Gaussian
copula to create novel coherent combinations, then recomputes HOMA-IR and total
cholesterol exactly. No participant row, identifier or authoring path is bundled.
Synthetic records are UI/research fixtures and are never added to SSL training.

## 8. What full training does *not* give you yet

A trained encoder does not enable future-risk scoring. That requires linked
incident outcomes with follow-up time; see the blocker section in
[`MODEL_CARD.md`](MODEL_CARD.md) and [`METHODOLOGY.md`](METHODOLOGY.md).
