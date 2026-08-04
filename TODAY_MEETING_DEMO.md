# TODAY_MEETING_DEMO.md - MetaboGuard demonstration runbook (2026-08-04)

**One-line framing for the room:** MetaboGuard currently learns an unlabelled metabolic
representation and flags unusual profiles for clinician review. It does **not** predict
future cancer or diabetes, and the code refuses to pretend it can.

**Backend for today: NumPy.** PyTorch is not installed in `.venv` (no network access
during preparation), so every verified number below comes from the deterministic NumPy
backend. **Torch-backend parity is unverified** - do not claim it. Nothing about the
demonstration depends on PyTorch.

A complete, verified full run is already saved in the repository, so nothing has to be
trained live:
`model_artifacts/metaboguard_ssl/meeting_2026-08-04/` (see its `README.md`).

---

## The 5 commands (exactly as they work in this checkout)

```bash
# 0. Repository root, project interpreter
cd /Volumes/Personal-Projects/KERI/api

# 1. Fail-closed data validation - prints "status": "ok"
../.venv/bin/python data_integrity.py \
  --dataset ../data/nhanes_multicycle_v2.csv \
  --output ../model_artifacts/reports/data_integrity_nhanes_multicycle_v2.json

# 2. Meeting demo: validate + bounded smoke training + PCA/Isolation-Forest baselines
#    + one example score, into model_artifacts/demo_runs/smoke__<UTC>/   (~25 s)
../.venv/bin/python run_meeting_demo.py

# 3. Full self-supervised run, 40 epochs, into model_artifacts/demo_runs/full__<UTC>/
#    (~70 s here; already reproduced three times with identical losses)
../.venv/bin/python run_meeting_demo.py --full

# 4. Test suite: data integrity, leakage, gates, splits, pipeline, baselines, API
#    Expect "OK" over 60 tests. Skip count varies by environment; the verdict is what matters.
../.venv/bin/python -m unittest discover -p "test_*.py"

# 5. Frontend production build (separate terminal tab)
cd ../frontend && npm run build
```

Show the saved artifact instead of retraining:

```bash
cd /Volumes/Personal-Projects/KERI/api
../.venv/bin/python score_prevention_record.py \
  --artifact ../model_artifacts/metaboguard_ssl/meeting_2026-08-04/ssl_artifact \
  --input ../model_artifacts/metaboguard_ssl/nhanes_multicycle_v2/sample_input.json
cat ../model_artifacts/metaboguard_ssl/meeting_2026-08-04/RUN_SUMMARY.md
```

Optional API demonstration:

```bash
cd /Volumes/Personal-Projects/KERI/api && ../.venv/bin/uvicorn main:app --port 8000
# POST /api/v1/prevention-capabilities -> longitudinal_heads_enabled: false
# POST /api/v1/prevention-future-risk  -> HTTP 409 with the exact blocker
# POST /api/v1/prevention-score        -> deviation score + percentile + latent
# POST /api/v1/data-integrity          -> full coding/leakage/gate report
# POST /api/v1/data-reliability        -> eligibility tiers + drift + label confidence
# GET  /api/v1/evidence-catalogue      -> clinician-ready evidence rows with provenance
# POST /api/v1/research-clusters       -> phenotype summary or explicit abstain
# POST /api/v1/data-reliability        -> eligibility tiers + drift + label confidence
# GET  /api/v1/evidence-catalogue      -> clinician-ready evidence rows with provenance
# POST /api/v1/research-clusters       -> phenotype summary or explicit abstain
```

---

## What each command demonstrates

| Command | Demonstrates |
| --- | --- |
| `data_integrity.py` | MCQ230 coding is correct (**29 = Pancreas** → 19 cases; **39 = Other** → 318 rows, never counted), 107,622 rows / 63,041 adults, no duplicate participant ids, allowlist vs denylist clean, all three horizons ungateable |
| `run_meeting_demo.py` | End-to-end fail-closed pipeline: validate → label-free training → baselines → example score → versioned artifacts + `RUN_SUMMARY.md` |
| `run_meeting_demo.py --full` | Full configuration, and that the encoder beats a capacity-matched PCA baseline on holdout reconstruction |
| `unittest discover` | The safety properties are tested, not just asserted in prose. Expect `OK` over a suite of **60 tests**; the number reported as skipped varies by environment (some tests skip when the model-artifact directory is not writable), so read the `OK` verdict rather than the skip count |
| `npm run build` | The dashboard compiles and its wording matches the API's honest terminology |

---

## Verified results - durable artifact `meeting_2026-08-04`

Dataset `nhanes_multicycle_v2.csv`, sha256 `af790f9fed1f1317…`, 107,622 rows, 63,041
adults, participant-grouped 70/15/15 split at seed 42, NumPy backend, CPU.

| Run | Epochs | Train / val / holdout | Final val loss | Holdout reconstruction MSE | Wall time |
| --- | --- | --- | --- | --- | --- |
| Smoke (3 epochs, 6k rows) | 3 | 6,000 / 9,456 / 9,457 | 0.2668 | 0.2953 | ~3 s |
| **Full (saved artifact)** | 40 / 40 | 44,128 / 9,456 / 9,457 | 0.0417277788500920 | **0.0428615594540357** | 69.8 s |

Determinism: three independent full runs at seed 42 produced identical per-epoch losses
(epoch 1 train 0.6150440173807605, validation 0.2372332485828424) and identical final
metrics. Wall time varies with machine load (28–70 s observed).

Baselines on the identical preprocessing and splits:

| Method | Holdout reconstruction MSE |
| --- | --- |
| MetaboGuard-SSL v1 (16-d latent) | **0.0428615594540357** |
| PCA (16 components, capacity-matched) | 0.0460477098822594 |
| Isolation Forest (200 trees) | not applicable (anomaly score only) |

Agreement on the holdout (Spearman of deviation scores / Jaccard of the top-5 % flags):
SSL vs Isolation Forest 0.716 / 0.151, SSL vs PCA 0.546 / 0.248, PCA vs Isolation Forest
0.505 / 0.145. Interpretation: the methods flag overlapping but far from identical
profiles, so "unusual" is method-dependent and needs clinician adjudication.

Cross-sectional association probes on the holdout partition (**already-present
conditions, not future risk**): any-cancer prevalence AUROC 0.731 / AUPRC 0.201 (837
positives), Type 2 diabetes proxy AUROC 0.927 / AUPRC 0.690 (1,075 positives), Type 1
proxy not evaluated (fewer than 50 holdout cases).

### Where the artifact lives

```txt
model_artifacts/metaboguard_ssl/meeting_2026-08-04/
├── README.md                      # provenance, metrics, promotion status
├── RUN_SUMMARY.md / run_summary.json
├── data_integrity_report.json
├── baselines/baseline_report.json
└── ssl_artifact/
    ├── autoencoder_weights.npz    # 12 arrays, backend-independent names
    ├── preprocessor.joblib        # fit on the training partition only
    ├── splits.npz                 # 44,128 / 9,456 / 9,457 row positions
    ├── metadata.json              # config, history, capabilities, run_manifest
    ├── MODEL_CARD.md              # generated scope-and-limits card
    └── checkpoint_weights.npz / checkpoint_state.json   # resumable at epoch 40
```

`model_artifacts/metaboguard_ssl/CURRENT.json` is intentionally absent: this run is
**not promoted** for API serving and is not a clinical or future-risk model.

---

## Sentences that are safe to say

- "The encoder is trained without any disease label; labels are only used afterwards for
  clearly-marked cross-sectional association checks on a held-out partition."
- "The score is a metabolic deviation score plus a percentile against the training
  reference. It is not a probability of developing anything."
- "Our intended horizons are 1, 3 and 5 years with a minimum of 50 events and 50
  non-events each. No horizon passes that gate today, so those heads are disabled in code
  and the API returns HTTP 409."
- "The Type 2 association number is prevalence separation. It shows the representation
  carries metabolic signal, not that we can forecast diabetes."
- "The historical pancreatic-cancer results are invalidated and cannot be re-enabled: the
  corrected coding leaves 19 prevalent cases, and both the dataset files and the
  supervised target are blocked in code."
- "Today's numbers are from the deterministic NumPy backend; PyTorch parity is still to be
  verified."

## Sentences to avoid

- Anything of the form "X % risk of cancer/diabetes".
- "Validated", "screening tool", "early detection" - none applies yet.
- Quoting the Type 2 AUROC as prevention performance.
- Claiming the torch backend has been benchmarked.

---

## The one blocker

Future-risk modelling is blocked by **data, not code**. NHANES here is a repeated
cross-section with one observation per participant and no follow-up time; TCGA-CDR is
post-diagnosis and is denylisted from prevention scoring by column prefix. Until a linked
incident-outcome cohort exists (NHANES-linked mortality/registry linkage, or an EHR cohort
with dated diagnoses), the 1/3/5-year heads stay disabled. The gate logic is already
implemented and tested, so enabling them is a data-ingestion task, not a modelling
rewrite.

## What remains for a full, publishable run

1. Torch parity is done: PyTorch 2.13.0 is installed and the full run on `--backend torch`
   reached holdout MSE 0.04052 (NumPy 0.04286). Promote whichever run you want served with
   `train_self_supervised.py --config configs/ssl_full.json --promote`.
2. Longitudinal cohort acquisition, then clinician review of the deviation-flag workflow.
3. A non-artefact clustering solution: harmonise assay handling across cycles (or analyse a
   single cycle) so survey cycle stops dominating, then repeat the phenotype pass.

See [`docs/TRAINING.md`](docs/TRAINING.md), [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).
