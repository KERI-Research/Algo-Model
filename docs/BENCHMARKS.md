# MetaboGuard Benchmarks (unsupervised deviation)

## What is benchmarked, and what is not

Benchmarked: **how well an unsupervised method models the metabolic feature
distribution**, and **how much different methods agree about which profiles are
unusual**.

Not benchmarked: disease prediction. No baseline here predicts cancer or diabetes,
and none of these numbers may be presented as risk performance. Cross-sectional
association probes live with the encoder artifact and are labelled separately.

## Methods

| Method | Implementation | Latent budget |
| --- | --- | --- |
| MetaboGuard-SSL v1 | denoising autoencoder (`api/self_supervised.py`) | 16 |
| PCA reconstruction | [`sklearn.decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | matched to the encoder (16) |
| Isolation Forest | [`sklearn.ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) | n/a (tree ensemble, 200 trees) |

All three use the identical feature allowlist, the identical `ColumnTransformer`
fit **on the training partition only**
([scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html))
and the identical participant-grouped 70/15/15 split at seed 42. PCA's component
count is deliberately matched to the autoencoder latent dimension so the comparison
is about the mapping, not the capacity.

## Running

```bash
cd api
python baselines.py \
  --dataset ../data/nhanes_multicycle_v2.csv \
  --output-dir ../model_artifacts/benchmarks \
  --ssl-artifact ../model_artifacts/metaboguard_ssl/nhanes_multicycle_v2
```

`run_meeting_demo.py` runs the same code path against the artifact it just trained.
Output: `baseline_report.json` with the dataset fingerprint, split policy, per-method
metrics, holdout deviation distributions and pairwise agreement.

## Metrics reported

- **Reconstruction MSE** on train / validation / holdout (PCA and the autoencoder).
- **Deviation distribution** on the holdout (mean, sd, p50/p90/p95/p99) per method.
- **Spearman rank correlation** between methods' holdout deviation scores.
- **Top-5 % flag Jaccard**: overlap of the profiles each method flags as most
  unusual — the number that matters operationally, because it bounds how much a
  reviewer's worklist depends on the method choice.

## Verified results (2026-08-04, `nhanes_multicycle_v2.csv`, full run)

Source artifact: `model_artifacts/metaboguard_ssl/meeting_2026-08-04/` (40/40 epochs,
NumPy backend, CPU; `baselines/baseline_report.json` in that directory holds the raw
numbers).

| Method | Holdout reconstruction MSE |
| --- | --- |
| MetaboGuard-SSL v1 (16-d) | **0.0429** |
| PCA (16 components) | 0.0460 |
| Isolation Forest (200 trees) | not applicable |

| Pair | Spearman (holdout) | Top-5 % flag Jaccard |
| --- | --- | --- |
| SSL vs Isolation Forest | 0.716 | 0.151 |
| SSL vs PCA | 0.546 | 0.248 |
| PCA vs Isolation Forest | 0.505 | 0.145 |

Reading: the encoder gives a modest reconstruction improvement over a
capacity-matched linear baseline, so the non-linearity is earning something but not
dramatically. Flag overlap is low (0.15–0.25), which is the honest headline: "most
unusual" is method-dependent, so any review workflow must fix a method and have
clinicians adjudicate flags.

Under-trained models look worse, as expected: the 3-epoch smoke run reaches 0.2953
holdout MSE versus PCA's 0.0460. Reporting a smoke run as a result would be
misleading, which is why smoke artifacts are labelled and cannot be promoted.

## Deferred comparisons (deliberately not run today)

| Candidate | Status | Reason |
| --- | --- | --- |
| [TabPFN](https://github.com/PriorLabs/TabPFN) | deferred | supervised tabular foundation model; not a substitute for an unsupervised encoder, and supervised targets are gated |
| VIME / SCARF / ReMasker-style objectives ([ReMasker](https://arxiv.org/abs/2309.13793)) | scaffold only | re-masking loss on observed values only is a plausible upgrade to the masking objective; not adopted today to keep the demonstrated pipeline stable |
| [`inria-soda/tabular-benchmark`](https://huggingface.co/datasets/inria-soda/tabular-benchmark) | reference only | generic tabular algorithm benchmarking, not clinical validation |
| [`gabrielaltay/tcga-tabular-open`](https://huggingface.co/datasets/gabrielaltay/tcga-tabular-open) | excluded from scoring | TCGA is post-diagnosis; denylisted for prevention work |
| PyOD anomaly zoo | not added | native scikit-learn PCA + Isolation Forest is sufficient; avoid a new dependency before the meeting |

No project or patient-level data is uploaded to any external service, and no cloud
inference is used.