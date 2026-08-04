# Exploratory phenotype clustering

## Why clustering, and why exploratory

Prof. Nada's recommendation (see
[`decisions/2026-08-04-professor-feedback.md`](decisions/2026-08-04-professor-feedback.md))
was to look for structure the data can actually support instead of forcing a supervised
outcome it cannot. Clustering fits that: it needs no labels, so it is not blocked by the
absence of incident outcomes.

It is **exploratory** for three reasons that will not change with better code:

1. **No ground truth.** There is no external criterion that says a metabolic phenotype is
   correct. Cluster validity indices measure geometry, not biology.
2. **No outcome to validate against.** Confirming a phenotype means showing it predicts
   something later. That needs follow-up time, which these files do not have.
3. **Survey data invites artefacts.** A pooled cross-section carries cycle effects,
   subsample assay availability and missingness patterns that clustering will happily
   recover instead of biology.

So the honest framing is: clustering generates **hypotheses about patient/metabolic
phenotypes** for a future longitudinal study. A cluster is never a cancer diagnosis, a
cancer subtype or a cancer site, and the output schema has no field for one.

## What is clustered

`api/clustering.py` consumes the **frozen** self-supervised artifact:

- preprocessing comes from the artifact (`preprocessor.joblib`), so it stays fit on the
  training partition only;
- the encoder is loaded read-only and never retrained;
- split boundaries come from the artifact's persisted `splits.npz` when the row count
  matches, and are otherwise recomputed with the same seeded, participant-grouped policy —
  the report states which happened;
- `--space latent` (default) clusters the 16-dimensional representation;
  `--space features` clusters the shared preprocessed feature matrix.

Models are fit on the **training partition only**; k-means and Gaussian mixtures are then
assigned out of sample to the holdout, and the density arm reports train-partition
diagnostics because it has no out-of-sample assignment.

## Methods

| Arm | Implementation | Notes |
| --- | --- | --- |
| Centroid | `sklearn.cluster.KMeans` | k = 2…8 |
| Model-based | `sklearn.mixture.GaussianMixture` | full covariance, gives membership posteriors |
| Density | `sklearn.cluster.HDBSCAN` (in-tree since scikit-learn 1.3) | degrades to `DBSCAN` and then to "unavailable" rather than adding a dependency ([scikit-learn](https://github.com/scikit-learn/scikit-learn); [HDBSCAN, Campello et al.](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)) |

`method_availability` in every report states which density arm was actually used.
Consensus clustering in the sense of [Monti et al.](https://link.springer.com/article/10.1023/A:1023949509487)
is the natural next step and is deliberately deferred; the resample machinery needed for it
is already in place.

## Selection criteria and gates

A candidate is reported only if it passes **every** gate:

| Gate | Threshold | Why |
| --- | --- | --- |
| Cluster count | ≥ 2 | A single cluster is not a finding. |
| Silhouette | ≥ 0.15 | Basic separation ([Rousseeuw 1987](https://wis.kuleuven.be/stat/robust/papers/publications-1987/rousseeuw-silhouettes-jcam-sciencedirectopenarchiv.pdf)). |
| Smallest cluster | ≥ 5% of rows | Tiny clusters are noise or outlier pockets. |
| Bootstrap ARI | ≥ 0.60 mean | Resample-refit agreement. |
| Cluster-wise Jaccard | no cluster ≤ 0.50 | A cluster that dissolves under resampling is not real, even when global ARI is high ([Hennig](https://www.homepages.ucl.ac.uk/~ucakche/papers/clusta.pdf)). |
| Seed ARI | ≥ 0.60 | Not an artefact of initialisation. |
| Permuted-null gain | silhouette − null silhouette ≥ 0.05 | Column-permutation destroys joint structure while keeping marginals; the solution must beat that. |
| Outlier sensitivity | ARI ≥ 0.60 after dropping the top 1% deviation rows | A high silhouette can be produced by outliers rather than separation. |
| Negative controls | every gating control ≤ 0.30 | See below. |

Davies-Bouldin and Calinski-Harabasz are reported for every candidate as additional
internal-validity context.

Selection among passing candidates is by silhouette + bootstrap ARI. **No disease label
takes part in fitting or selection.**

Stability is necessary, not sufficient: a stable clustering can be a stable artefact, which
is why the permuted null and the negative controls are gates too.

## Negative controls (mandatory)

| Control | Statistic | Gating |
| --- | --- | --- |
| Survey cycle | bias-corrected Cramér's V | yes |
| Missingness burden | correlation ratio (η) | yes |
| Assay-availability burden (quartile bins of observed features) | bias-corrected Cramér's V | yes |
| Assay-availability pattern (exact per-row string) | bias-corrected Cramér's V | **no** — high-cardinality diagnostic only |
| Age | correlation ratio | yes |
| Sex | bias-corrected Cramér's V | yes |

Cramér's V is bias-corrected because the raw statistic inflates with the number of
categories; without the correction any high-cardinality control appears dominant regardless
of the clustering. The exact availability pattern is reported for transparency but never
gates a decision.

A solution above threshold on a gating control is marked `is_data_artefact` and cannot be
selected.

## Abstain

When nothing passes, the report status is **`no_stable_clusters`** with an
`abstain_reason` and a per-candidate `gate_failure_summary`. This is a result, not a
failure: the API returns it, the dashboard displays it, and no phenotype is invented.

## Current finding (2026-08-04, `nhanes_multicycle_v2.csv`)

Status: **`no_stable_clusters`**, in both the all-adults and the complete-case sensitivity
analysis.

- k-means solutions are geometrically clean (silhouette 0.53–0.56) and extremely stable
  (bootstrap ARI up to 0.999, cluster-wise Jaccard ≥ 0.99, silhouette far above the
  permuted null at ~0.15–0.17) — but they are **dominated by survey cycle**
  (bias-corrected V 0.32–0.54), so they are batch structure, not phenotypes.
- Gaussian mixtures at k=2 avoid control domination but fail resample stability
  (bootstrap ARI ≈ 0.54 against the 0.60 gate).
- HDBSCAN recovers clusters with good Jaccard (0.87–0.97) but they are both too small and
  control-dominated.
- On the wider grid (k-means and Gaussian mixtures, k = 2…6, complete cases) two further
  gates fire and are worth showing: `solution_driven_by_top_outliers` (k-means k=4,5 and
  GMM k=2,3 lose their labelling once the top 1% deviation rows are dropped) and
  `silhouette_not_better_than_permuted_null` (GMM k=2, whose silhouette of 0.243 does not
  clear the permuted-null baseline by the required margin). All ten candidates failed at
  least one gate.
- Restricting to complete cases on the 17 `usable_now` features (23,228 of 63,041 adult
  rows) removes the missingness/assay drivers but **not** the cycle effect.

Interpretation: on the pooled cross-section, the dominant recoverable structure is *when
and how a participant was measured*, not *how their metabolism differs*. That is a data
finding worth reporting, and it is exactly what the negative controls exist to catch.

## Reproducing

```bash
cd api

# full research pass: reliability + evidence + both clustering variants + charts
../.venv/bin/python run_research_pass.py

# clustering alone, complete-case sensitivity analysis
../.venv/bin/python clustering.py --complete-cases-only \
  --methods kmeans,gaussian_mixture,hdbscan --k-values 2,3,4,5,6 \
  --output-dir ../model_artifacts/clustering/sensitivity
```

Outputs per run: `clustering_report.json`, `candidate_metrics.csv`,
`negative_controls.csv`, and when a solution is selected `projection_points.csv` and
`cluster_feature_panels.csv`. The research pass adds accessible SVG charts plus the CSV
behind each one.

## What would make clustering confirmatory

1. Linked incident outcomes with follow-up time, so a phenotype can be tested against what
   happens next.
2. Harmonised assay handling across cycles (or single-cycle analysis) so cycle effects stop
   dominating.
3. Survey weights with PSU/strata variance estimation for any population-level statement.
4. Consensus clustering across methods and resamples, and clinician review of whether the
   surviving phenotypes are clinically meaningful.