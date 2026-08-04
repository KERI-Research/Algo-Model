# MetaboGuard Safety and Model Card (project level)

A per-run card is generated automatically next to every artifact
(`MODEL_CARD.md` inside the artifact directory). This document is the project-level
statement that applies to all current MetaboGuard outputs.

## Identity

| Field | Value |
| --- | --- |
| System | MetaboGuard-SSL v1 (denoising autoencoder, 16-dimensional latent) |
| Code version | `metaboguard-ssl-v1.1` (`api/self_supervised.py`) |
| Owner | KERI department, research use |
| Status | Research prototype, clinician-reviewed, **non-diagnostic** |
| Reference artifact | `model_artifacts/metaboguard_ssl/meeting_2026-08-04/` (40/40 epochs, NumPy backend, CPU, not promoted) |
| Backends | Both verified on the full run: torch 2.13.0 holdout MSE 0.04052, deterministic NumPy fallback 0.04286, capacity-matched PCA 0.04605 |
| Data | NHANES 1999–March 2020 corrected pooled cross-section (`nhanes_multicycle_v2.csv`), TCGA-CDR for post-diagnosis context only |

## Intended use

- Learn an unlabelled representation of adult metabolic measurements.
- Produce a **metabolic deviation score** and its **reference percentile** to help a
  clinician decide which profiles deserve a closer look.
- Support methodological research towards a future, properly validated early-warning
  model.

## Out of scope (must not be done)

- Diagnosis, triage, treatment decisions or patient reassurance.
- Any statement of the form "N % chance of developing cancer/diabetes".
- Presenting cross-sectional association metrics as future-risk performance.
- Re-enabling the invalidated pancreatic-cancer models or datasets.
- Reporting the research-only Type 1 diabetes proxy as a Type 1 endpoint.

## Supervised comparison model naming contract

The cross-sectional association models are benchmarked against each other and the winner
is selected by AUPRC then AUROC. There is exactly **one canonical name per candidate**, and
these strings are the keys of `benchmarks` and the value of `model` / `model_name`:

| Constant in `api/biomarker.py` | Canonical name |
| --- | --- |
| `MODEL_NAME` | `metaboguard_hist_gradient_boosting_v1` |
| `ALTERNATE_MODEL_NAME` | `metaboguard_xgboost_v1` |

The pre-rename key `hist_gradient_boosting_biomarker_v1` is **not** produced and **not**
aliased: a repository-wide search found it only inside a stale test assertion, with no
runtime consumer in the API, dashboard, docs or artifacts, so no backward-compatibility
shim was warranted. `api/test_biomarker_flow.py` now asserts against the imported
constants and additionally enforces that every benchmark key starts with `metaboguard_`,
so the contract cannot drift silently again.

## Output vocabulary (enforced in the API)

| Field | Meaning |
| --- | --- |
| `metabolic_deviation_score` | How unusual a profile is versus the training reference. Unbounded, not a probability. |
| `reference_percentile` | Rank of that score within the training reference distribution. |
| `latent_representation` | 16-dimensional learned encoding. |
| `top_deviation_features` | Features contributing most to reconstruction error. |
| `cross_sectional_association_probability` | Probability that a profile **already has** a recorded diagnosis in a cross-sectional survey. Not future risk. |
| `is_future_risk_probability` | Always `false` on current data. |

`cancer_risk_probability` is retained for one release as a deprecated alias of
`cross_sectional_association_probability` and must not be used in new surfaces.

## Professor-facing outputs (2026-08-05 envelope)

Professor-facing responses now expose an explicit, additive structure so each
surface can render the same interpretation contract:

- `current_profile_assessment`: what this result means now (deviation band +
   percentile on deviation routes, or prevalence-association framing on
   cross-sectional biomarker routes).
- `standout_factors`: top model contributions for this record.
- `data_readiness`: missing inputs, priority and why each field matters.
- `research_association`: research-only association status and scope text.
- `safety_contract`: explicit non-diagnostic + future-risk disabled flags.

This envelope does not change capability boundaries: no cancer-type prediction,
no future-development probabilities, and no horizon outputs on current
cross-sectional NHANES data.

## Safety gates implemented and tested

| Gate | Behaviour |
| --- | --- |
| Invalidated datasets | `nhanes_merged.csv`, `nhanes_multicycle.csv` raise on any load path and are hidden from the API dataset list |
| Invalidated targets | `PancreaticCancer`, `NODM_PancreaticCancer` raise in both `train_biomarker_model` and `load_biomarker_model`, so a stale artifact on disk cannot be served |
| Cancer coding | Validator recomputes the pancreatic label from MCQ230A–D **code 29 (Pancreas)** and blocks if the stored label matches code 39 (**Other**) counts |
| Leakage denylist | Outcome labels, label-derived columns and every `tcga_*` post-diagnosis column are refused as model inputs |
| Split boundaries | Participant-grouped, seeded; preprocessing and deviation reference fit on the training partition only |
| Event-count gate | 1 y / 3 y / 5 y horizons require ≥50 events **and** ≥50 non-events; otherwise longitudinal heads stay disabled and `/api/v1/prevention-future-risk` returns HTTP 409 |
| Minimum association cases | Post-hoc probes are suppressed below 50 cases per class |
| Artifact promotion | Smoke runs cannot be promoted to the artifact the API serves |

## Added scope, 2026-08-04: exploratory phenotypes and evidence provenance

| Surface | Status | Rule |
| --- | --- | --- |
| Exploratory phenotype clustering (`api/clustering.py`) | research only | Label-free in fit and selection; clusters are patient/metabolic phenotypes and may never be named after a cancer, cancer site or disease subtype; abstains when stability, permuted-null, outlier-sensitivity or negative-control gates fail. |
| Data reliability report (`api/data_reliability.py`) | gate | Feature eligibility tiers `usable_now` / `qualified_use` / `unavailable` / `prohibited`; hard violations raise. |
| Evidence catalogue (`data/evidence/biomarker_evidence.json`) | gate | Mandatory provenance; a row reaches a clinician view only with a real source and a graded evidence level; causal phrasing rejected without a causal design. |
| Cancer-site assignment | **disabled** | Site comes from a self-reported multi-select item with 19 prevalent pancreatic cases. |

### Statement classes required in clinician-facing surfaces

`data observation`, `model association`, `published evidence`,
`causal claim not established`. The API returns the class with every payload and the
dashboard renders it as a visible badge.

### Panel framing (scientific correction)

Early detection is a **panel and feature-interaction** problem. The statement that no cancer
has a specific biomarker is **false** and prohibited; the defensible statement is that no
single marker is universally sufficient across cancers. Catalogued counter-examples: CA19-9
(pancreas) and the GALAD components (HCC), both with their lead-time limits recorded. See
[`EVIDENCE_AND_CLAIMS.md`](EVIDENCE_AND_CLAIMS.md).

### Approved burden wording

Global pancreatic cancer burden was 531,318 cases and 490,786 deaths in 2024
([CA Cancer J Clin](https://pmc.ncbi.nlm.nih.gov/articles/PMC13343830/), DOI 10.3322/caac.70090);
a demographic constant-rate projection gives 998,663 cases and 936,038 deaths in 2050
([JAMA Netw Open](https://pmc.ncbi.nlm.nih.gov/articles/PMC11539015/), DOI 10.1001/jamanetworkopen.2024.43198).
This is not a causal forecast.

### Claims contract

Detection-performance claims must satisfy PRoBE
([design](https://edrn.cancer.gov/documents/158/PRoBEStudyDesign.pdf), DOI 10.1093/jnci/djn326),
TRIPOD+AI ([BMJ](https://www.bmj.com/content/385/bmj-2023-078378), DOI 10.1136/bmj-2023-078378),
PROBAST+AI ([site](https://www.probast.org/probast_ai)) and STARD
([EQUATOR](https://www.equator-network.org/reporting-guidelines/stard/)). Current outputs meet
none of the specimen or lead-time requirements and are therefore research only.

## Known limitations

1. **Cross-sectional data.** One observation per participant, no follow-up. Nothing
   about disease *development* can be measured, only co-occurrence.
2. **Self-reported outcomes.** NHANES cancer and diabetes status come from
   questionnaire items (MCQ220, MCQ230A–D, DIQ010), with recall and survivorship bias.
3. **Prevalent, not incident, pancreatic cancer.** 19 corrected prevalent cases in
   107,622 rows; no supervised pancreatic model is possible or permitted.
4. **Type 1 diabetes is research-only.** No autoantibodies, no approved genetics and
   no confirmatory C-peptide criteria exist in these files, so the subtype flag is a
   proxy derived from diagnosis age and insulin use.
5. **TCGA-CDR is post-diagnosis.** Stage, grade, tumour status, treatment response
   and follow-up time are excluded from prevention scoring by denylist.
6. **Deviation is not pathology.** An unusual profile may be unusual for benign
   reasons; agreement between methods on the top-5 % flags is only ~0.15–0.25
   (see [`BENCHMARKS.md`](BENCHMARKS.md)).
7. **Survey weights are not applied** to the representation model, so deviation
   percentiles describe the pooled sample, not the US population.
8. **Clustering currently abstains.** On the pooled cross-section the dominant recoverable
   structure is survey cycle, not metabolism, so no phenotype is reported. This is a data
   limitation, not a tuning problem.
9. **PyTorch is optional.** Today's verified runs used the deterministic NumPy
   backend; the torch backend must be re-verified after installing
   `requirements-ssl.txt`.

## Invalidated history (kept for audit only)

The historical supervised pancreatic-cancer artifacts, including
`model_artifacts/huggingface/diapan-risk-xgboost`, were invalidated when NHANES
MCQ230 site coding was corrected (**29 = Pancreas**, **39 = Other**). Their metrics
are meaningless and must never be quoted, exported or benchmarked. The retained
files exist purely as an audit trail and are blocked in code.

## Blocker for the next capability

Linked incident-outcome follow-up (for example NHANES-linked mortality or registry
linkage, or an EHR cohort with dated diagnoses) is the single prerequisite for
1/3/5-year horizon heads. The gate logic, split policy, and artifact/metadata plumbing
are already in place and tested, so enabling horizons is a data-ingestion task.

## Human oversight

Every output is intended for clinician review. The dashboard and API carry
non-diagnostic warnings on the capability, score and association surfaces, and the
future-risk endpoint is fail-closed rather than silently approximate.

## Sources already referenced by this repository

- [Zhou et al. 2025: genetic and clinical NODM-PC model](https://doi.org/10.1186/s12916-025-04048-4)
- [Yang et al. 2026: clinical CatBoost and multi-omics integration](https://doi.org/10.1186/s12967-026-07767-1)
- [scikit-learn: common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html)
