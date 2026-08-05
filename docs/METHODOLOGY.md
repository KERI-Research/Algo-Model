# MetaboGuard Methodology and Research Decision Log

## Study question

MetaboGuard asks whether self-supervised representations of metabolic,
clinical and behavioural data can identify unusual patient profiles and
support future clinician-reviewed prevention research across cancer and
diabetes. The project is exploratory research software, not a diagnostic
device.

The biological rationale is the bidirectional, duration-dependent association
between diabetes and pancreatic cancer. New-onset diabetes carries materially
higher risk than long-standing diabetes, while risk attenuates with duration
([Lee et al. 2023](https://pubmed.ncbi.nlm.nih.gov/36548964/);
[Song et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4519136/)).
This motivates continued pancreatic-cancer research, while the broadened
pan-cancer scope uses cancer labels only for post-hoc representation checks
until longitudinal incident outcomes become available.

## Self-supervised modelling decision

A pure unsupervised score cannot identify which disease will develop.
MetaboGuard therefore uses a hybrid design:

1. a denoising autoencoder learns representations without disease labels;
2. reconstruction and latent-distance scores identify unusual profiles;
3. frozen embeddings are tested with cross-sectional association heads;
4. future disease-specific heads are allowed only with longitudinal outcomes.

The current encoder uses 25 prevention-safe raw features, 55 transformed
dimensions and a 16-dimensional latent space. It was trained on 50,000
unlabelled adult NHANES rows.

Current post-hoc checks are:

| Label | AUROC | AUPRC | Status |
| --- | ---: | ---: | --- |
| Any-cancer prevalence | 0.699 | 0.169 | Cross-sectional association only |
| Type 2 diabetes proxy | 0.923 | 0.675 | Cross-sectional association only |
| Type 1 proxy | 0.909 | 0.135 | Unvalidated research-only proxy |

Diagnosis age and insulin-use variables used to derive diabetes subtype were
excluded from encoder inputs to avoid direct proxy-label leakage.

The current dataset capability is
`cross_sectional_representation_and_deviation_only`. It does not support future
disease-development claims.

## Why NHANES and TCGA-CDR are used

The datasets answer different questions:

- **NHANES** supplies general-population metabolic measurements and
  self-reported diabetes/cancer history. It supports risk-pattern discovery
  before or outside a confirmed cancer diagnosis.
- **TCGA-CDR** supplies outcomes among already diagnosed cancer patients. It
  supports mortality/progression prognosis, not early detection. TCGA-CDR
  standardises OS, DSS, DFI and PFI across more than 11,000 tumours
  ([Liu et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29625055/)).

Neither dataset contains serial, dated pre-diagnosis biomarker measurements.
Pooling NHANES increases sample size but does not create patient-level
longitudinal trajectories. MetaboGuard therefore names its engineered time features
**repeated-cross-sectional trajectory proxies**, not slopes or within-patient
changes.

## Multi-cycle NHANES cohort

### Included periods

`api/fetch_nhanes_multicycle.py` pools ten non-overlapping releases:

1. 1999-2000
2. 2001-2002
3. 2003-2004
4. 2005-2006
5. 2007-2008
6. 2009-2010
7. 2011-2012
8. 2013-2014
9. 2015-2016
10. 2017-March 2020 pre-pandemic

The standalone 2017-2018 J release is deliberately excluded because its
participants are already contained in the combined pre-pandemic release.
The August 2021-2023 release is excluded because CDC advises against pooling it
with earlier cycles after the pandemic-era collection gap
([CDC weighting guidance](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)).

### Resulting cohort

| Quantity | Count |
| --- | ---: |
| Total pooled participants | 107,622 |
| Diabetes-labelled | 101,532 |
| Diabetes-positive | 7,359 |
| Pancreatic-cancer-labelled | 58,682 |
| Correct pancreatic-cancer-positive | 19 |
| Pancreatic-cancer-positive diabetics | 7 |
| Usable positive diabetics after required-field cleaning | 6 |
| Pancreatic cancer 0-3 years after diabetes | 2 |

The exact source URL, filename, status, row count and retained columns for every
component/cycle are recorded in
`data/nhanes_multicycle_v2_build_report.json`. Missing files are reported rather
than silently converted into apparent biological missingness.

### Harmonisation

Early NHANES cycles bundled laboratory analytes in numbered files:

- `LAB10` / `L10`: glycohaemoglobin
- `LAB10AM` / `L10AM`: fasting glucose, insulin and C-peptide
- `LAB13` / `L13`: total cholesterol, HDL, triglycerides and LDL

Later cycles use named files such as `GHB`, `GLU`, `INS`, `TRIGLY`, `HDL` and
`TCHOL`. The builder maps these variants to one stable MetaboGuard schema. This is
necessary because NHANES variable names, units and eligibility rules can
change across cycles ([Nguyen et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9934713/)).

### Survey weights

The pooled file includes `combined_mec_weight_1999_2020` for descriptive
examined-sample population estimates:

- 1999-2002: `WTMEC4YR * (4 / 21.2)`
- 2003-2016: `WTMEC2YR * (2 / 21.2)`
- 2017-March 2020: `WTMECPRP * (3.2 / 21.2)`

The 1999-2002 bridge uses the NCHS-provided four-year weight because 1999-2000
used a different Census population base. These formulas follow CDC's
cycle-combining guidance
([CDC NHANES weighting module](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)).

This MEC weight is **not valid for fasting-subsample population estimates**.
The predictive pipeline is currently sample-level and unweighted. Population
inference using glucose, insulin or C-peptide requires the appropriate
fasting-subsample weights and design-based variance estimation. CDC warns that
ignoring weights, strata and PSUs biases estimates and understates uncertainty
([CDC quality-analysis guidance](https://wwwn.cdc.gov/nchs/nhanes/QualityAnalysesGuidelines.aspx)).

## Outcome construction

### Pancreatic cancer

`PancreaticCancer` is derived from NHANES `MCQ230A-D`; official code 29 means
pancreas and code 39 means Other. A participant is positive if any reported
cancer slot equals 29. Participants
with `MCQ220` equal to yes or no but no pancreatic site are negatives; unknown
responses remain missing.

This is self-report, not registry-confirmed ground truth. A CDC validation study
reported pancreatic-cancer self-report sensitivity of 90.9% and positive
predictive value of 83.3%, with wide intervals due to small counts
([CDC validation study](https://stacks.cdc.gov/view/cdc/210552/cdc_210552_DS1.pdf)).
Label noise is therefore expected and is part of the uncertainty budget.

### Diabetic risk cohort

`cohort_filter=diabetics_only` retains rows where `Diabetes == 1`. This directly
matches the brief's intended use: prioritising people inside an already
monitored diabetic population. The general-population model remains as a
baseline, not the primary clinical framing. However, only six corrected positive
cases survive required-field cleaning, so no pancreatic-risk model is fitted.

## Feature decisions

| Feature | Reason for inclusion | Qualification |
| --- | --- | --- |
| `recent_diabetes_onset`, `diabetes_duration_years` | New-onset risk is materially higher and decays with duration | Age at diagnosis is self-reported |
| `GHB_LBXGH`, HbA1c interactions | Higher HbA1c is associated with pancreatic-cancer risk, especially near diabetes onset ([EPIC](https://link.springer.com/article/10.1007/s00125-011-2316-0)) | One measurement per participant |
| `INS_LBXIN`, `homa_ir` | Represents insulin resistance/metabolic dysfunction | Fasting-subsample coverage only |
| `CPEP_LBXCPSI` | C-peptide has reported OR 4.24, highest vs lowest quartile ([Michaud et al.](https://pubmed.ncbi.nlm.nih.gov/17905943/)) | Available only 1999-2004; 9,501 non-null records |
| Weight-loss variables | Prediagnosis weight loss is a strong signal ([case-control study](https://pubmed.ncbi.nlm.nih.gov/33835301/)) | Self-reported current/prior weight |
| BMI, waist and interactions | Obesity and central adiposity are established moderate risk factors ([pooled analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073156/)) | Cross-sectional |
| hs-CRP | Retained for discovery completeness | Prior evidence is null/inconsistent ([Bao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3495286/)); should not be assumed causal |
| `diabetes_subtype` | Exploratory distinction between early-onset insulin-treated and other diabetes | Unvalidated heuristic, not a clinical diagnosis |
| `survey_cycle_index` | Allows the model to detect secular measurement/cohort shifts | A strong importance would be a warning for dataset drift, not a biomarker |
| `smoking_status`, `current_smoker` | Smoking survived selection in the reviewed genetic model | Harmonised 0 never, 1 former, 2 current |
| `alcohol_status`, `average_drinks_per_day` | Alcohol was selected by the reviewed CatBoost model | Questionnaire wording changed across cycles |
| `CBC_LBXHGB`, `CBC_LBXPLTSI` | Haemoglobin and platelet count were paper candidates | Routine CBC; >81% pooled coverage |
| `BIOPRO_LBXSATSI`, `BIOPRO_LBXSAPSI`, `BIOPRO_LBXSCR` | ALT, alkaline phosphatase and creatinine were paper candidates | Routine biochemistry; ~63% coverage |
| `hba1c_reciprocal_100`, `hba1c_squared` | Supports nonlinear HbA1c sensitivity analysis | Derived terms; not separate biomarkers |

`CA19-9` remains absent from NHANES. Published lead-time studies show useful
pre-diagnosis performance, so this is a substantive data gap rather than a
feature-engineering omission
([Fahrmann et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC8783758/)).

## Repeated-cross-sectional trajectory proxies

The following features approximate cohort-level temporal patterning:

- `hba1c_cycle_age_sex_z`: HbA1c standardised within survey cycle, age band
  and sex
- `hba1c_age_interaction`
- `hba1c_diabetes_duration_interaction`
- `hba1c_weight_loss_interaction`
- `survey_cycle_index`

They capture relative position and effect modification across repeated
cross-sections. They do **not** demonstrate that an individual's HbA1c changed
before cancer diagnosis.

## Modelling and rare-event evaluation

The pipeline benchmarks `HistGradientBoostingClassifier` and XGBoost. If
prevalence is below 10%, inverse-frequency class weighting is used. Weighting
is preferred to naive row duplication because it preserves observed records
while penalising missed rare cases.

Both AUROC and AUPRC are reported:

- AUROC measures rank discrimination across thresholds.
- AUPRC describes screening yield under extreme imbalance and must be compared
  with prevalence.
- `auprc_lift_over_prevalence` reports this relative yield explicitly.

AUPRC can favour higher-prevalence subgroups, so it is not treated as the only
model-comparison metric
([McDermott et al. 2024](https://arxiv.org/html/2401.06091v1)).

The selected model also reports:

- Brier score
- deterministic stratified-bootstrap 95% intervals for AUROC and AUPRC
- held-out positive/negative counts

The bootstrap samples positive and negative test indices separately so every
replicate remains evaluable. It captures held-out sample uncertainty only; it
does not include feature-selection, model-selection, survey-design or label
misclassification uncertainty.

## Current model status

There is no valid NHANES pancreatic-risk classifier. The former results used
MCQ230 code 39, which means Other, and are invalidated.

The corrected v2 target has 19 positives overall, 7 among diabetics, 6 after
required-field cleaning and only 2 meeting the exact three-year NODM-PC
definition. MetaboGuard requires at least 20 usable positives and negatives before
fitting. Dataset signatures prevent old artifacts from being reused after a
CSV changes.

Cycle-held-out evaluation is postponed until a larger incident
pancreatic-cancer cohort is available. With six usable positives, multiple
cycles contain no events and performance estimates would be dominated by
individual participants.

## Leakage controls

MetaboGuard excludes TCGA `OS`, `OS.time`, `PFI` and `PFI.time` because they define or
occur after the outcomes. Clinical leakage can substantially inflate apparent
performance; one published example fell from AUC 0.76 to 0.64 after a single
leaked feature was removed
([Chiavegatto Filho et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7880048/)).

For NHANES, reverse causation remains possible: biomarkers and weight loss may
have been measured after an earlier pancreatic-cancer diagnosis. This is not
technical leakage, but it means the model may recognise prevalent disease
rather than predict future disease.

## Reproduction

```bash
cd api

# Download, harmonise and write the pooled dataset plus build report
python fetch_nhanes_multicycle.py

# Audit corrected outcome and feature coverage. Model fitting intentionally
# fails until at least 20 usable positive cases exist.
python -c "import pandas as pd; d=pd.read_csv('../data/nhanes_multicycle_v2.csv'); print(d[['PancreaticCancer','NODM_PancreaticCancer']].sum())"
```

## Next validation steps

1. Acquire a linked incident pancreatic-cancer cohort with exact diabetes and
   cancer diagnosis dates.
2. Perform sensitivity analysis with and without the unvalidated
   `diabetes_subtype` proxy once the event threshold is met.
3. Add survey-design-aware descriptive analysis using the combined MEC weight,
   strata and PSU fields.
4. Derive or ingest fasting-subsample combined weights before making weighted
   claims about insulin, glucose, HOMA-IR or C-peptide.
5. Calibrate only after substantially more events are available; stable
   calibration curves commonly require roughly 200 events and 200 non-events
   ([calibration review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996/)).
6. Seek a longitudinal cohort with serial biomarkers and CA19-9 to answer the
   true early-detection trajectory question.

## Engineering decision log (2026-08-04 pass)

Each decision below is implemented in code and covered by a test.

| Decision | Reason |
| --- | --- |
| Single fail-closed validator (`api/data_integrity.py`) called by every training, benchmarking and API entry point | Coding, leakage and capability rules must be impossible to bypass by choosing a different script. Blocking findings raise instead of warning. |
| Pancreatic-cancer label recomputed from MCQ230A–D **code 29** and compared with the stored column; a match against code 39 counts is a blocking error | Makes the historical 29/39 error a test failure rather than a memory. Corrected data: 19 pancreas rows vs 318 "Other" rows. |
| `PancreaticCancer` / `NODM_PancreaticCancer` targets and the two pre-correction CSVs raise on both the train and the load path | A stale artifact directory on disk could otherwise be served without retraining. |
| Participant-grouped, seeded 70/15/15 splits with a disjointness assertion | NHANES rows are one-per-participant today, but the same code must stay correct when longitudinal data arrive. Split indices are persisted in `splits.npz`. |
| Preprocessing and the deviation reference distribution fit on the training partition only | Standard leakage control ([scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)); a percentile only has meaning against a fixed reference. |
| Post-hoc association probes moved to the holdout partition and gated at 50 cases per class | Cross-validated probes on all rows overlap the encoder's training rows and inflate apparent separation. |
| Deterministic NumPy backend added alongside PyTorch | PyTorch was unavailable when the pipeline was built, and a research pipeline should not be blocked by an optional accelerator dependency. Torch 2.13.0 has since been installed and both backends are verified on the full run. A pure-NumPy Adam implementation of the same architecture keeps the demonstration reproducible; both backends export identical weight names. |
| CPU default, `mps` opt-in | Bit-for-bit reproducibility ([PyTorch randomness notes](https://pytorch.org/docs/stable/notes/randomness.html)). |
| Run manifest (seed, backend, device, package versions, epochs, wall time, checkpoint policy) plus generated model card per artifact | An artifact must be auditable without reading the code that produced it. |
| Timestamped run directories with an explicit `--promote` step, refused for smoke runs | Prevents an under-trained demonstration artifact from silently becoming the served model. |
| PCA (components matched to the latent dimension) and Isolation Forest baselines under identical preprocessing/splits | The encoder must justify its complexity against simple comparators; flag-overlap statistics show how method-dependent "unusual" is. |
| API renames the supervised output to `cross_sectional_association_probability`, keeps `cancer_risk_probability` only as a deprecated alias, and returns `is_future_risk_probability: false` | The old field name implies future risk that the data cannot support. |
| `/api/v1/prevention-future-risk` returns HTTP 409 with the horizon gate report | Fail-closed is safer and more informative than an approximate answer. |
| Future risk is engineered on a **synthetic longitudinal** cohort only, behind `simulation_mode=true` | NHANES is cross-sectional and cannot answer "what happens next". A synthetic cohort lets the schema, endpoint definitions, masks, competing-risk handling, calibration and evaluation be built and tested truthfully without implying clinical performance. Preferred generator: official Synthea v3.3.0 (DOI 10.1093/jamia/ocx079; validity limits DOI 10.1186/s12911-019-0793-0). |
| Horizons are gated independently at 50 events and 50 non-events; failing horizons abstain | An underpowered horizon produces a number that looks like knowledge and is not. Abstention is reported as a result. |
| Ordinary-incidence and declared-enrichment synthetic cohorts are reported separately | Enriched cohorts have deliberately inflated event rates, so their absolute probabilities are not population-calibrated and must never be pooled with ordinary-incidence metrics. |
| Patients censored before a horizon are masked out, never treated as negatives | Treating censored patients as event-free is the classic way to manufacture optimistic performance. Death is handled as a competing event with cause-specific coding. |
| Model selection is calibration-first, and the temporal model must pass a time-reversal control | Discrimination without calibration misleads clinicians (DOI 10.1136/bmj.e5900), and a sequence model that scores identically on reversed visits is not using time. |
 **synthetic longitudinal** cohort only, behind `simulation_mode=true` | NHANES is cross-sectional and cannot answer "what happens next". A synthetic cohort lets the schema, endpoint definitions, masks, competing-risk handling, calibration and evaluation be built and tested truthfully without implying clinical performance. Preferred generator: official Synthea v3.3.0 (DOI 10.1093/jamia/ocx079; validity limits DOI 10.1186/s12911-019-0793-0). |
| Horizons are gated independently at 50 events and 50 non-events; failing horizons abstain | An underpowered horizon produces a number that looks like knowledge and is not. Abstention is reported as a result. |
| Ordinary-incidence and declared-enrichment synthetic cohorts are reported separately | Enriched cohorts have deliberately inflated event rates, so their absolute probabilities are not population-calibrated and must never be pooled with ordinary-incidence metrics. |
| Patients censored before a horizon are masked out, never treated as negatives | Treating censored patients as event-free is the classic way to manufacture optimistic performance. Death is handled as a competing event with cause-specific coding. |
| Model selection is calibration-first, and the temporal model must pass a time-reversal control | Discrimination without calibration misleads clinicians (DOI 10.1136/bmj.e5900), and a sequence model that scores identically on reversed visits is not using time. |
 **synthetic longitudinal** cohort only, behind `simulation_mode=true` | NHANES is cross-sectional and cannot answer "what happens next". A synthetic cohort lets the schema, endpoint definitions, masks, competing-risk handling, calibration and evaluation be built and tested truthfully without implying clinical performance. Preferred generator: official Synthea v3.3.0 (DOI 10.1093/jamia/ocx079; validity limits DOI 10.1186/s12911-019-0793-0). |
| Horizons are gated independently at 50 events and 50 non-events; failing horizons abstain | An underpowered horizon produces a number that looks like knowledge and is not. Abstention is reported as a result. |
| Ordinary-incidence and declared-enrichment synthetic cohorts are reported separately | Enriched cohorts have deliberately inflated event rates, so their absolute probabilities are not population-calibrated and must never be pooled with ordinary-incidence metrics. |
| Patients censored before a horizon are masked out, never treated as negatives | Treating censored patients as event-free is the classic way to manufacture optimistic performance. Death is handled as a competing event with cause-specific coding. |
| Model selection is calibration-first, and the temporal model must pass a time-reversal control | Discrimination without calibration misleads clinicians (DOI 10.1136/bmj.e5900), and a sequence model that scores identically on reversed visits is not using time. |
| Masking objective left unchanged (no ReMasker-style observed-value-only loss yet) | A plausible upgrade ([ReMasker](https://arxiv.org/abs/2309.13793)) but not worth destabilising a demonstrable pipeline today; recorded as deferred work. |

## 2026-08-04 supervisor feedback: methodological consequences

Full decision record: [`decisions/2026-08-04-professor-feedback.md`](decisions/2026-08-04-professor-feedback.md)
(recollected notes, not a transcript).

### Early detection is a panel problem, not a single-marker problem

The scientifically correct statement, now enforced by the evidence-catalogue validator, is
that **no single marker is universally sufficient for early detection across cancers**, so
panels and interacting features are required. The opposite claim — that cancers have no
specific biomarkers — is **false** and is rejected in code: CA19-9 is catalogued with
prediagnostic AUC 0.998 at diagnosis falling to 0.74 at 12 months and 0.55 at 5 years
([BJS Open meta-analysis](https://academic.oup.com/bjsopen/article/8/3/zrae046/7700226)),
and GALAD is catalogued as a validated five-component panel for high-risk cirrhosis
surveillance ([Gastroenterology 2024](https://pubmed.ncbi.nlm.nih.gov/39293548/)). Both are
counter-examples to a blanket denial and simultaneously demonstrate why single markers are
not enough at useful lead times. Full details, including the allowed and denied statement
lists and the PRoBE / TRIPOD+AI / PROBAST+AI / STARD claims contract, are in
[`EVIDENCE_AND_CLAIMS.md`](EVIDENCE_AND_CLAIMS.md).

### Terminology

Causal phrasing is removed from anything describing our findings. Features are
**risk-associated features**, **early-development signals** or **biological pathways**.
`causal` survives only where it names the DoWhy estimation method, not a finding. The
evidence loader rejects causal phrasing unless a row's `study_design` is a causal design.

### Clustering replaces the unsupportable supervised framing

| Decision | Reason |
| --- | --- |
| Clustering is exploratory phenotype discovery, never disease classification | There is no ground truth for a metabolic phenotype and no outcome to validate against in cross-sectional data. |
| Labels are excluded from fit **and** from model selection | Otherwise "unsupervised" discovery becomes weak supervision through the selection step. |
| Frozen encoder, artifact preprocessing, persisted split indices | Clustering must not silently refit or leak; the report states whether persisted splits were reused or recomputed. |
| Mandatory negative controls (survey cycle, missingness burden, assay-availability burden, age, sex) | Pooled survey data reproduces measurement context first. Without controls, batch structure would be reported as biology. |
| Bias-corrected Cramér's V; exact assay pattern is diagnostic only | The raw statistic inflates with category count, which would make every high-cardinality control look dominant. |
| Cluster-wise bootstrap Jaccard alongside global ARI | A high global ARI can hide one dissolving cluster ([Hennig](https://www.homepages.ucl.ac.uk/~ucakche/papers/clusta.pdf)). |
| Column-permutation null and top-1% outlier-sensitivity refit | Silhouette can be produced by outliers or by geometry that appears in structureless data of the same shape ([Rousseeuw 1987](https://wis.kuleuven.be/stat/robust/papers/publications-1987/rousseeuw-silhouettes-jcam-sciencedirectopenarchiv.pdf)). |
| Explicit **abstain** status | Stability does not establish validity, and an unstable clustering must not be presented as a phenotype finding. |
| HDBSCAN from scikit-learn's own tree, degrading to DBSCAN | Density-based coverage ([Campello et al.](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)) without dependency churn. Consensus clustering ([Monti et al.](https://link.springer.com/article/10.1023/A:1023949509487)) is deferred, not claimed. |
| Post-hoc label use only, suppressed below 50 cases per class | Labels may characterise a phenotype; they may never define it, and small-count prevalence is not reportable. |

Current outcome on the corrected file: **`no_stable_clusters`** in both the all-adults and
complete-case analyses, because the strongest recoverable structure is survey cycle. See
[`CLUSTERING.md`](CLUSTERING.md).

### Data reliability became a gate

`api/data_reliability.py` produces a structured report (provenance, fingerprints, schema,
unit/range plausibility, duplicates, coverage, missingness by split/cycle/subgroup,
assay-cycle drift, label confidence, leakage, survey-weight applicability, capability state)
and assigns every model input to `usable_now`, `qualified_use`, `unavailable` or
`prohibited`. Hard violations raise, so no analysis can run on data that failed review.
Plausibility windows are project-set review thresholds for catching encoding and sentinel
errors, explicitly not clinical reference intervals.

## Remaining longitudinal blocker

The intended default horizons are 1, 3 and 5 years with a minimum of 50 events and
50 non-events per horizon. `data_integrity.horizon_gate_report` evaluates that gate
and currently reports zero events for every horizon, because:

- NHANES here is a repeated cross-section: one observation per participant, no
  event time and no follow-up indicator;
- TCGA-CDR follow-up exists but is measured **after** diagnosis, so it is denylisted
  from prevention scoring by column prefix.

Until a linked incident-outcome cohort is ingested, deviation scores and latent
representations are the only defensible outputs. No modelling change can substitute
for that data.
