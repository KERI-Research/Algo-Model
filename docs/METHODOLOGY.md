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
