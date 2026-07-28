# DiaPan Methodology and Research Decision Log

## Study question

DiaPan asks whether combinations of metabolic, clinical and behavioural
variables can stratify pancreatic-cancer risk among people with diabetes.
The project is exploratory research software, not a diagnostic device.

The biological rationale is the bidirectional, duration-dependent association
between diabetes and pancreatic cancer. New-onset diabetes carries materially
higher risk than long-standing diabetes, while risk attenuates with duration
([Lee et al. 2023](https://pubmed.ncbi.nlm.nih.gov/36548964/);
[Song et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4519136/)).
This supports explicit features for age at diabetes diagnosis, diabetes
duration and recent onset rather than a single diabetes yes/no flag.

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
longitudinal trajectories. DiaPan therefore names its engineered time features
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
| Pancreatic-cancer-positive | 318 |
| Labelled diabetics | 7,204 |
| Pancreatic-cancer-positive diabetics | 50 |

The exact source URL, filename, status, row count and retained columns for every
component/cycle are recorded in
`data/nhanes_multicycle_build_report.json`. Missing files are reported rather
than silently converted into apparent biological missingness.

### Harmonisation

Early NHANES cycles bundled laboratory analytes in numbered files:

- `LAB10` / `L10`: glycohaemoglobin
- `LAB10AM` / `L10AM`: fasting glucose, insulin and C-peptide
- `LAB13` / `L13`: total cholesterol, HDL, triglycerides and LDL

Later cycles use named files such as `GHB`, `GLU`, `INS`, `TRIGLY`, `HDL` and
`TCHOL`. The builder maps these variants to one stable DiaPan schema. This is
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

`PancreaticCancer` is derived from NHANES `MCQ230A-D`; code 39 means pancreas.
A participant is positive if any reported cancer slot equals 39. Participants
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
baseline, not the primary clinical framing.

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

## Current pooled results

| Cohort | Rows used | Test positives | AUROC (95% CI) | AUPRC (95% CI) | AUPRC lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| General population | 52,891 | 55 | 0.679 (0.598-0.748) | 0.011 (0.008-0.020) | 2.14x |
| Diabetics only | 6,473 | 8 | 0.641 (0.419-0.830) | 0.135 (0.007-0.386) | 21.81x |

The diabetics-only point estimate is not statistically stable: only eight
positive cases appear in the held-out fold and the confidence interval is
wide. The correct conclusion is **promising enrichment with high uncertainty**,
not validated early detection.

The top diabetics-only features are:

1. `hba1c_age_interaction`
2. `TCHOL_LBXTC`
3. `GHB_LBXGH`
4. `age_bmi_interaction`
5. `survey_cycle_index`

`survey_cycle_index` appearing in the top five signals possible secular drift
or measurement heterogeneity. A cycle-held-out validation is required before
claiming generalisation.

## Cycle-held-out temporal validation

DiaPan now performs leave-one-survey-cycle-out validation through
`api/validate_cycle_holdout.py`. Each cycle is treated as unseen test data in
turn. Imputation medians and models are learned exclusively from the remaining
cycles. This is an internal-external temporal validation design, consistent
with guidance that clinical models should be evaluated in new settings and
that performance heterogeneity should be examined rather than hidden behind
one random split
([BMJ evaluation guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC10772854/);
[Nieboer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5708595/)).

| Variant | Model | AUROC (cycle-bootstrap 95% CI) | AUPRC (95% CI) | Lift |
| --- | --- | ---: | ---: | ---: |
| Clinical only | HistGradientBoosting | 0.635 (0.582-0.685) | 0.0119 (0.0091-0.0246) | 1.88x |
| Clinical only | XGBoost | **0.643 (0.561-0.703)** | 0.0128 (0.0084-0.0289) | 2.03x |
| With cycle proxies | HistGradientBoosting | 0.609 (0.578-0.667) | 0.0115 (0.0095-0.0212) | 1.81x |
| With cycle proxies | XGBoost | 0.630 (0.562-0.685) | **0.0137 (0.0083-0.0394)** | 2.17x |

The out-of-cycle results are substantially weaker than the random 80/20 split.
The random-split AUPRC of 0.135 should therefore be treated as optimistic.
Clinical-only XGBoost offers the best pooled AUROC and avoids dependence on
cycle-derived variables, so it is the preferred research benchmark.

Per-cycle AUROC varies widely (approximately 0.42-0.95 depending on model and
cycle), confirming temporal heterogeneity. Full fold results are stored in
`data/cycle_holdout_validation.json` and rendered in
`docs/CYCLE_HOLDOUT_VALIDATION.md`.

## Leakage controls

DiaPan excludes TCGA `OS`, `OS.time`, `PFI` and `PFI.time` because they define or
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

# General-population baseline
python train_biomarker_model.py \
  --dataset nhanes_multicycle.csv \
  --target PancreaticCancer \
  --force

# Primary brief-aligned model
python train_biomarker_model.py \
  --dataset nhanes_multicycle.csv \
  --target PancreaticCancer \
  --cohort-filter diabetics_only \
  --force
```

## Next validation steps

1. Perform sensitivity analysis with and without the unvalidated
   `diabetes_subtype` proxy.
2. Add survey-design-aware descriptive analysis using the combined MEC weight,
   strata and PSU fields.
3. Derive or ingest fasting-subsample combined weights before making weighted
   claims about insulin, glucose, HOMA-IR or C-peptide.
4. Calibrate only after substantially more events are available; stable
   calibration curves commonly require roughly 200 events and 200 non-events
   ([calibration review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996/)).
5. Seek a longitudinal cohort with serial biomarkers and CA19-9 to answer the
   true early-detection trajectory question.
