# MetaboGuard Column Dictionary

This dictionary covers identifiers, outcomes, survey-design fields and every
feature used by the public clinical-only model. The machine-readable equivalent
is `model_artifacts/huggingface/metaboguard-risk-xgboost/feature_schema.json`.

## Outcomes and identifiers

| Column | Meaning | Values / units | Is higher better? |
| --- | --- | --- | --- |
| `global_participant_id` | Stable pooled identifier | `survey_cycle:SEQN` | Not applicable |
| `SEQN` | NHANES participant identifier inside a cycle | Integer | Not applicable |
| `survey_cycle` | NHANES collection period | Text, e.g. `2017-March2020` | Not applicable |
| `survey_cycle_index` | Ordered cycle number | 0-9 | Not a health measure |
| `PancreaticCancer` | Corrected target derived from MCQ230A-D code 29 | 1 positive, 0 negative | Not applicable |
| `NODM_PancreaticCancer` | Pancreatic cancer diagnosed 0-3 years after diabetes | 1 positive, 0 diabetic control, missing when timing unknown | Not applicable |
| `pancreatic_cancer_diagnosis_age` | Age when pancreatic cancer was diagnosed | Years; MCQ240T where available | Not applicable |
| `pancreatic_cancer_minus_diabetes_years` | Cancer diagnosis age minus diabetes diagnosis age | Years | 0-3 defines NODM-PC |
| `Cancer` | Self-reported history of any cancer | 1 yes, 0 no | Not applicable |
| `Diabetes` | Self-reported diagnosed diabetes | 1 yes, 0 no | Not applicable |

## Survey design

| Column | Meaning | Use |
| --- | --- | --- |
| `DEMO_SDMVPSU` | Masked variance pseudo-primary sampling unit | Survey variance estimation |
| `DEMO_SDMVSTRA` | Masked variance stratum | Survey variance estimation |
| `combined_mec_weight_1999_2020` | Combined examination weight | Descriptive examined-sample population estimates only |
| `DEMO_WTMEC2YR` | Original two-year examination weight | Cycle-specific analyses |
| `DEMO_WTMEC4YR` | NCHS 1999-2002 bridge weight | Required for the first two cycles |
| `DEMO_WTMECPRP` | 2017-March 2020 pre-pandemic weight | Pre-pandemic cycle |

The combined MEC weight is not a fasting-subsample weight and is not used to
train the prediction model.

## Demographics and body measurements

| Column | Meaning | Values / units | Direction |
| --- | --- | --- | --- |
| `DEMO_RIDAGEYR` | Age | Years | Higher age was associated with higher risk; not inherently “worse” |
| `DEMO_RIAGENDR` | NHANES sex code | 1 male, 2 female | Categorical; higher is meaningless |
| `DEMO_RIDRETH3` | Race/ethnicity category | NHANES code | Categorical; higher is meaningless |
| `BMX_BMXBMI` | Body mass index | kg/m² | Higher generally means more adiposity; relationship may be nonlinear |
| `BMX_BMXWAIST` | Waist circumference | cm | Higher generally means greater central adiposity |
| `Obesity` | Derived BMI threshold | 1 if BMI ≥30 | 1 indicates obesity |

## Diabetes history

| Column | Meaning | Values / units | Direction |
| --- | --- | --- | --- |
| `DIQ_DID040` | Reported age at diabetes diagnosis | Years | Lower means earlier onset |
| `DIQ_DIQ160` | Prediabetes questionnaire response | NHANES code | Categorical |
| `DIQ_DIQ170` | Diabetes-risk questionnaire response | NHANES code | Categorical |
| `DIQ_DIQ180` | Diabetes-risk questionnaire response | NHANES code | Categorical |
| `diabetes_duration_years` | Current age minus diagnosis age | Years | Risk is duration-dependent, not simply higher/lower |
| `recent_diabetes_onset` | Diabetes duration ≤3 years | 1 yes, 0 no | 1 is a recognised risk-enrichment signal |
| `new_onset_diabetes` | Alias of recent-onset flag | 1 yes, 0 no | 1 indicates recent onset |
| `diabetes_subtype` | Exploratory subtype proxy | 0 non-diabetic, 1 Type-1-like, 2 Type-2-like | Categorical; unvalidated heuristic |

## Glycaemic and metabolic laboratories

| Column | Meaning | Values / units | Direction |
| --- | --- | --- | --- |
| `GHB_LBXGH` | Glycated haemoglobin, HbA1c | Percent | Higher means poorer average glycaemic control |
| `GLU_LBXGLU` | Fasting plasma glucose | mg/dL | Higher means higher fasting glucose |
| `INS_LBXIN` | Fasting insulin | µU/mL | Interpret with glucose; higher is not automatically better/worse |
| `CPEP_LBXCPSI` | Fasting C-peptide | SI units | Higher means more endogenous insulin secretion; early cycles only |
| `homa_ir` | Glucose × insulin / 405 | Derived index | Higher suggests greater insulin resistance |
| `elevated_hba1c` | HbA1c ≥6.5% | 1 yes, 0 no | 1 indicates diabetic-range HbA1c |
| `fasting_hyperglycemia` | Fasting glucose ≥126 mg/dL | 1 yes, 0 no | 1 indicates diabetic-range glucose |

## Lipids and inflammation

| Column | Meaning | Values / units | Direction |
| --- | --- | --- | --- |
| `TRIGLY_LBXTR` | Triglycerides | mg/dL | Higher is generally metabolically adverse |
| `TRIGLY_LBDLDL` | LDL cholesterol | mg/dL | Higher is generally metabolically adverse |
| `HDL_LBDHDD` | HDL cholesterol | mg/dL | Higher is generally metabolically favourable |
| `TCHOL_LBXTC` | Total cholesterol | mg/dL | No universal “better” direction in this model |
| `HSCRP_LBXHSCRP` | High-sensitivity C-reactive protein | mg/L | Higher indicates inflammation; pancreatic-risk evidence is weak |

## Paper-supported Priority A additions

| Column | Meaning | Values / units | Direction |
| --- | --- | --- | --- |
| `smoking_status` | Harmonised smoking category | 0 never, 1 former, 2 current | Categorical; current smoking was higher risk in the cited study |
| `current_smoker` | Current smoking flag | 1 yes, 0 no | 1 means current smoking |
| `alcohol_status` | Harmonised alcohol category | 0 never/low, 1 ever, 2 current quantified use | Categorical |
| `average_drinks_per_day` | Drinks on drinking days | Drinks/day | Higher means greater intake |
| `CBC_LBXHGB` | Haemoglobin concentration | g/dL | No simple pancreatic-risk direction |
| `CBC_LBXPLTSI` | Platelet count | 10³ cells/µL | No simple pancreatic-risk direction |
| `BIOPRO_LBXSATSI` | Alanine aminotransferase, ALT | U/L | Higher may indicate liver injury |
| `BIOPRO_LBXSAPSI` | Alkaline phosphatase | U/L | Higher may indicate hepatobiliary or bone processes |
| `BIOPRO_LBXSCR` | Serum creatinine | mg/dL | Higher generally indicates lower renal filtration |
| `hba1c_reciprocal_100` | 100 divided by HbA1c | Derived | Lower corresponds to higher HbA1c |
| `hba1c_squared` | HbA1c squared | Derived | Nonlinear sensitivity term |

## Weight history

| Column | Meaning | Formula / values | Direction |
| --- | --- | --- | --- |
| `weight_loss_1yr_lb` | Reported one-year weight change | Weight 1 year ago − current weight | Positive means weight loss |
| `significant_weight_loss_flag` | Recent loss of at least 10 lb | 1 yes, 0 no | 1 indicates substantial recent loss |
| `weight_loss_10yr_lb` | Reported ten-year weight change | Weight 10 years ago − current weight | Positive means long-term weight loss |

These variables do not distinguish intentional from unintentional weight loss.

## Interaction and trajectory-proxy features

| Column | Formula | Interpretation |
| --- | --- | --- |
| `age_bmi_interaction` | Age × BMI | Combined age/adiposity pattern |
| `waist_bmi_interaction` | Waist × BMI | Combined central/general adiposity |
| `hba1c_age_interaction` | HbA1c × age | Whether glycaemic signal differs with age |
| `hba1c_diabetes_duration_interaction` | HbA1c × diabetes duration | Whether HbA1c signal differs by duration |
| `hba1c_weight_loss_interaction` | HbA1c × one-year weight loss | Combined glycaemia/weight-change pattern |
| `hba1c_cycle_age_sex_z` | HbA1c z-score within cycle, age band and sex | Cohort-relative value; excluded from the public clinical-only model |

For interaction terms, a larger value is not automatically better or worse.
Importance means the combination aided ranking.

## TCGA-only columns

| Column | Meaning | Model role |
| --- | --- | --- |
| `tcga_cancer_type` | TCGA tumour abbreviation | Prognosis models only |
| `tcga_stage_ordinal` | Encoded AJCC stage | Prognosis feature |
| `tcga_grade_ordinal` | Encoded histological grade | Prognosis feature |
| `tcga_tumor_status` | With tumour vs tumour-free | Prognosis feature |
| `tcga_treatment_response` | First-course response encoding | Post-treatment prognosis feature |
| `tcga_followup_days`, `tcga_event` | Overall-survival metadata | **Excluded from mortality features to prevent leakage** |
| `tcga_pfi_days`, `tcga_pfi_event` | Progression metadata | **Excluded from progression features to prevent leakage** |

## Missing values

Model features are converted to numeric values and filled with the training-set
median. Each temporal fold learns medians without accessing its held-out cycle.
The final public artifact stores its medians in `feature_schema.json` and
`model.joblib`.

Median imputation does not mean a missing test is clinically normal. Missingness
often reflects which NHANES cycle or subsample measured the biomarker.

## Model-input allowlist (prevention / deviation work)

Defined once in `api/self_supervised.py::PREVENTION_FEATURES` and validated by
`api/data_integrity.py`. Only these columns may enter the encoder:

`DEMO_RIDAGEYR`, `DEMO_RIAGENDR`, `DEMO_RIDRETH3`, `BMX_BMXBMI`, `BMX_BMXWAIST`,
`GHB_LBXGH`, `GLU_LBXGLU`, `INS_LBXIN`, `CPEP_LBXCPSI`, `TRIGLY_LBXTR`,
`TRIGLY_LBDLDL`, `HDL_LBDHDD`, `TCHOL_LBXTC`, `HSCRP_LBXHSCRP`, `CBC_LBXHGB`,
`CBC_LBXPLTSI`, `BIOPRO_LBXSATSI`, `BIOPRO_LBXSAPSI`, `BIOPRO_LBXSCR`,
`smoking_status`, `alcohol_status`, `average_drinks_per_day`, `weight_loss_1yr_lb`,
`weight_loss_10yr_lb`, `homa_ir`.

Categorical members (`DEMO_RIAGENDR`, `DEMO_RIDRETH3`, `smoking_status`,
`alcohol_status`) are one-hot encoded; numeric members are median-imputed with a
missingness indicator and robust-scaled on the 10–90 percentile range. All statistics
are fit on the training partition only.

## Model-input denylist (never an input)

| Column / pattern | Why it is denylisted |
| --- | --- |
| `Cancer`, `MCQ_MCQ220`, `MCQ_MCQ230A–D`, `MCQ_MCQ240T` | Outcome and outcome-source columns |
| `PancreaticCancer`, `NODM_PancreaticCancer`, `pancreatic_cancer_diagnosis_age`, `pancreatic_cancer_minus_diabetes_years`, `same_year_diabetes_pancreatic_cancer` | Label-derived; also invalidated targets |
| `Diabetes`, `DIQ_DIQ010`, `diabetes_subtype`, `new_onset_diabetes` | Outcome / label-derived |
| any `tcga_*` column | Post-diagnosis context (stage, grade, tumour status, treatment response, follow-up time) that cannot inform prevention scoring |

`data_integrity.is_denylisted_input()` enforces both the explicit list and the
`tcga_` prefix rule, and `select_prevention_features()` asserts the resulting feature
set is disjoint from it.

## Label definitions (as implemented)

| Label | Definition | Nature |
| --- | --- | --- |
| `Cancer` | `MCQ220 == 1` (ever told had cancer) | Prevalent, self-reported, cross-sectional |
| `PancreaticCancer` | any of `MCQ230A–D == 29` (**Pancreas**). Code **39 is "Other"** and is never counted | Prevalent; 19 cases in 107,622 rows |
| `Diabetes` | `DIQ010 == 1` | Prevalent, self-reported |
| `diabetes_subtype` | `1` = research-only Type 1 proxy (young onset + insulin), `2` = Type 2 proxy, `0` = no diabetes | Proxy only: no autoantibodies, no approved genetics, no confirmatory C-peptide criteria |

## Feature eligibility tiers (generated, 2026-08-04)

`api/data_reliability.py` assigns every candidate column a tier. Regenerate with
`python data_reliability.py --output ../model_artifacts/reports/data_reliability.json`.

| Tier | Columns |
| --- | --- |
| `usable_now` | `DEMO_RIDAGEYR`, `DEMO_RIAGENDR`, `BMX_BMXBMI`, `BMX_BMXWAIST`, `GHB_LBXGH`, `TRIGLY_LBDLDL`*, `HDL_LBDHDD`, `TCHOL_LBXTC`, `CBC_LBXHGB`, `CBC_LBXPLTSI`, `BIOPRO_LBXSATSI`, `BIOPRO_LBXSAPSI`, `BIOPRO_LBXSCR`, `smoking_status`, `alcohol_status`, `average_drinks_per_day`, `weight_loss_1yr_lb`, `weight_loss_10yr_lb` (17 columns in the current file) |
| `qualified_use` | `CPEP_LBXCPSI`, `GLU_LBXGLU`, `INS_LBXIN`, `HSCRP_LBXHSCRP`, `TRIGLY_LBXTR`, `TRIGLY_LBDLDL`, `homa_ir`, `DEMO_RIDRETH3` |
| `unavailable` | none in the current file |
| `prohibited` | all outcome labels, label-derived columns and every `tcga_*` column (16 columns) |

\* tier membership is regenerated per dataset; the authoritative list is the JSON report.

Reasons attached to the `qualified_use` columns in the current file:

| Column | Caveat |
| --- | --- |
| `CPEP_LBXCPSI` | Fasting subsample; measured in a minority of cycles (cycle availability gap). |
| `HSCRP_LBXHSCRP` | Cycle availability gap; catalogued evidence shows **no consistent association** with pancreatic cancer, so treat as a known-weak feature ([Bao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3495286/)). |
| `GLU_LBXGLU`, `INS_LBXIN`, `homa_ir` | Fasting subsample: coverage below the 50% threshold. |
| `TRIGLY_LBXTR`, `TRIGLY_LBDLDL` | Fasting subsample coverage, plus a declared lipid evidence gap in the catalogue. |
| `DEMO_RIDRETH3` | Category definitions change across cycles; not comparable as a continuous level. |

## Columns with catalogued published evidence

Each mapping points at a row in `data/evidence/biomarker_evidence.json`; read the row's
`stage_or_lead_time` and `limitations` before quoting anything.

| Column | Evidence row | One-line takeaway |
| --- | --- | --- |
| `GHB_LBXGH` | `ev-hba1c-panc-epic` | Association strongest within 2 years of diabetes diagnosis. |
| `CPEP_LBXCPSI` | `ev-cpeptide-panc-michaud` | Nonfasting C-peptide associated; fasting insulin was null. |
| `homa_ir` | `ev-insulin-resistance-panc-toledo` | Review-level association, mortality-based attributable risk. |
| `weight_loss_1yr_lb` | `ev-weightloss-panc-casecontrol`, `ev-recent-diabetes-weightloss-pdac-jamaoncol-2020` | Strong relative association; absolute 4-year incidence still 0.29%. |
| `BMX_BMXBMI` | `ev-adiposity-panc-pooled`, `ev-excess-body-fatness-multisite-nejm-2016` | IARC sufficient evidence for risk; no early-detection use; weight-loss causality unestablished. |
| `HSCRP_LBXHSCRP` | `ev-hscrp-panc-bao-null` | Explicit negative result. |
| `CBC_LBXPLTSI` | `ev-thrombocytosis-multisite-bjgp-2017` | Non-specific multi-site risk marker; does not indicate site. |
| `TRIGLY_LBXTR` | `ev-lipids-panc-gap` | Declared evidence gap: model it, do not claim it. |

## Identifiers, splits and capability columns

| Column | Meaning |
| --- | --- |
| `SEQN` | NHANES respondent sequence number, unique within a cycle |
| `global_participant_id` | `"<cycle>:<SEQN>"`, the participant key used for grouped splitting; 107,622 unique values, no duplicates |
| `survey_cycle`, `survey_cycle_index`, `survey_year_midpoint` | Pooling metadata; `survey_cycle_index` is a repeated-cross-section proxy, not a within-patient trajectory |
| *(absent)* `event_time_days`, `event` | Required for horizon gating. Their absence is why 1/3/5-year heads are disabled |
