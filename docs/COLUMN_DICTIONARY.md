# DiaPan Column Dictionary

This dictionary covers identifiers, outcomes, survey-design fields and every
feature used by the public clinical-only model. The machine-readable equivalent
is `model_artifacts/huggingface/diapan-risk-xgboost/feature_schema.json`.

## Outcomes and identifiers

| Column | Meaning | Values / units | Is higher better? |
| --- | --- | --- | --- |
| `global_participant_id` | Stable pooled identifier | `survey_cycle:SEQN` | Not applicable |
| `SEQN` | NHANES participant identifier inside a cycle | Integer | Not applicable |
| `survey_cycle` | NHANES collection period | Text, e.g. `2017-March2020` | Not applicable |
| `survey_cycle_index` | Ordered cycle number | 0-9 | Not a health measure |
| `PancreaticCancer` | Research target derived from MCQ230A-D code 39 | 1 positive, 0 negative | Not applicable |
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
