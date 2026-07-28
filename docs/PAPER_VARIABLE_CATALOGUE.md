# MetaboGuard Paper-Derived Variable Catalogue

## Purpose

This catalogue converts two recent new-onset diabetes and pancreatic-cancer
studies into an actionable feature roadmap for MetaboGuard:

- A genetic plus clinical Cox model using UK Biobank data
  ([Zhou et al., 2025](https://doi.org/10.1186/s12916-025-04048-4)).
- A CatBoost clinical model followed by proteomic and metabolomic validation
  ([Yang et al., 2026](https://doi.org/10.1186/s12967-026-07767-1)).

The papers study pancreatic cancer arising within three years of new-onset
type-2 diabetes. This is more specific than MetaboGuard's current NHANES target,
which is self-reported prevalent pancreatic cancer among all labelled
diabetics.

## Highest-priority findings

1. **New-onset timing must remain central.** Both papers define new-onset
   diabetes using a three-year window.
2. **HbA1c should be modelled nonlinearly.** The 2025 study selected a nonlinear
   HbA1c term, not a simple linear coefficient.
3. **Smoking should be added immediately.** It survived the 2025 model
   selection alongside age at diabetes diagnosis and HbA1c.
4. **Routine blood tests can materially expand MetaboGuard.** Haemoglobin, platelet
   count, ALT, creatinine and alkaline phosphatase were candidate predictors
   but are not currently loaded into the pooled dataset.
5. **Body composition and function matter.** The 2026 CatBoost model used
   walking pace, hip circumference, arm/trunk fat mass and disability status.
6. **Genetic and omics variables should form separate model tiers.** They have
   very different availability and cost from routine clinical variables.
7. **PLTP, CRTAC1 and ITGAV are the leading confirmatory serum candidates.**
   They were supported by cross-platform analyses and clinical assays in the
   2026 study.

## Clinical variables from the 2025 genetic model

### Candidate variables

| Paper variable | MetaboGuard mapping | Current status | Recommendation |
|---|---|---|---|
| Age at T2DM diagnosis | `DIQ_DID040` | Available | Keep; one of the final predictors |
| Sex | `DEMO_RIAGENDR` | Available | Keep as categorical |
| BMI | `BMX_BMXBMI` | Available | Keep; test nonlinear effects |
| Smoking status | `smoking_status`, `current_smoker` | Implemented in v2 | 0 never, 1 former, 2 current |
| Glucose | `GLU_LBXGLU` | Available | Keep; fasting-subsample limitation |
| HbA1c | `GHB_LBXGH` | Available | Keep; add nonlinear/spline sensitivity analysis |
| Haemoglobin | `CBC_LBXHGB` | Implemented in v2 | 87,554 non-null |
| Triglycerides | `TRIGLY_LBXTR` | Available | Keep |
| ALT | `BIOPRO_LBXSATSI` | Implemented in v2 | 67,442 non-null |
| Creatinine | `BIOPRO_LBXSCR` | Implemented in v2 | 67,542 non-null |
| Platelet count | `CBC_LBXPLTSI` | Implemented in v2 | 87,552 non-null |
| Total cholesterol | `TCHOL_LBXTC` | Available | Keep |
| Alkaline phosphatase | `BIOPRO_LBXSAPSI` | Implemented in v2 | 67,536 non-null |

The final non-genetic model retained only:

- age at T2DM diagnosis;
- smoking status;
- nonlinear HbA1c.

The best genetic model added the combined NODM-PC polygenic risk score. Its
internal-external C-index was 0.823, compared with 0.749 for the non-genetic
model ([Zhou et al., 2025](https://doi.org/10.1186/s12916-025-04048-4)).

### Desired variables missing from that study

The authors explicitly wanted but could not use:

- weight change;
- glucose change;
- BMI change;
- abdominal pain;
- weight loss;
- jaundice;
- heartburn;
- indigestion;
- nausea.

MetaboGuard already derives one- and ten-year weight change. It does not have true
serial glucose/BMI trajectories or reliable pancreatic-symptom coverage.

## Clinical variables from the 2026 machine-learning model

### Full 22-variable candidate set

| Category | Variables |
|---|---|
| Demographic/lifestyle | age, sex, ethnicity, smoking, alcohol |
| Anthropometric/body composition | BMI, hip circumference, whole-body fat mass, left-leg impedance, right-arm fat mass, left-arm fat mass, trunk fat mass |
| Functional/health status | usual walking pace; long-standing illness, disability or infirmity |
| Haematology | haemoglobin; immature reticulocyte fraction |
| Biochemistry | apolipoprotein A, glucose, HbA1c, total cholesterol, HDL-C, direct LDL-C |

### Final 14-variable CatBoost set

In feature-importance order shown in the paper:

1. HbA1c
2. Total cholesterol
3. Age
4. Apolipoprotein A
5. Usual walking pace
6. Immature reticulocyte fraction
7. Left-arm fat mass
8. BMI
9. HDL cholesterol
10. Long-standing illness, disability or infirmity
11. Hip circumference
12. Trunk fat mass
13. Alcohol status
14. Sex

The model achieved validation AUROC 0.844. The same-day diabetes/cancer
sensitivity analysis reduced AUROC to 0.801, showing the importance of temporal
ordering and reverse-causation controls
([Yang et al., 2026](https://doi.org/10.1186/s12967-026-07767-1)).

## MetaboGuard implementation priorities

### Priority A: add to the pooled NHANES clinical model

These should be addressed first because they are routine and potentially
obtainable from NHANES:

| Variable | Reason | Expected work |
|---|---|---|
| Smoking status | Final predictor in the 2025 model | Add SMQ files across cycles |
| Alcohol status | Included in the 2026 final model | Add ALQ files across cycles |
| Haemoglobin | Used in both candidate sets | Add CBC files and harmonise units |
| Platelet count | 2025 candidate | Add CBC files |
| ALT | 2025 candidate | Add biochemical-profile files |
| Creatinine | 2025 candidate | Add biochemical-profile files |
| Alkaline phosphatase | 2025 candidate | Add biochemical-profile files |
| Nonlinear HbA1c | Significant nonlinear association | Compare spline/reciprocal terms against tree handling |
| Cancer/diabetes diagnosis interval | Needed for true NODM-PC definition | Add age/date-at-cancer-diagnosis fields where available |

### Priority B: audit NHANES availability

Coverage is expected to be incomplete or cycle-specific:

- apolipoprotein A;
- immature reticulocyte fraction;
- hip circumference;
- whole-body, arm and trunk fat mass;
- leg impedance;
- usual walking pace;
- long-standing illness/disability indicator.

Waist circumference is a reasonable adiposity proxy but is not equivalent to
hip circumference or measured regional fat mass.

### Priority C: requires a richer longitudinal/genetic cohort

- exact diabetes diagnosis date;
- incident pancreatic-cancer date and three-year horizon;
- repeated HbA1c, glucose, BMI and weight;
- PC PRS;
- T2DM PRS;
- NODM-PC PRS;
- ancestry principal components;
- abdominal pain, jaundice, dyspepsia/indigestion, heartburn and nausea.

UK Biobank or a comparable linked health-record/genotype cohort is the natural
source. These variables cannot be reconstructed reliably from repeated
cross-sectional NHANES cycles.

## Genetic variables

The 2025 work constructed:

- a 49-SNP pancreatic-cancer PRS;
- a 424-SNP T2DM PRS;
- a combined NODM-PC PRS using 33 SNPs selected from both sets.

The 33 SNP identifiers are:

`rs13303010`, `rs4655617`, `rs41276588`, `rs76263492`, `rs9873618`,
`rs9854769`, `rs3887925`, `rs9854771`, `rs72501964`, `rs56337234`,
`rs2853677`, `rs2736098`, `rs35226131`, `rs2735948`, `rs401681`,
`rs6878122`, `rs12539264`, `rs6971499`, `rs2737226`, `rs2862954`,
`rs4929965`, `rs2237895`, `rs11602873`, `rs28884829`, `rs10844518`,
`rs9543325`, `rs1475655`, `rs12912777`, `rs72802342`, `rs11870735`,
`rs144239147`, `rs10404726`, `rs450960`.

The machine-readable catalogue also stores each risk allele, fitted beta,
p-value and source PRS. These coefficients were estimated in the study cohort
and should not be treated as externally validated MetaboGuard coefficients.

## Proteomic candidates

The 2026 study identified 39 proteins with consistent direction in both the
clinical case-control and model-risk comparisons.

### Upregulated in NODM-PC/high-risk groups

`XPNPEP2`, `NCAN`, `PLA2G7`, **`CRTAC1`**, `ADAMTS8`, **`ITGAV`**,
`KITLG`, `NTRK3`, **`PLTP`**, `MOG`, `DKK3`, `RGMB`, `GPA33`.

### Downregulated

`AMIGO2`, `EPO`, `ICAM1`, `CD22`, `SH2D1A`, `PLXNB2`, `ATF2`, `CDH1`,
`NRP1`, `DEFB4A`, `SIGLEC10`, `SEMA7A`, `ROBO1`, `LGALS4`, `FCER2`,
`LIFR`, `LTA`, `IL18R1`, `FCRL1`, `DKK4`, `FGF23`, `TCL1A`, `ADA2`,
`TNFRSF11A`, `THBS2`, `TNFRSF13B`.

Prioritise **PLTP, CRTAC1 and ITGAV** for a confirmatory biomarker panel because
the paper additionally validated them across Olink, CPTAC, plasma ELISA and
tissue assays ([Yang et al., 2026](https://doi.org/10.1186/s12967-026-07767-1)).

## Metabolomic candidates

The study found 145 metabolites with consistent direction across clinical and
model-defined risk comparisons. They are dominated by:

- VLDL, IDL, LDL and HDL particle concentrations and lipid composition;
- remnant and non-HDL cholesterol;
- triglyceride-to-particle ratios;
- ApoB;
- glucose and glycolysis-related measures;
- branched-chain amino acids: valine, leucine and isoleucine;
- alanine;
- GlycA;
- sphingomyelins and fatty-acid composition.

Five highlighted pathway families were:

- valine, leucine and isoleucine biosynthesis;
- linoleic-acid metabolism;
- glycerophospholipid metabolism;
- alpha-linolenic-acid metabolism;
- cysteine and methionine metabolism.

The full 145-variable list is retained in the machine-readable research
candidate data. These platform-specific NMR features should be evaluated as a
separate metabolomics model, not median-imputed into the routine clinical
model.

## Recommended MetaboGuard model tiers

### MetaboGuard Clinical v2

Routine, scalable variables:

- age and age at diabetes diagnosis;
- sex and ethnicity;
- smoking and alcohol;
- BMI, waist and weight change;
- HbA1c and glucose;
- triglycerides, total cholesterol, HDL and LDL;
- haemoglobin and platelets;
- ALT, creatinine and alkaline phosphatase.

### MetaboGuard Clinical-Body v3

Clinical v2 plus:

- hip circumference;
- arm, trunk and whole-body fat mass;
- leg impedance;
- walking pace;
- disability/long-standing illness;
- apolipoprotein A;
- immature reticulocyte fraction.

### MetaboGuard Genetic

Clinical v2 plus PC, T2DM and/or NODM-PC PRS. Validate by geography or external
cohort and include ancestry principal components.

### MetaboGuard Biomarker Confirm

Used only after clinical high-risk triage:

- PLTP;
- CRTAC1;
- ITGAV;
- broader 39-protein panel;
- targeted NMR lipid/amino-acid panel;
- CA19-9 where available.

## Variables that must not enter an early-detection model

Avoid post-diagnosis leakage from:

- tumour stage or grade;
- tumour status;
- treatment response;
- survival/progression times;
- pathology obtained after cancer diagnosis.

Those variables belong to MetaboGuard's separate prognosis experiments, not the
NODM early-detection model.

## Implementation outcome

Smoking, alcohol, CBC, routine biochemistry and nonlinear HbA1c terms are now
implemented in `nhanes_multicycle_v2.csv`.

The same audit corrected the pancreatic-cancer site code from 39 (Other) to 29
(Pancreas). This reduced the cohort to 19 pancreatic-cancer cases overall, 7
among diabetics and only 2 meeting the exact three-year NODM-PC definition.
Consequently, the next priority is no longer feature expansion: it is acquiring
a sufficiently large linked incident pancreatic-cancer cohort.
