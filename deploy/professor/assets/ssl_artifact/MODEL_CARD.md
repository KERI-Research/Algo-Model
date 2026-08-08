# Model card: metaboguard_ssl_v1 (retrain-wide run)

- **Code version:** metaboguard-ssl-v1.1
- **Created:** 2026-08-08T14:31:01.056625+00:00
- **Backend / device:** torch / cpu
- **Output type:** metabolic_deviation_and_representation
- **Dataset:** nhanes_multicycle_v2.csv (sha256 `af790f9fed1f13179726f02262de296ac9f7bcd85e3369fa2fdb4ea62d6b8b7c`)
- **Rows:** 44128 train / 9456 validation / 9457 holdout
- **Latent dimension:** 16

## What this model does

It learns a label-free representation of adult metabolic features and reports:

1. a **metabolic deviation score** (how unusual a profile is versus the training reference),
2. a **reference percentile** of that score,
3. a **16-dimensional latent representation**.

## What this model does NOT do

- It does not diagnose cancer or diabetes.
- It does not estimate the probability of developing any disease.
- It is not validated for the intended future horizons (1 year (365d), 3 years (1095d), 5 years (1825d)); those heads are
  disabled because no horizon passes the 50-event / 50-non-event safety gate on the
  current cross-sectional data.
- Type 1 diabetes remains research-only: no autoantibodies, no approved genetics and
  no confirmatory C-peptide criteria exist in these files.

## Training

- No outcome label is used during encoder training. Label columns present in the file
  but excluded from inputs: Cancer, Diabetes, NODM_PancreaticCancer, PancreaticCancer, diabetes_subtype, new_onset_diabetes, pancreatic_cancer_diagnosis_age, pancreatic_cancer_minus_diabetes_years, same_year_diabetes_pancreatic_cancer.
- Splits are participant-grouped and seeded (seed 42,
  fractions [0.7, 0.15, 0.15]).
- Preprocessing and deviation reference statistics are fit on the training partition only.
- Final validation reconstruction loss: 0.03679179400205612.
- Holdout reconstruction MSE: 0.03486661899305853.

## Post-hoc association checks (cross-sectional, holdout only)

- `any_cancer_prevalence`: AUROC 0.726, AUPRC 0.198 (837 positives / 7963 negatives) - cross-sectional association only.
- `type2_diabetes_proxy`: AUROC 0.922, AUPRC 0.675 (1075 positives / 8133 negatives) - cross-sectional association only.
- `type1_diabetes_proxy_research_only`: not evaluated (Fewer than 50 cases in one class on the holdout partition.).

These numbers describe association with conditions that are **already present**. They
must never be presented as future-risk performance.

## Known limitations and blocker

The single blocking limitation for future-risk work is data: NHANES here is a repeated
cross-section with one observation per participant and no follow-up, and TCGA is
post-diagnosis. Linked incident-outcome follow-up (for example NHANES-linked mortality
or registry linkage, or an EHR cohort) is required before any horizon-based head can be
trained or reported.
