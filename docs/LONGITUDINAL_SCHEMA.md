# Longitudinal schema and data dictionary

Schema version: `metaboguard-longitudinal-v1`. Visit matrix version: `metaboguard-visits-v1`.
Defined and enforced in `api/longitudinal_schema.py`. Machine-readable JSON Schemas come from
`json_schemas()` and are copied into every artifact as `longitudinal_schema.json`.

> **SIMULATION ONLY** today. The schema is designed so a real cohort can be dropped in without
> changing the protocol, but no real longitudinal data exists in this repository yet.

## Patient event table (`patient_events.csv`)

One row per observation or outcome. Long format, so irregular visits and per-feature
missingness are represented exactly rather than being padded away.

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Must equal `metaboguard-longitudinal-v1`. |
| `patient_id` | string | Stable pseudonymous id. Never a real identifier. |
| `observation_timestamp` | ISO 8601 UTC | When the value was measured. |
| `source` | enum | Provenance of the row (`synthea`, `metaboguard_simulator`, …). |
| `feature_code` | string | Prevention-safe feature code, or `outcome:<name>` for outcome rows. |
| `value` | float | Value in the feature's canonical unit after harmonisation. |
| `unit` | string | Unit as supplied; harmonised to canonical before modelling. |
| `missingness_reason` | enum | `observed`, `not_measured`, `invalid_value_removed`, … |
| `visit_id` | string | Groups observations taken at the same encounter. |
| `index_date` | ISO 8601 UTC | Prediction origin for this patient. |
| `outcome_type` | enum | `none` for observations; otherwise `type2_diabetes`, `pan_cancer`, `death`. |
| `event_date` | ISO 8601 UTC | Outcome date; required on outcome rows. |
| `cancer_site` | string | Post-hoc description only; never a model target while site heads are disabled. |
| `cancer_stage` | string | Descriptive only. |
| `censoring_date` | ISO 8601 UTC | End of follow-up. |
| `provenance` | string | Generator, version and seed. |

## Prevention-safe features

Only these 11 codes may become model inputs:

| Code | Description | Canonical unit |
| --- | --- | --- |
| `DEMO_RIDAGEYR` | Age | years |
| `BMX_BMXBMI` | Body mass index | kg/m² |
| `BMX_BMXWAIST` | Waist circumference | cm |
| `BMX_BMXWT` | Weight | kg |
| `GHB_LBXGH` | HbA1c | % (DCCT) |
| `GLU_LBXGLU` | Fasting glucose | mg/dL |
| `INS_LBXIN` | Insulin | µU/mL |
| `TCHOL_LBXTC` | Total cholesterol | mg/dL |
| `HDL_LBDHDD` | HDL cholesterol | mg/dL |
| `TRIGLY_LBXTR` | Triglycerides | mg/dL |
| `BPX_SYSTOLIC` | Systolic blood pressure | mmHg |

Anything treatment-, medication-, encounter-cost- or utilisation-derived is denylisted as a
model input: those encode care pathways and would leak outcome information.

## Unit harmonisation

`harmonise_units` converts to canonical units and returns a conversion log. Multiplicative
factors handle simple conversions; HbA1c IFCC (`mmol/mol`) uses the affine NGSP relation
(`% = 0.09148 × mmol/mol + 2.152`), so 53 mmol/mol → 7.00 % and 39 mmol/mol → 5.72 %.
Unknown units are reported, not guessed.

## Validation (fail-closed)

`validate_event_frame(frame, strict=True)`:

- **Raises** on missing schema columns, unknown `source`/`missingness_reason`/`outcome_type`,
  an outcome row without an `event_date`, an event date before the index date, a censoring date
  before the index date, a `cancer_site` on a non-cancer row, a missing value that claims
  `missingness_reason == "observed"`, or a denylisted feature used as an input.
- **Removes and reports** individual implausible values (outside `PLAUSIBLE_VALUE_RANGES`),
  setting `missingness_reason = invalid_value_removed`. If more than 10 % of a feature's values
  are implausible, that becomes a hard issue and strict mode raises.
- Reports per-feature coverage, missingness by reason, outcome counts and cancer sites.

## Visit matrix (`visit_matrix.csv`)

`build_visit_matrix` pivots pre-index observations into one row per patient-visit:

| Column | Meaning |
| --- | --- |
| `patient_id`, `visit_index` | Visit order, 0-based, strictly time-ordered. |
| `relative_time_days` | Signed days from the index date (negative = before index). |
| `delta_days_since_previous_visit` | Irregular gap; 0 on the first visit. |
| `feature_<CODE>` | Harmonised value, `NaN` when not measured at that visit. |
| `mask_<CODE>` | 1 if measured at that visit, else 0. |

The temporal model consumes `[features, masks, delta]` per visit; masks and deltas are inputs,
so missingness and irregular timing are modelled rather than hidden.

## Patient feature table (`patient_features.csv`)

`build_patient_features` summarises each patient's **pre-index** trajectory: per feature
`_last`, `_mean`, `_slope_per_year`, `_delta`, `_observed_count`, plus `visit_count`,
`visit_density_per_year`, `history_days`, `median_visit_gap_days` and `missingness_burden`.
`baseline_feature_columns` filters this to prevention-safe columns and asserts no label,
eligibility, cause, outcome, site, event-date or censoring column can reach a model.

## Label frame (`cohort_labels.csv`)

Per patient and outcome: `<outcome>_date`, `<outcome>_time_days`, `<outcome>_cause`
(1 target / 2 competing death / 0 censored), `follow_up_days`, `index_year`, and per horizon
`<outcome>_<h>_label`, `_eligible`, `_censored_before_horizon`,
`_competing_death_before_horizon`.

## Manifests and fingerprints

`DatasetManifest` records schema version, generator (release, jar hash, runtime, seeds, index
rule, batch list), enrichment declaration, row counts, SHA-256 fingerprints of the events and
fixture frames, horizons, outcomes, disabled outcomes, and the standing simulation-only notes.
`frame_fingerprint` and `file_fingerprint` make any dataset or artifact independently checkable.

## Committed vs regenerated

Committed: small deterministic fixtures plus all manifests and validation reports. Regenerated
on demand and git-ignored: full event tables, visit matrices, feature tables and prepared
splits. Exact commands are in `docs/FUTURE_RISK_PROTOCOL.md` and
`data/synthetic_longitudinal/README.md`.