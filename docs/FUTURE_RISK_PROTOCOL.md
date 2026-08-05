# Future-risk protocol (simulation only)

This document defines the endpoint, split and gating protocol for MetaboGuard's future-risk
pipeline. Protocol version: `future-risk-protocol-v1`. Seed: `20260805`.

> **SIMULATION ONLY.** Every number produced under this protocol today comes from synthetic
> data. It verifies software and protocol correctness. It is not evidence of clinical
> performance, it establishes no real-world calibration, and no output may be shown as a
> patient's risk. The clinical endpoint `POST /api/v1/prevention-future-risk` returns **HTTP
> 409** and stays that way until an approved real cohort exists.

## Why a new pipeline was needed

The existing MetaboGuard probes are **cross-sectional** NHANES associations: one observation
per participant, outcome status recorded at the same time. That design cannot answer "what
happens next", and presenting it as future risk would be false. TCGA is post-diagnosis context
only. A future-risk question needs dated repeated measurements, a defined prediction origin, and
incident outcomes after that origin — which is what the longitudinal schema provides.

## Capability states

`api/longitudinal_schema.py` defines five states and what each permits:

| State | Future risk permitted? |
| --- | --- |
| `cross_sectional` | no — association probes only |
| `repeated_without_outcomes` | no — trajectories, no labels |
| `simulation_only_longitudinal` | **simulated only**, requires explicit `simulation_mode=true` |
| `longitudinal_with_incident_outcomes` | yes — the only clinical state, unreachable in this repo |
| `post_diagnosis` | no — descriptive context only |

`assert_clinical_future_risk_allowed` raises for every state this repository can actually reach.
`assert_simulated_future_risk_allowed(state, simulation_mode)` requires **both** the
simulation-only state and an explicit flag. Type 1 diabetes and cancer-site outcomes are
permanently disabled via `assert_outcome_allowed`.

## Endpoint definitions

- **Index date (prediction origin).** For simulated cohorts the generator assigns it. For
  Synthea it is the latest visit satisfying: at or before a fixed calendar cutoff 1,825 days
  before the end of the data window; ≥365 days of prior observation history; ≥2 prior visits;
  age ≥50 years; strictly before the censoring date. The rule depends only on the calendar and
  the patient's own pre-index history — never on outcomes or death — so it cannot leak future
  information.
- **Prevalent-disease exclusion.** Any patient with the outcome recorded at or before the index
  date is excluded from that outcome's incident question, and the count is reported.
- **Washout / history requirement.** ≥365 days of pre-index observation and ≥2 pre-index visits.
- **Feature window.** Only observations at or before the index date become model inputs. Nothing
  after the index date is used, for any model.
- **Censoring.** End of record, or death date when earlier. Follow-up = censoring − index.
- **Competing outcomes.** Cause-specific coding: `1` = target outcome, `2` = death before the
  target, `0` = censored. Death is a competing event, never an ordinary negative. Reporting
  follows the competing-risk validation guidance in
  [BMJ 2022, DOI 10.1136/bmj-2021-069249](https://www.bmj.com/lookup/doi/10.1136/bmj-2021-069249).
- **Horizon labels.** 1 year (365 d), 3 years (1,095 d), 5 years (1,825 d). For each horizon a
  patient is **eligible** if either the target event occurred inside the horizon, or the patient
  was followed event-free and alive for the whole horizon. Patients censored before the horizon
  are marked `censored_before_horizon` and are **excluded by mask**, never counted as negatives.
  Every metric, loss and calibration fit uses the eligibility mask.
- **IPCW metadata.** Time-to-event, cause and follow-up are persisted per patient so inverse
  probability of censoring weighting can be added without re-deriving labels.

## Splits

Written by `build_splits` **before any preprocessing is fit**, and fingerprinted:

- Patient-level random split: train 0.70 / validation 0.15 / test 0.15, seed 20260805, grouped
  so no patient appears in two splits (overlaps asserted to be zero).
- Temporal holdout: patients whose index year is ≥ 2016 are held out separately, so a
  chronological check exists alongside the random one.
- Preprocessing statistics (imputation medians, scaling, calibration) are fit on **train only**;
  calibration is fit on **validation**; the test split and temporal holdout are touched once at
  evaluation. This follows the leakage guidance in
  [scikit-learn's common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- `splits.json` and `split_manifest.json` (sizes, overlaps, per-split patient-id fingerprints)
  are persisted and copied into every artifact.

## Safety gates

`MIN_EVENTS_PER_HORIZON = MIN_NON_EVENTS_PER_HORIZON = 50`.

- A horizon trains **only** if it independently has ≥50 events and ≥50 non-events among
  eligible patients. Horizons that fail **abstain**: the pipeline records the failing counts and
  produces no metric and no probability for them.
- Gates are evaluated per outcome and per horizon, so a cohort may enable 3- and 5-year horizons
  while abstaining at 1 year. That is the expected behaviour for rare short-horizon events.
- Site-specific cancer heads stay disabled unless a site independently clears 50 events; the
  pan-cancer composite is the only cancer target.
- Type 1 diabetes is never enabled — it remains research-only until autoantibodies, appropriate
  C-peptide and approved genetics exist.

## Cohort classes: ordinary vs enriched

Two cohort classes exist and their metrics are **never pooled**:

1. **`simulation_ordinary_incidence`** — official Synthea v3.3.0, ordinary module incidence, no
   enrichment or reweighting. Realistic event rates; short horizons often abstain. See
   `docs/SYNTHEA_GENERATION.md`.
2. **`simulation_enriched`** — the declared in-repo simulator. Enrichment is explicit: three
   sampling strata (`population_typical` share 0.55 weight 1.00; `metabolic_high_risk` 0.30 /
   0.18; `older_high_risk` 0.15 / 0.12) with weights and stratum labels retained per patient in
   `patient_strata.csv` and the manifest. **Raw predicted probabilities from this cohort are not
   population-calibrated** and absolute risks are meaningless outside the sample.

Silently inflating incidence is prohibited. If a gate cannot be met, the pipeline abstains
rather than reporting a number.

## Models

Transparent baselines first, in line with published interpretable horizon/survival models such
as the eight-cancer UK Biobank Cox analysis
([DOI 10.1093/jncics/pkae008](https://pmc.ncbi.nlm.nih.gov/articles/PMC10919929/)):

1. **Horizon logistic regression** — median imputation with missingness indicators, scaling,
   balanced class weights, one model per outcome/horizon.
2. **Cause-specific discrete-time hazard** — person-interval expansion (183-day intervals),
   logistic hazard on features plus interval index, a parallel competing-death hazard, and
   cumulative incidence accumulated as `Σ S(t) · h_target(t)` with survival decremented by both
   hazards. Intervals stop at the event, so censoring is handled by construction.
3. **Gradient-boosted trees** — `HistGradientBoostingClassifier`, depth 3, native missing
   handling.
4. **Temporal GRU (experimental)** — masked GRU over irregular visit sequences with explicit
   time deltas and per-feature masks, reading the **last valid hidden state**, one head per
   horizon, masked loss. Sequential-EHR designs of this shape motivate the architecture
   ([DOI 10.1016/j.xcrm.2025.102359](https://linkinghub.elsevier.com/retrieve/pii/S266637912500432X);
   temporal diabetes prediction
   [DOI 10.1371/journal.pdig.0000354](https://pmc.ncbi.nlm.nih.gov/articles/PMC10599553/)) — as
   *design context only*, not as a performance comparator.

The temporal model uses its **own encoder**. The cross-sectional MetaboGuard SSL encoder is
frozen to a single-visit NHANES feature distribution; reusing it for irregular multi-visit
sequences with deltas and masks is not defensible, and that artifact is left untouched.

### Temporal admissibility rule

A temporal model is only admissible if reversing visit order **degrades** test AUROC by at least
0.02. Below that threshold it is not demonstrably using time, and it is recorded as
`experimental_rejected_time_reversal_control` and excluded from selection. A calibrated
discrete-time / logistic / tree baseline being selected as the primary artifact is an acceptable
and scientifically preferable outcome. Temporal superiority is never manufactured.

## Selection

**Calibration-first**: among evaluated models, the smallest |calibration slope − 1| on the test
split wins, ties broken by AUROC. Discrimination never overrides poor calibration — external
validations of real risk models routinely find substantial over-prediction on new populations
([BMJ 2012, DOI 10.1136/bmj.e5900](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445426/)).

## Artifacts

Every run writes a versioned, simulation-only artifact containing: fitted models, calibrators,
feature column list, the longitudinal JSON schema, split IDs, the dataset validation report,
raw **and** calibrated predictions per outcome/horizon/model/split, `results.json`,
`metadata.json` (capability state, `clinical_use: prohibited`, fingerprints, package versions),
`MODEL_CARD.md`, and `reload_parity_report.json` from the reload/scoring parity check.

## Reproduction

```bash
cd api
# Ordinary-incidence Synthea cohort (batched; see docs/SYNTHEA_GENERATION.md)
python synthetic_longitudinal.py --generator synthea --synthea-csv-dir <csv> --output-dir <out>
# Declared-enrichment simulator cohort
python synthetic_longitudinal.py --generator simulator --patients 6000 --output-dir <out>
# Endpoint protocol, features and splits
python longitudinal_dataset.py --events <out>/patient_events.csv --output-dir <out>/prepared
# Full (non-smoke) training, calibration, evaluation and packaging
python run_future_risk_pipeline.py --prepared <out>/prepared --output-dir <artifact_root>
# Artifact reload / scoring parity
python run_future_risk_pipeline.py --verify-artifact <artifact_root>/artifact --prepared <out>/prepared
```

## The remaining real-cohort blocker

No real longitudinal cohort with dated incident outcomes exists in this repository. Until one is
approved and linked — UK Biobank, CPRD, or an equivalent EHR extract — the clinical endpoint
stays 409 and every number in this pipeline is synthetic. Replacing the synthetic cohort requires
no protocol change: the same schema, endpoint definitions, gates, splits, models and evaluation
run unchanged on real data.