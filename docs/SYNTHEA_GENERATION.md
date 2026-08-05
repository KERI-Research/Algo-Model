# Synthea generation guide (simulation-only longitudinal cohort)

MetaboGuard's preferred synthetic longitudinal source is official
[synthetichealth/synthea](https://github.com/synthetichealth/synthea) (Apache-2.0). Synthea
generates complete synthetic patient records with encounters, conditions, observations and
deaths, which is what a future-risk pipeline needs and what a single NHANES cross-section
cannot provide.

- Methodology: Walonoski et al., *JAMIA* — [Synthea: An approach, method, and software mechanism for generating synthetic patients](https://pmc.ncbi.nlm.nih.gov/articles/PMC7651916/), DOI 10.1093/jamia/ocx079.
- Validity limitations (read before quoting any number produced here): Chen et al., *BMC Medical Informatics and Decision Making* — [Validation of a synthetic data set](https://pmc.ncbi.nlm.nih.gov/articles/PMC6416981/), DOI 10.1186/s12911-019-0793-0.

**Synthetic outputs are for software and model engineering only.** They are not evidence of
clinical performance, they cannot establish real-world calibration, and nothing derived from
them may be presented as a patient's risk.

## Pinned generation environment

| Item | Value |
| --- | --- |
| Release | Synthea `v3.3.0`, official prebuilt `synthea-with-dependencies.jar` |
| Download | `https://github.com/synthetichealth/synthea/releases/download/v3.3.0/synthea-with-dependencies.jar` |
| Jar sha256 | `8ba04f7d73abadd5a377e41edf24c5c83935a1cb07c6d982cd5db731ef1cf445` |
| Java runtime used | OpenJDK `25.0.3+9-2-26.04.2-Ubuntu` (Linux build host) |
| Seeds | `-s 20260805 -cs 20260805 -r 20260805` (population, clinician, referral) |
| Population module state | Massachusetts |
| Age filter | `-a 50-90` (adult prevention population) |
| Recorded run size | 1,818 patients converted (generation stopped early on a disk limit; see below) |
| Exporters | CSV only (`--exporter.csv.export true`, all FHIR exporters off) |

The prebuilt release jar is used deliberately rather than a Gradle build: it removes the
toolchain-compatibility question entirely (Synthea's Gradle wrapper does not target Java 25),
and the release artefact is a fixed, hash-verifiable input. Recording the jar hash plus the
three seeds is what makes the run reproducible, not the build path.

### Exact command

```bash
# 1. Fetch and verify the pinned release jar (once).
curl -L -o vendor/synthea-with-dependencies.jar \
  https://github.com/synthetichealth/synthea/releases/download/v3.3.0/synthea-with-dependencies.jar
shasum -a 256 vendor/synthea-with-dependencies.jar
# expect 8ba04f7d73abadd5a377e41edf24c5c83935a1cb07c6d982cd5db731ef1cf445

# 2. Generate.
java -Xmx3g -jar vendor/synthea-with-dependencies.jar \
  -s 20260805 -cs 20260805 -r 20260805 \
  -p 4000 -a 50-90 \
  --exporter.csv.export true \
  --exporter.fhir.export false \
  --exporter.hospital.fhir.export false \
  --exporter.practitioner.fhir.export false \
  --exporter.baseDirectory ./synthea_out \
  Massachusetts

# 3. Convert into the MetaboGuard longitudinal event schema.
cd api
python synthetic_longitudinal.py --generator synthea \
  --synthea-csv-dir ../synthea_out/csv \
  --output-dir ../data/synthetic_longitudinal/generated_synthea

# 4. Apply the endpoint protocol and build splits.
python longitudinal_dataset.py \
  --events ../data/synthetic_longitudinal/generated_synthea/patient_events.csv \
  --output-dir ../data/synthetic_longitudinal/generated_synthea/prepared
```

On a Mac without a working JRE, `python synthetic_longitudinal.py --print-synthea-status`
reports exactly why Synthea cannot run and how to enable it. The pipeline then falls back to
the in-repo declared simulator and records `used_generator: metaboguard_simulator` in the
manifest, so provenance is never ambiguous.

## Mapping into the MetaboGuard schema

`api/synthetic_longitudinal.py` performs the conversion.

**Observations** (LOINC → MetaboGuard feature code): `4548-4` → `GHB_LBXGH` (%),
`2339-0` → `GLU_LBXGLU` (mg/dL), `2093-3` → `TCHOL_LBXTC`, `2085-9` → `HDL_LBDHDD`,
`2571-8` → `TRIGLY_LBXTR`, `39156-5` → `BMX_BMXBMI`, `29463-7` → `BMX_BMXWT`,
`8480-6` → `BPX_SYSTOLIC`, `56086-2` → `BMX_BMXWAIST`.

**Age** is not an observation in Synthea, so `DEMO_RIDAGEYR` is derived per visit from
`patients.csv` `BIRTHDATE`. Only prevention-safe features are emitted; no treatment,
medication, encounter-cost or care-utilisation column becomes a model input.

**Outcomes** (SNOMED CT → MetaboGuard outcome): `44054006` → `type2_diabetes`;
`254637007`, `254632001` (lung), `363406005`, `93761005` (colorectal), `254837009` (breast),
`126906006` (prostate) → `pan_cancer` composite with the site retained for post-hoc
description only. Death comes from `patients.csv` `DEATHDATE` and is handled as a **competing
event**, never as an ordinary negative. Type 1 diabetes is not generated or mapped: it stays
research-only until autoantibodies, appropriate C-peptide and approved genetics exist.

## Index date rule

`assign_synthea_index_dates` picks, per patient, the **latest** visit that satisfies all of:

1. it falls on or before a fixed calendar cutoff of 1825 days before the end of the data window,
2. at least 365 days of prior observation history,
3. at least 2 prior visits,
4. age at index at least 50 years.

The calendar cutoff guarantees the full 5-year horizon lies inside the data window, so horizon
labels are not systematically censored. The age floor defines an adult prevention population
instead of indexing on a childhood visit, where these outcomes barely occur. The rule depends
only on the calendar and the patient's own pre-index history — never on outcome or death dates —
so it cannot leak future information. Censoring date is the end of record, or the death date
when Synthea records one. Patients who never satisfy the rule get no index date and are dropped
as ineligible rather than silently retained.

## Event yield and the 50-event gate

Synthea reproduces ordinary incidence from its disease modules. That is the honest behaviour we
want, and it has a direct consequence: **incident events inside a 1-year horizon are rare**, so
small populations cannot clear MetaboGuard's safety gate of 50 events and 50 non-events per
horizon. Measured yields for the recorded run (1,818 converted patients, 1,371 eligible after
prevalent-disease exclusion) are:

| Outcome | 1-year events | 3-year events | 5-year events | Gate (50 events) |
| --- | --- | --- | --- | --- |
| `type2_diabetes` | 0 | 18 | 24 | fails at every horizon |
| `pan_cancer` | 8 | 18 | 29 | fails at every horizon |

Per-site cancer counts were lung 2, colorectal 12, breast 10, prostate 3, so every
site-specific head stays disabled. Non-event counts are ample (825-1,153 per horizon); events
are the binding constraint. Running the pipeline on this cohort therefore trains **nothing**:
`run_future_risk_pipeline.py` reports `outcomes_trained: []` and records the failing gate for
each horizon rather than producing a number. That refusal is the intended behaviour and was
verified on this exact cohort.

Extrapolating linearly, roughly 7,500-9,000 patients would clear the 50-event gate at 5 years,
and the 1-year horizon would need on the order of 45,000 patients for cancer and more than that
for type 2 diabetes, whose 1-year incident count here is zero. Full per-horizon numbers are in
`data/synthetic_longitudinal/synthea/prepared_dataset_validation_report.json`. Two honest routes exist when a gate
fails:

1. **Scale the population.** Re-run with a larger `-p` (for example `-p 25000`), which costs
   proportionally more wall-clock time and disk. Nothing else changes and incidence stays
   ordinary. Throughput on the recorded 2-core build host was roughly 100 patients/minute, and
   the raw CSV export is roughly 1.3 GB per 1,000 patients before trimming, so plan disk before
   starting: the recorded run was stopped at 1,818 patients when the export filled the volume.
   Disable the claims, imaging and procedure exporters, or trim them immediately after the run,
   to keep only `patients.csv`, `observations.csv`, `conditions.csv` and `encounters.csv`.
2. **Declared enrichment.** Sample an outcome-enriched analysis cohort with explicit strata and
   retained sampling weights, exactly as the in-repo simulator does. This must be declared in
   the manifest, and raw predicted probabilities are then **not population-calibrated**.

Silently inflating incidence is prohibited. If a gate is not met, the pipeline refuses to train
that horizon instead of reporting a number.

## Timeline defect found and fixed by the fail-closed checks

The first Synthea conversion produced negative times-to-event for 16 patients, and
`longitudinal_dataset.py` refused to proceed. Cause: Synthea can export observations dated at
or after a death encounter, so a naive "latest qualifying visit" index rule could place the
index date after the end of follow-up. `assign_synthea_index_dates` now resolves censoring
first and rejects any candidate index visit at or after the censoring date, and the affected
patients are dropped as ineligible rather than silently kept. This is recorded because it is
the kind of defect that would otherwise silently inflate apparent performance.

## What Synthea cannot establish

Synthea's patients are generated from module rules, not sampled from a real population, and its
own validation paper reports limited agreement with real-world data
([DOI 10.1186/s12911-019-0793-0](https://pmc.ncbi.nlm.nih.gov/articles/PMC6416981/)).
Therefore:

1. No calibration measured here transfers to real patients. External validations of real risk
   models routinely find substantial over-prediction on new populations
   ([BMJ, DOI 10.1136/bmj.e5900](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445426/)).
2. No discrimination number here supports an early-detection claim.
3. The clinical future-risk endpoint stays HTTP 409 regardless of any artefact trained here.

The next step is replacing Synthea with a real linked cohort — UK Biobank, CPRD or an equivalent
EHR extract with dated incident outcomes — and re-running the identical protocol unchanged. See
`docs/FUTURE_RISK_PROTOCOL.md`.