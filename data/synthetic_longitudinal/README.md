# Synthetic longitudinal cohort (simulation only)

Everything in this directory is **synthetic**. It exists to engineer and verify the
future-risk pipeline: schema, endpoint definitions, eligibility masks, competing-risk
handling, calibration and evaluation code paths. It is not evidence of clinical
performance, and no output derived from it may be shown as a patient's risk.

## Preferred generator: Synthea

The preferred source is official [synthetichealth/synthea](https://github.com/synthetichealth/synthea)
(Apache-2.0). Methodology: [Walonoski et al., JAMIA](https://pmc.ncbi.nlm.nih.gov/articles/PMC7651916/),
DOI 10.1093/jamia/ocx079. Known validity limitations (important, read before using any
number from here): [Chen et al., BMC Med Inform Decis Mak](https://pmc.ncbi.nlm.nih.gov/articles/PMC6416981/),
DOI 10.1186/s12911-019-0793-0.

Synthea requires a Java runtime. See `docs/SYNTHEA_GENERATION.md` for the pinned version,
seed, jar location and enable steps. When Java is unavailable the pipeline falls back to the
in-repo declared simulator and records `generator: metaboguard_simulator` in the manifest, so
the provenance of every row stays explicit.

## Committed files

| File | What it is |
| --- | --- |
| `fixture_patient_events.csv` | Small deterministic fixture (seed 20260805) used by tests. |
| `dataset_manifest.json` | Generator, version, seed, strata, weights, fingerprints, row counts. |
| `event_validation_report.json` | Strict schema/plausibility validation of the generated events. |
| `prepared_cohort_report.json` | Index dates, exclusions, washout, censoring, per-site gates. |
| `prepared_dataset_validation_report.json` | Per-horizon event gates and timeline checks. |
| `prepared_split_manifest.json` | Patient-level and temporal split sizes and fingerprints. |

Large generator output (`patient_events.csv`, `prepared/`) is **not committed**. It is
regenerated deterministically:

```bash
cd api
python synthetic_longitudinal.py --generator simulator --patients 6000 \
  --output-dir ../data/synthetic_longitudinal/generated
python longitudinal_dataset.py \
  --events ../data/synthetic_longitudinal/generated/patient_events.csv \
  --output-dir ../data/synthetic_longitudinal/generated/prepared
python run_future_risk_pipeline.py \
  --prepared ../data/synthetic_longitudinal/generated/prepared
```

## Event enrichment is declared, not hidden

The simulator samples from three explicit strata and stores sampling weights in
`patient_strata.csv` and the manifest: `population_typical` (0.55, weight 1.00),
`metabolic_high_risk` (0.30, weight 0.18) and `older_high_risk` (0.15, weight 0.12).
Event rates are therefore **higher than ordinary incidence by design**, so raw predicted
probabilities are not population-calibrated and absolute risks from this cohort are
meaningless outside it.

## What this cohort cannot do

1. It cannot establish real-world calibration, discrimination or clinical utility.
2. It cannot support any early-detection claim.
3. It cannot unlock the clinical future-risk endpoint, which stays HTTP 409 by capability gate.
4. Type 1 diabetes stays research-only; site-specific cancer heads stay disabled for clinical use.

Replacing this with a real cohort (UK Biobank, CPRD or an equivalent linked EHR extract with
dated incident outcomes) is the only route to meaningful performance numbers. See
`docs/FUTURE_RISK_PROTOCOL.md`.