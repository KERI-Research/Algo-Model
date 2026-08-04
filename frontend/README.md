# MetaboGuard dashboard (frontend)

Create React App interface for the MetaboGuard research API.

```bash
npm install
npm start                     # dev server on http://localhost:3000
npm run build                 # production build
CI=true npm test -- --watchAll=false --runInBand   # unit tests
```

The API base URL comes from `REACT_APP_API_URL` and defaults to `http://localhost:8000`.

## Surfaces

| Panel | What it shows |
| --- | --- |
| Causal / predictive panels | DoWhy estimates and the supervised baseline. `causal` here names the estimation method, not a proven mechanism. |
| **Biomarker Probe** | Cross-sectional association model inputs and output. The output is labelled "cross-sectional association with an already-recorded diagnosis", never future risk. |
| Prevention deviation | Metabolic deviation score, reference percentile and latent representation. |
| Research panel (`interface/research_panel.js`) | Data-reliability tiers, exploratory phenotype clusters or the explicit abstain result, and clinician-ready evidence rows. Each block carries an explanation-class badge: `data observation`, `model association`, `published evidence`, `causal claim not established`. |

## Synthetic patient generator (Biomarker Probe)

`interface/synthetic_patient.js` fills the probe inputs with fabricated, internally coherent
values so the interface can be exercised without typing nine fields or using real patient
data.

- **Button:** `Generate synthetic patient` — a normal `<button type="button">`, keyboard
  reachable in document order, with `aria-label` and `aria-describedby` pointing at the
  note beneath it. It never submits and never calls the scoring API; run the probe
  explicitly afterwards.
- **Indicator:** after each click a `role="status"` region reports **Synthetic research
  data**, the archetype used, the time, how many of the nine fields were populated, which
  were deliberately left blank, and whether the form has been manually edited since
  generation. The indicator is identified by text and a dashed border, never by colour
  alone.
- **Manual editing is preserved.** Generated values are ordinary form state; editing any
  field simply marks the indicator as "manually edited since generation".
- **Every click produces a new profile** (the component uses `Math.random`; the generator
  accepts an injectable source so tests can pin a seed).

### Where the field rules come from

Ranges, units, precision and categorical encodings are derived from the existing form
schema, `api/biomarker.py` (`REQUIRED_FIELDS` = `Diabetes`, `DEMO_RIDAGEYR`,
`DEMO_RIAGENDR`, `BMX_BMXBMI`; `OPTIONAL_HIGH_IMPACT_FIELDS` for the labs),
`docs/COLUMN_DICTIONARY.md`, and the plausibility windows in `api/data_reliability.py`.
Generation ranges are deliberately narrower than those windows, so generated values are
always inside them.

| Field | Kind | Range | Step / precision |
| --- | --- | --- | --- |
| `Diabetes` | categorical | `"0"`, `"1"` | reported condition **input**, not an outcome label |
| `DEMO_RIDAGEYR` | number (years) | 18–85 | 1 / 0 dp |
| `DEMO_RIAGENDR` | categorical | `"1"` male, `"2"` female | — |
| `BMX_BMXBMI` | number (kg/m²) | 16–55 | 0.1 / 1 dp |
| `BMX_BMXWAIST` | number (cm) | 60–170 | 0.1 / 1 dp |
| `DIQ_DID040` | number (age at onset) | 1–85 | 1 / 0 dp, only when diabetes is reported |
| `GHB_LBXGH` | number (% HbA1c) | 3.5–15 | 0.1 / 1 dp |
| `GLU_LBXGLU` | number (mg/dL) | 50–400 | 1 / 0 dp |
| `INS_LBXIN` | number (µU/mL) | 1–150 | 0.1 / 1 dp |

No outcome label (`Cancer`, `PancreaticCancer`, `diabetes_subtype`, any `tcga_*` column) is
ever generated; those are denylisted model inputs and a unit test asserts their absence.

### Archetypes

Measurement patterns for interface testing. **None is a disease label, diagnosis or risk
stratum.**

| Archetype | Pattern |
| --- | --- |
| `reference_range` | All exposed measurements inside commonly reported reference intervals. |
| `metabolic_deviation` | Higher adiposity with dysglycaemic and insulin-resistant values. |
| `reported_diabetes_metabolic` | Reported diabetes with an onset age and correspondingly higher glycaemic values. |
| `sparse_but_valid` | Required fields only; optional laboratory fields blank, exercising the missing-field path the API already supports. Required fields stay populated so the probe remains submittable. |

### Coherence rules

- Waist circumference is derived from BMI with a sex-specific intercept and a small age
  term, then jittered — it is not independent noise.
- Fasting glucose is derived from HbA1c through the usual monotone relationship, then
  jittered, so the two never contradict each other.
- Insulin scales with BMI and the archetype's insulin factor.
- Age at diabetes onset is produced only when diabetes is reported and is always at least
  one year below the generated age.
- Every value is clamped to its field range, snapped to its step and formatted at its
  precision; nothing is negative.

### Tests

- `interface/synthetic_patient.test.js` — 18 deterministic tests on the pure generator with
  a seeded source: reproducibility, per-field constraints over 200 draws, categorical
  mapping, step/precision, BMI↔waist and HbA1c↔glucose covariance, onset-age logic, per
  archetype behaviour, outcome-label absence and validator behaviour.
- `interface/probe_synthetic_ui.test.js` — static `react-dom/server` render checks for the
  button semantics, accessible label/description, note wording, absence of the indicator
  before use, and that rendering makes no API call.

### Input attributes and known limitation

Five inputs already declare `step="0.1"` (BMI, waist, HbA1c, fasting glucose, insulin); age
and diabetes-onset age use the default integer step. The generator matches all of them - the
integer glucose values it emits are valid multiples of 0.1.

Limitation: the inputs carry no HTML `min`/`max` attributes, so the browser itself does not
bound them. The generator, `validateSyntheticProfile` and the unit tests enforce the ranges,
and `SYNTHETIC_FIELD_SCHEMA` is the single source of truth if `min`/`max` are added to the
markup later.