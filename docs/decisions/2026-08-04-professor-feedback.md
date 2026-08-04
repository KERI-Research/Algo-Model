# Research decision record — 2026-08-04: professor feedback on scope and method

**Status:** accepted, implemented in this repository on 2026-08-04.
**Scope:** MetaboGuard research direction, evidence handling, and analysis method.

> **These are recollected notes, written after the meeting from memory. They are not a
> transcript and contain no verbatim quotations.** Where a statement is our own inference
> or engineering consequence rather than something a supervisor said, it is marked
> *(our interpretation)*. Corrections from attendees are welcome and should be added as a
> dated amendment below rather than by rewriting this record.

## Attendees referenced

- **Prof. Helmount** — early-stage detection and clinical value requirement.
- **Prof. Nada** — clustering / phenotype discovery recommendation.

## Recollected feedback

### 1. Prof. Helmount — clinical value depends on early-stage detection

Recollected substance:

- A model only has clinical value if it contributes at a stage where intervention is still
  possible. Separating people who *already* have a diagnosis is not clinically useful.
- Early cancer detection realistically depends on **panels of markers and interacting
  features**, not on one universal marker that works across cancers.
- Any claim about detection must be tied to the stage and lead time at which the evidence
  was obtained.

Explicitly **not** claimed, and not to be written anywhere in this repository *(our
interpretation of the correct scientific statement)*:

- It is **false** to say that "no cancer has any specific biomarker". Site-specific markers
  exist and are clinically used (for example CA19-9 in pancreatic disease, with the
  documented sensitivity/lead-time limits recorded in our evidence catalogue). The
  defensible statement is that **no single marker is universally sufficient for early
  detection across cancers**, and that panels plus interacting features are required.

### 2. Prof. Nada — use clustering to discover phenotypes

Recollected substance:

- Rather than forcing a supervised outcome the data cannot support, run **unsupervised
  clustering** to look for patient/metabolic phenotypes.
- Disease labels may then be used to **characterise** the resulting clusters, but not to
  fit or select them.

Consequences *(our interpretation)*:

- Clusters are **patient/metabolic phenotypes**, never cancer diagnoses, cancer subtypes or
  cancer sites. Naming a cluster after a disease is prohibited in code, API and UI.
- Label use is strictly **post hoc** and must be reported as cross-sectional association
  with an already-recorded diagnosis, suppressed when class counts are inadequate.
- Clustering on survey data will happily rediscover the survey rather than biology, so
  **negative controls are mandatory**: any cluster solution dominated by survey cycle,
  assay availability, missingness burden, age or sex is reported as a data artefact.

## Decisions taken

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | Terminology: no causal phrasing. Use `risk-associated features`, `early-development signals`, `biological pathways`. `causal` is reserved for the DoWhy estimation module, where it names a method, not a finding. | Nothing in the current data supports causal claims. |
| D2 | Early detection is framed as a **panel + interaction** problem; the repository must never state that cancers have no specific biomarkers. | Scientific accuracy; see above. |
| D3 | Clustering is **exploratory phenotype discovery**, label-free in fit and selection, with an explicit **abstain** result when stability criteria fail. | A clustering that is not stable is not a finding. |
| D4 | An **evidence catalogue** with mandatory provenance (URL/DOI, study design, validation status, evidence grade, limitations) gates any doctor-facing statement. Unknowns are recorded as `unknown`, never inferred. | Prevents evidence drift and fabricated citations. |
| D5 | A **data reliability report** with feature eligibility tiers (`usable_now`, `qualified_use`, `unavailable`, `prohibited`) runs before analysis and fails closed on hard violations. | Reliability must be a gate, not a footnote. |
| D6 | Future-risk and cancer-site outputs stay **disabled and fail-closed**; NHANES here is cross-sectional and TCGA is post-diagnosis context only. | Unchanged from the pre-existing capability gates. |
| D7 | Clinician-facing explanations must label every statement as `data observation`, `model association`, `published evidence`, or `causal claim not established`. | Keeps the epistemic status visible at the point of reading. |

## What this does not change

- No new cancer-site prediction, no future-risk head, no re-enabling the invalidated
  pancreatic-cancer supervised artifacts.
- The self-supervised encoder remains frozen and label-free; clustering consumes its
  representation, it does not retrain it.

## Blocker that this feedback makes sharper, not smaller

Prof. Helmount's early-stage requirement cannot be met with the current data at all: there
is no follow-up time, no incident outcome and no stage information in the cross-sectional
NHANES files, and TCGA is post-diagnosis. Cluster phenotypes are therefore a hypothesis
generator for a future longitudinal study, not evidence of early detection.

## Amendments

*(none yet — append dated entries here rather than editing the notes above)*
