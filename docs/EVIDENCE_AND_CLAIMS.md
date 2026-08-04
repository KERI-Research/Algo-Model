# Evidence provenance and claims contract

Two things this document fixes in place:

1. **Where every biomarker statement comes from** (`data/evidence/biomarker_evidence.json`,
   loaded and validated by `api/evidence_catalogue.py`).
2. **What may and may not be said**, including the exact sentences that are allowed and
   the ones that are prohibited.

## The four interpretation classes

Every clinician-facing statement in MetaboGuard carries one of these labels, in the API,
the dashboard and this documentation:

| Class | Meaning | Example |
| --- | --- | --- |
| `data observation` | Measured directly in the data file | "HbA1c is observed in 62% of adults." |
| `model association` | Produced by our model on our sample; not validated, not causal | "This phenotype has a higher median HbA1c than the reference." |
| `published evidence` | Catalogued source with URL, study design and evidence grade | "GALAD reached AUC 0.78 within 12 months in a prospective 7-centre study." |
| `causal claim not established` | Default status for any mechanism statement | "Nothing here shows that adiposity caused this person's disease." |

## Catalogue schema

Required per row: `entry_id`, `cancer_site`, `marker_or_panel`, `marker_class`, `specimen`,
`intended_use`, `stage_or_lead_time`, `direction`, `study_design`, `sample_size`,
`performance`, `validation_status`, `evidence_grade`, `limitations`,
`primary_source_url`, `doi`, `related_verified_sources`.

Optional: `repo_reference`, `available_in_current_data`, `current_data_column`, `notes`,
`screening_recommendation_status`, `allowlisted_statements`, `denied_statements`.

Two explicit placeholders, never inferred:

- `unknown` — the value exists but has not been extracted from the source.
- `n.a.` — the field does not apply to this row.

An empty string is a validation error, so a missing field can never be mistaken for a
negative finding.

### Gates enforced in code

| Gate | Rule |
| --- | --- |
| Provenance | `primary_source_url` must be a structurally valid URL, or `doi` a valid DOI, or both explicit placeholders. |
| Clinician-facing | A row reaches a clinician view only with a real source **and** a graded `evidence_grade`. |
| Statements | `allowlisted_statements` require a real source; rows without one may hold no allowlisted statement. |
| Causal language | Causal phrasing is rejected unless `study_design` is a causal design (RCT, Mendelian randomisation). |
| Universal-denial | The claim that cancers have no specific biomarkers is rejected outright. |

`python api/evidence_catalogue.py --strict` exits non-zero on any hard issue.

## Catalogue contents (v1.1.0, 2026-08-04)

20 rows: 17 clinician-ready, 3 research-only. Sites: pancreas, liver (HCC), multi-site.

| Row | Marker or panel | Design | Key figures | Grade |
| --- | --- | --- | --- | --- |
| `ev-ca199-alone-pdac-bjsopen-2024` | CA19-9 alone | Meta-analysis of prediagnostic studies | AUC 0.998 at diagnosis, 0.87 at 6 mo, 0.74 at 12 mo, 0.55 at 5 yr | phase 3 prediagnostic, not screening grade ([source](https://academic.oup.com/bjsopen/article/8/3/zrae046/7700226), DOI 10.1093/bjsopen/zrae046; [USPSTF](https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/pancreatic-cancer-screening)) |
| `ev-thbs2-ca199-pdac-scitranslmed-2017` | THBS2 + CA19-9 | Case-control phase 2b, n=537 | c-statistic 0.97; 87% sensitivity at 98% specificity; no lead time | discovery only ([source](https://pubmed.ncbi.nlm.nih.gov/28701476/), DOI 10.1126/scitranslmed.aah5583) |
| `ev-five-marker-panel-pdac-bjsopen-2024` | CA19-9 + CA125 + VWF + THBS2 + IL6ST | Panel evaluation within the meta-analysis | AUC 0.91 within 1 yr; 0.78 up to 4 yr | internal discovery only ([source](https://academic.oup.com/bjsopen/article/8/3/zrae046/7700226)) |
| `ev-endpac-pdac-digdissci-2020` | ENDPAC (age, glucose change, weight change) | Retrospective cohort, 13,947 NOD / 99 PDAC | AUC 0.75; PPV 2.0%; NPV 99.7% | emerging external validation, risk stratification ([source](https://pubmed.ncbi.nlm.nih.gov/32112260/), DOI 10.1007/s10620-020-06139-z) |
| `ev-recent-diabetes-weightloss-pdac-jamaoncol-2020` | Recent-onset diabetes ≤4 yr + >8 lb weight loss | Prospective cohorts, 112,818 women + 46,207 men, 1,116 PDAC | Incidence ratio 10.57 (7.18–15.56); absolute 4-yr incidence 0.29% | moderate prospective association ([source](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7426876/), DOI 10.1001/jamaoncol.2020.2948) |
| `ev-thrombocytosis-multisite-bjgp-2017` | Thrombocytosis >400×10⁹/L | Retrospective cohort with controls, ~40,000 exposed / 10,000 controls | 1-yr cancer PPV 11.6% men, 6.2% women; 18.1% / 10.1% after a second raised count; mainly lung and colorectal | moderate for risk marking ([source](https://pubmed.ncbi.nlm.nih.gov/28533199/), DOI 10.3399/bjgp17X691109) |
| `ev-excess-body-fatness-multisite-nejm-2016` | Excess body fatness | IARC working-group review | Sufficient evidence: colon, kidney, postmenopausal breast, corpus uteri, liver, pancreas, ovary | IARC sufficient evidence for risk association ([source](https://www.nejm.org/doi/full/10.1056/NEJMsr1606602), DOI 10.1056/NEJMsr1606602) |
| `ev-galad-hcc-gastro-2024` | GALAD (sex, age, AFP-L3, AFP, DCP) | Prospective phase 3, 7 centres, n=1,558 / 109 HCC | AUC 0.78 vs AFP 0.66 within 12 mo; 62% sensitivity at 82% specificity | phase 3 prospective, high-risk surveillance ([source](https://pubmed.ncbi.nlm.nih.gov/39293548/), DOI 10.1053/j.gastro.2024.09.008) |
| `ev-cancerseek-multisite-science-2018` | CancerSEEK (proteins + cfDNA) | Case-control, clinically detected, 1,005 cases / 812 controls | Median sensitivity 70% at >99% specificity; stage I 43%; breast 33% | discovery only, spectrum-biased for screening ([source](https://pubmed.ncbi.nlm.nih.gov/29348365/), DOI 10.1126/science.aar3247) |

Earlier rows transcribed from `RESEARCH_EVIDENCE.md` (HbA1c, C-peptide, HOMA-IR, CA19-9
lead time, weight loss, adiposity, hs-CRP null, lipid evidence gap, NOD panels) are
retained unchanged.

## Claims contract

Any MetaboGuard claim about detection performance must satisfy these standards before it
leaves the research setting:

| Standard | Applies to | Source |
| --- | --- | --- |
| PRoBE | biomarker study design and specimen provenance | [PRoBE design paper](https://edrn.cancer.gov/documents/158/PRoBEStudyDesign.pdf), DOI 10.1093/jnci/djn326 |
| TRIPOD+AI | reporting of prediction-model development and validation | [BMJ 2024](https://www.bmj.com/content/385/bmj-2023-078378), DOI 10.1136/bmj-2023-078378 |
| PROBAST+AI | risk-of-bias and applicability appraisal | [probast.org](https://www.probast.org/probast_ai) |
| STARD | reporting of diagnostic accuracy studies | [EQUATOR](https://www.equator-network.org/reporting-guidelines/stard/) |

**Consequence for this project.** Current outputs are cross-sectional deviation scores,
exploratory phenotypes and reliability audits. They do not meet PRoBE specimen
requirements, have no prediagnostic lead time, and are reported as research only.

## Allowed statements (18 catalogued, each tied to a source)

Representative examples:

- "A pancreas-specific marker exists (CA19-9), but its discrimination falls from near-perfect
  at diagnosis to about 0.74 at 12 months and 0.55 at 5 years before diagnosis."
- "CA19-9 is not recommended for screening asymptomatic adults."
- "GALAD is a five-component panel for surveillance of high-risk cirrhosis that reached AUC
  0.78 within 12 months, against 0.66 for AFP alone."
- "A raised platelet count carries a 1-year cancer positive predictive value of 11.6% in men
  and 6.2% in women, and does not indicate which site is involved."
- "Recent-onset diabetes combined with weight loss carries an incidence ratio of 10.57, yet
  the absolute 4-year incidence is only 0.29%."
- "IARC found sufficient evidence linking excess body fatness to cancers of the colon, kidney,
  postmenopausal breast, corpus uteri, liver, pancreas and ovary."
- "The human causal effect of intentional weight loss on cancer incidence remains unestablished."

## Denied statements (23 catalogued)

Never say, write or display:

- "No cancer has a specific biomarker." — **false**; CA19-9 and AFP/GALAD components are
  catalogued counter-examples. The defensible statement is that **no single marker is
  universally sufficient for early detection across cancers**, so panels and interacting
  features are required.
- "CA19-9 detects pancreatic cancer early."
- "An AUC of 0.998 / 0.97 / 0.91 shows this is ready for screening."
- "Recent diabetes plus weight loss means the patient probably has pancreatic cancer."
- "A high platelet count indicates pancreatic cancer."
- "Losing weight prevents cancer." / "Obesity causes pancreatic cancer in this patient."
- "GALAD can screen the general population for liver cancer."
- "A single blood test can already detect any cancer early."
- "Pancreatic cancer cases will double by 2050." / "The rise is caused by metabolic disease."

## Approved burden wording

> Global pancreatic cancer burden was **531,318 cases and 490,786 deaths in 2024**
> ([source](https://pmc.ncbi.nlm.nih.gov/articles/PMC13343830/), DOI 10.3322/caac.70090), and a
> **demographic constant-rate projection** gives **998,663 cases and 936,038 deaths in 2050**
> ([source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11539015/), DOI 10.1001/jamanetworkopen.2024.43198)
> if incidence and mortality rates stay unchanged.

This is a demographic projection holding rates constant. It is **not a causal forecast**,
not a prediction of what will happen, and not attributable to any risk factor.

## Adding newly verified rows

1. Append the row to `data/evidence/biomarker_evidence.json` with every required field.
2. Use `unknown` / `n.a.` explicitly instead of guessing.
3. Run `python api/evidence_catalogue.py --strict` and `python -m unittest test_research_pass`.
4. If the row carries `allowlisted_statements`, confirm the source URL or DOI resolves.