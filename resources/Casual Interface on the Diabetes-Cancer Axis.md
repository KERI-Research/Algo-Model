# Causal Inference on the Diabetes–Cancer Axis: Blind Spots, Methodological Limits, and Prospects for Agentic Causal ML

## Executive Overview

Recent causal inference work suggests that diabetes, especially type 2 diabetes (T2D), has site-specific causal relationships with several cancers (notably pancreatic, liver, endometrial, kidney, uterine, and cervical), while many other observed associations attenuate after careful bias analysis. However, the mechanistic and causal picture remains incomplete, with obesity, hyperinsulinaemia, chronic inflammation, dyslipidaemia, and treatment effects entangled in complex, poorly instrumented pathways. Mendelian randomization (MR) and structural causal model (SCM) analyses provide partial answers but exhibit important blind spots: residual pleiotropy, crude adiposity instruments, weak time-resolved phenotyping, and limited modeling of treatment/medication and multi-morbidity trajectories.[^1][^2][^3][^4][^5][^6][^7]

This report critically evaluates the current causal evidence linking diabetes and cancer, focusing on missed confounders and backdoor paths, limitations of MR and SCMs, and how role-based multi-agent AI systems can be used to stress-test DAGs, automate refutation (e.g., via DoWhy), and integrate heterogeneous epidemiological evidence. The goal is to inform the design of a causal inference pipeline ("KERI") that isolates the diabetes→cancer signal from shared upstream risk factors and measurement/treatment artefacts.[^8][^9][^10][^11]

## Current Causal Evidence: Site-Specific Signals and Global Ambiguities

Umbrella reviews and bias analyses show that while diabetes is associated with incidence of many cancers, only a minority of these associations remain robust when small-study effects, heterogeneity, and unmeasured confounding are explicitly assessed. A large meta-analysis of 151 cohorts (32 million people) found strong, bias-robust evidence for causal associations of T2D with liver, pancreatic, endometrial, gallbladder, kidney, colon, and colorectal cancers, while associations with most other cancers were sensitive to plausible unmeasured confounding.[^12][^13][^2]

Two-sample MR studies using genetic liability to T2D and fasting insulin generally support harmful effects of T2D and hyperinsulinaemia on pancreatic, kidney, uterine, cervical, and lung cancers, with protective or null effects on melanoma and esophageal cancer and most other sites. Conversely, some MR studies with different instruments or ancestries (e.g., Japanese or East Asian cohorts) find minimal evidence that genetic liability to T2D per se drives total or site-specific cancer risk, emphasizing population-specific genetic architectures and phenotype definitions.[^14][^4][^6][^15]

Recent cohort work with rich electronic health record data continues to show elevated risks of liver and pancreatic cancer among diabetics, with dose-response relationships between fasting glucose and these cancers, but also links between low lipid levels and increased risk for several malignancies, hinting at reverse causality, cachexia, or medication/treatment pathways. Overall, the causal signal is strongest for pancreatic and liver cancers; for many other cancers, diabetes appears more as a proxy for shared metabolic and inflammatory states rather than a distinct causal driver.[^5][^16]

## Shared Confounders and Mis-Specified Backdoor Paths

### Obesity, Adiposity, and Body Composition

Excess adiposity is a major shared risk factor for both diabetes and multiple cancers, yet most MR work still relies on crude BMI instruments instead of more causally proximal measures such as visceral adiposity, ectopic fat, or waist-to-hip ratio. Meta-analyses of MR studies show that genetically predicted higher BMI increases risk for T2D and numerous cancers (digestive system, uterus, kidney, bladder), while decreasing risk for breast, prostate, and non-melanoma skin cancer, indicating highly heterogeneous site-specific relationships.[^3][^17]

These findings suggest that many apparent diabetes→cancer associations may be explained by backdoor paths Diabetes ← Adiposity → Cancer, mis-specified when only BMI is modeled and more granular adiposity phenotypes are omitted. Moreover, adiposity-related metabolites (branched-chain amino acids, lipid fractions, inflammatory markers) have emerging causal roles, mediating obesity→diabetes and obesity→cancer links, but are rarely incorporated as explicit nodes or mediators in current DAGs.[^18][^19][^1][^3]

### Hyperinsulinaemia, Glycaemic States, and Treatment Regimes

Hyperinsulinaemia, insulin resistance, and chronic hyperglycaemia have direct effects on proliferation, apoptosis, and the tumor microenvironment, yet many causal analyses treat "diabetes" as a binary phenotype, ignoring dynamic trajectories of insulin, C-peptide, and glucose over time. MR studies using fasting insulin instruments suggest that insulin, rather than T2D diagnosis per se, is positively associated with uterine, kidney, pancreatic, and lung cancers, but treatment-mediated changes (e.g., exogenous insulin, insulin secretagogues) are often unmodeled.[^4][^1]

Drug exposures such as metformin, SGLT2 inhibitors, GLP-1 receptor agonists, and insulin analogues likely sit on mediator and confounder pathways that are rarely disentangled. For example, metformin has been hypothesized to reduce cancer risk via AMPK activation and mTOR inhibition, but large Cox regression analyses have not consistently supported protective effects once confounding by indication and time-related biases (immortal time bias) are accounted for. DAGs that collapse "diabetes treatment" into a single node miss important backdoor structures like Diabetes → Insulin Use → Weight Gain → Cancer and Cancer → Weight Loss → Discontinuation of Therapy → Glycaemic Drift.[^20][^1]

### Multi-Morbidity, Organ Dysfunction, and Reverse Causality

Many cancers, especially pancreatic and liver, can induce diabetes via direct organ destruction, paraneoplastic syndromes, or treatment-related effects, generating complex bidirectional relationships. Two-sample MR on reverse causation (site-specific cancers → diabetes) has shown suggestive evidence that lymphoid leukaemia increases diabetes risk, pointing to cancer→diabetes pathways that are largely overlooked in forward causal schemes.[^7][^21][^1]

Cohort studies that exclude cancer diagnoses only at baseline, without time-varying exclusion or careful lagging, can mis-attribute prodromal cancer effects on glucose metabolism to diabetes→cancer causation. Furthermore, comorbid conditions like chronic liver disease, chronic kidney disease, and cardiovascular disease both alter diabetes risk and cancer risk, often through shared inflammation and metabolic disruption, yet are typically controlled by simple adjustment rather than explicit DAG modeling of mediated and collider paths.[^2][^22][^23][^19]

### Socioeconomic Position, Healthcare Access, and Surveillance Bias

Socioeconomic status (SES), healthcare access, and screening intensity are strong determinants of both diabetes diagnosis and cancer detection, but often occupy a single "adjusted covariate" role rather than being explicitly modeled as colliders or mediators. Increased screening in diabetic patients can inflate apparent incidence rates for certain cancers (e.g., colorectal, liver) by increasing detection, while reduced access in low-SES diabetics may mask true incidence for others; these opposing biases coexist and are rarely modeled jointly.[^19][^12]

In some MR designs that rely on biobank participants, selection bias via healthy volunteer effects can also distort the diabetes–cancer relationship, as individuals with severe multi-morbidity or advanced cancers are underrepresented. Without explicit selection nodes in DAGs, these biases are usually handled only informally.[^6][^4]

## Limitations of Mendelian Randomization in the Diabetes–Cancer Intersection

### Phenotype Heterogeneity and Misclassification

MR analyses on diabetes and cancer commonly use broad T2D GWAS instruments that conflate different etiological subtypes (e.g., autoimmune T2D, lean vs obese T2D, diabetes from pancreatic disease) and do not reflect treatment history or duration of hyperglycaemia. This phenotype heterogeneity attenuates estimates, obscures subtype-specific causal links, and increases vulnerability to pleiotropic pathways that bypass the intended diabetes phenotype.[^4][^6]

Similarly, cancer phenotypes in MR tend to be site-specific but rarely subtype-specific (e.g., molecular subtypes of breast, colorectal, or lung cancers), despite differential sensitivity to insulin, glucose, and inflammatory milieus. Instruments that aggregate across diverse tumor biology risk averaging away causal signals for specific subtypes and inflating null findings.[^24][^4]

### Pleiotropy, Horizontal Pathways, and Metabolic Instruments

Many T2D loci also affect BMI, lipid profiles, inflammatory markers, and other metabolic traits, so classical MR assumptions of no horizontal pleiotropy are often violated. Although sensitivity methods (MR-Egger, weighted median, MR-PRESSO) are widely applied, they rely on parametric assumptions and cannot fully resolve complex pleiotropy where the same variant influences multiple mediators on different pathways to cancer.[^3][^6]

Obesity-focused MR studies reinforce that BMI instruments are themselves deeply pleiotropic, affecting cancers via multiple organ-specific mechanisms and systemic inflammatory/metabolic cascades; in this context, separating "pure" diabetes effects from adiposity-driven effects using standard MR alone is arguably impossible. Multi-instrument MR that simultaneously models T2D, BMI, lipids, and insulin improves identification but still struggles to capture non-linear dose–response, time-varying effects, and threshold phenomena.[^18][^3]

### Time, Dose, and Trajectory Effects

Cancer development often depends on duration and intensity of metabolic disturbances, but MR generally captures lifelong genetic liability, not actual clinical trajectories (age at onset, disease duration, glycaemic control, treatment adherence). For example, short-lived hyperglycaemia around cancer onset can be driven by tumor biology, while long-standing poorly controlled diabetes might exert very different effects; MR cannot differentiate these trajectories.[^2][^5]

Bias analyses that integrate meta-analytic hazard ratios with hypothetical unmeasured confounders show that plausible levels of unmeasured confounding can erase or reverse many weaker diabetes–cancer associations, underscoring that MR estimates may still be contaminated by time-varying confounding and selection. As a result, causal claims from MR in this domain should be treated as hypotheses to be tested within more comprehensive SCMs and longitudinal data, rather than definitive answers.[^2]

## Structural Causal Models and DAGs: Current Practice and Gaps

### DAG Use in Endocrinology and Oncology

DAGs are increasingly used to formalize assumptions for diabetes-related interventions and to guide decision-analytic models, highlighting causal pathways and backdoor structures, but most applied models focus on micro-level diabetes outcomes (e.g., insulin dosing, complication progression) rather than cancer endpoints. Causal Bayesian networks and model-averaging across structure learning algorithms have been used to derive intervention targets for diabetes progression, revealing that algorithm choice substantially alters inferred causal pathways.[^25][^26][^27]

In oncology and endocrinology, DAGs are often schematic and qualitative, summarizing hypothesized pathways (obesity→insulin resistance→T2D→cancer) without systematic structure learning or formal identification tests; confounders are included or omitted largely by expert judgment. These ad hoc DAGs rarely undergo rigorous robustness analysis to alternative structural assumptions, leaving many backdoor paths either unacknowledged or poorly controlled.[^1][^19]

### Identification and Refutation Workflows (DoWhy and Beyond)

Frameworks such as DoWhy emphasize four steps: (1) specifying a causal graph, (2) identifying estimands, (3) estimating causal effects, and (4) refuting estimates via robustness checks and sensitivity analyses. DoWhy includes placebo tests, bootstrap refutations, and unobserved confounding tests, and supports interoperability with EconML and CausalML, making it a natural backbone for diabetes–cancer SCM pipelines.[^28][^8]

However, most biomedical use of such frameworks still depends on a single DAG or a small set of hand-designed DAGs, without systematic exploration of the space of plausible structures or dynamic integration of new evidence. Moreover, DAGs are usually static and cross-sectional, ignoring longitudinal trajectories, feedback loops, and treatment changes, which are critical in multi-morbid patients moving between diabetes and cancer states.[^11][^8]

### Missed Nodes and Edges: Biological and Care-Process Mechanisms

Several classes of variables are systematically underrepresented in published DAGs on diabetes–cancer:

1. **Tumor–metabolism feedback loops**: Cancer can alter systemic metabolism (e.g., Warburg effect, cachexia, inflammatory cytokines), which feeds back into glucose control and insulin resistance, generating bidirectional arrows that are rarely modeled explicitly.[^21][^1]
2. **Multi-morbidity clusters**: Shared chronic conditions (NAFLD/NASH, CKD, HF) both predispose to diabetes and to specific cancers via organ dysfunction and inflammation, acting as upstream common causes that break simple diabetes→cancer narratives.[^19][^2]
3. **Care-process nodes**: Screening frequency, treatment intensification, adherence, therapeutic inertia, and post-cancer changes in diabetes management form complex pathways affecting both detection and progression of cancers and diabetes, yet are often lumped into "healthcare access" or omitted entirely.[^29][^19]
4. **Socio-behavioral and environmental exposures**: Diet, physical activity, smoking, alcohol, occupational exposures, and air pollution interact with diabetes and cancer risk; MR and SCMs seldom integrate such nodes beyond basic adjustment, even though they may be on causal or collider paths.[^12][^3]

For a pipeline like KERI, these missed nodes and edges are exactly where multi-agent systems and DAG exploration should concentrate, using both structured data (NHANES, UK Biobank) and unstructured literature to propose and test alternative causal structures.

## Missed Information and Blind Spots in Current Literature

### Under-Characterized Metabolic Mediators and Multi-Omics Signals

Recent work highlights branched-chain amino acids and other plasma metabolites as potential mediators linking obesity to diabetes and cancer, yet these variables have only sporadically appeared in causal analyses. Mendelian randomization analyses of obesity-related plasma metabolites show candidate mediators for cancer risk, but integration into joint diabetes–cancer DAGs remains rare.[^1][^18]

Similarly, liver enzymes, lipid subfractions, and inflammatory biomarkers (CRP, IL-6, TNF-α) show differential associations with site-specific cancers in diabetic versus non-diabetic patients, suggesting interaction effects and multiple causal channels, but most MR and SCM work treats these markers as confounders rather than mediators or interaction modulators. Multi-omics causal discovery (e.g., using graph autoencoders and structured neural networks) is only beginning to be applied to metabolic data, and has not yet been widely leveraged in diabetes–cancer research.[^16][^30][^5]

### Cancer Subtypes, Treatment Regimens, and Time Windows

The diabetes–cancer relationship is likely strongly subtype-dependent (e.g., hormone receptor status, mutational profiles, metabolic phenotypes of tumors), but causal analyses tend to aggregate across subtypes, masking potentially large heterogeneity. Treatment regimens (chemotherapy, immunotherapy, endocrine therapy, radiotherapy) and their metabolic side effects (steroid-induced hyperglycaemia, immunotherapy-related endocrine toxicities) are rarely modeled in longitudinal SCMs linking diabetes and cancer outcomes.[^23][^29][^24][^19]

Time windows around cancer diagnosis and treatment are especially critical: pre-diagnostic hyperglycaemia and weight change may reflect tumor biology, while post-treatment metabolic changes may be driven by therapy. Most cohort and MR designs use broad follow-up intervals and baseline exposures, ignoring these windows and potentially conflating reverse causality and forward effects.[^5][^23]

### Ancestry, Sex, and Social Determinants

Current MR and observational literature is heavily skewed toward European ancestry cohorts, with emerging but still limited data from East Asian, Black, and other populations. Given differences in genetic architecture, environmental exposures, and social determinants, causal pathways for diabetes and cancer may differ substantially by ancestry, yet DAGs seldom encode ancestry-specific structures or interaction terms.[^15][^31][^6]

Sex-specific effects also appear: for example, MR indicates increased uterine and cervical cancer risk with T2D, while obesity MR suggests decreased risk for breast cancer, pointing to complex hormonal and reproductive pathways that are under-characterized. Social determinants such as SES, discrimination, stress, and neighborhood environment, which systematically shape both diabetes and cancer risks, remain peripheral in causal modeling.[^3][^4]

## Agentic Operations: Multi-Agent AI to Stress-Test Causal Assumptions

### Multi-Agent Causal Discovery Frameworks

Recent work on multi-agent causal discovery using large language models (LLMs) demonstrates that agent-based debating systems can outperform single-method causal discovery by combining structured data with metadata and iteratively refining causal graphs. Frameworks such as MAC (Multi-Agent Causal discovery) and MATMCD (Multi-Agent Tool-augmented Multi-Modal Causal Discovery) deploy distinct agents for debate, coding, data augmentation, and constraint integration to select statistical causal discovery methods and refine graphs.[^9][^10][^32][^33]

These systems use a Debate-Coding Module where agents argue over method choice (e.g., PC, GES, NOTEARS, GRaSP, temporal causal discovery), apply the selected method, then a Meta-Debate Module to refine the resulting graph using metadata and domain knowledge. In biomedical settings, such multi-agent frameworks can leverage both structured longitudinal data and textual evidence (e.g., PubMed, guidelines) to propose and adjudicate edges and orientations in complex DAGs.[^10][^9]

### Tool-Augmented Agents and Multi-Modal Integration

Tool-augmented LLM agents can call specialized causal inference libraries (DoWhy, EconML, CausalML), statistical packages, and database APIs to perform estimation, refutation, and sensitivity analyses automatically. MATMCD introduces a Data Augmentation agent to retrieve and process multi-modal data (omics, imaging, lab results, clinical notes) and a Causal Constraint agent to integrate this data into knowledge-driven inference, demonstrating improved performance across multiple datasets.[^8][^10][^11]

For KERI, an analogous architecture could include:

- **Structure-learning agents**: Evaluate diverse algorithms (constraint-based, score-based, hybrid, continuous-time) on NHANES and UK Biobank data, compare outputs, and perform model averaging, similar to diabetes-structure-learning work.[^26]
- **Domain-knowledge agents**: Ingest mechanistic and clinical literature (endocrinology, oncology) and propose constraints (forbidden edges, required paths, plausible mediators) from textual evidence, grounding DAG editing.
- **Refutation/testing agents**: Wrap DoWhy and related tools to run placebo, subset, bootstrap, and unobserved-confounding tests under different DAGs, automatically logging which structures yield estimates robust to specific threats.[^8]
- **Critic/reviewer agents**: Act as skeptical peers, highlighting unmodeled backdoor paths, colliders, and missing variables based on current epidemiological and mechanistic evidence.

### Role-Based Research Squads for Diabetes–Cancer Causal Pipelines

A role-based multi-agent "research squad" can map naturally onto the needs of a diabetes–cancer causal pipeline:

1. **Causal Architect agent**: Maintains the core DAG(s) and SCMs, integrates output from structure-learning and domain-knowledge agents, and ensures identification conditions for the diabetes→cancer estimand are explicit.
2. **Data Scientist agent**: Runs estimation on NHANES (cross-sectional prototyping) and UK Biobank (longitudinal scaling), using diverse estimators (parametric g-computation, inverse probability weighting, targeted maximum likelihood, double machine learning) under the current DAG.
3. **Refutation/Robustness agent**: Automates DoWhy-style refutation: placebo treatments, outcome randomization, subset analyses (ancestry-, sex-, treatment-stratified), negative-control exposure/outcome tests, and simulated unobserved-confounding analysis.[^8]
4. **Literature Synthesizer agent**: Continuously scans new MR, cohort, and mechanistic literature (2024–2026 onward), updating evidence tables for each candidate path (e.g., insulin→pancreatic cancer, obesity→endometrial cancer, lymphoid leukemia→diabetes) and flagging contradictions.[^24][^7][^1][^2]
5. **Critic/Reviewer agent**: Evaluates current DAGs for missing nodes, implausible orientations, and unjustified conditional independencies, cross-checking against literature and structured data and issuing "review reports" that must be addressed before estimates are accepted.

These agents can be orchestrated via a workflow engine where any proposed causal conclusion (e.g., "T2D causally increases pancreatic cancer risk independent of BMI and lipids") triggers a mandatory debate cycle between the architect, data scientist, refutation agent, and critic.

### Dynamic DAG Challenge and Refutation Loops

To avoid DAG lock-in, agentic operations should encourage dynamic DAG challenge and refutation:

- **DAG proposal phase**: Structure-learning agents generate candidate DAGs consistent with observed conditional independencies; domain-knowledge agents impose mechanistic constraints (e.g., cancer cannot cause genetic liability to diabetes).
- **Estimand and identification phase**: Architect agents derive estimands for diabetes→site-specific cancer effects under each DAG, checking identifiability via backdoor/front-door criteria and, where necessary, instrumental-variable or mediation formulae.
- **Estimation and refutation phase**: Data scientist and refutation agents estimate effects using multiple estimators and run robustness checks, logging divergences across DAGs and estimators.
- **Debate and revision phase**: Critic/reviewer agents highlight DAGs where estimates are highly sensitive to small structural changes or show inconsistency with external MR/cohort evidence, prompting targeted revisions.

This loop can be formalized as a reinforcement-learning environment where the "reward" is robustness of estimates across sensitivity analyses and alignment with external evidence, while "actions" are DAG edits and variable inclusion/exclusion.

### Automating DoWhy Refutation and Bias Analysis

DoWhy’s built-in refuters can be wrapped by agents to systematically probe weaknesses in diabetes–cancer effects estimated from NHANES and UK Biobank:[^28][^8]

- **Placebo Tests**: Replace diabetes with pseudo-treatments (e.g., height) that should not causally affect cancer, ensuring estimated effects collapse to null, thereby validating pipeline calibration.
- **Subset Refutation**: Verify that estimated effects are stable across key subgroups (sex, age, BMI strata, ancestry, treatment regimens), or identify where DAGs must be stratified or interaction terms added.
- **Unobserved Confounding Tests**: Simulate unobserved confounders with specified strengths and correlations to assess how sensitive diabetes→cancer estimates are to plausible hidden variables, analogous to meta-analytic bias analyses.[^2]
- **Negative Controls**: Use exposures and outcomes where no causal relationship is expected (e.g., T2D→non-melanoma skin cancer) to detect residual confounding or modeling artefacts, guided by MR findings of protective or null associations.[^4][^3]

Agents can automatically produce dashboards summarizing which diabetes→cancer links remain robust across refuters, which fail specific tests, and which require DAG augmentation.

## Synthesis of Contradictory Epidemiological Evidence via Agents

### Evidence Graphs and Contradiction Maps

Given the heterogeneous and sometimes contradictory evidence (e.g., MR vs cohort vs mechanistic studies), an evidence-graph approach is useful: nodes represent causal claims (e.g., "T2D increases pancreatic cancer incidence"), edges encode supporting or contradicting studies (with weights for methodological quality, risk-of-bias, and relevance).[^24][^4][^2]

A Literature Synthesizer agent can continually update this graph, while a Critic agent can highlight inconsistencies where DAG-implied conditional independencies conflict with high-quality evidence. For instance, if DAGs imply that after conditioning on BMI and lipids, diabetes should not affect colorectal cancer, but a large bias-robust cohort suggests otherwise, the critic can demand DAG reconsideration or search for missing mediators.[^9][^10]

### Programmatic Protocol Generation and Registration

Agents can also assist in generating and updating analysis protocols (analogous to pre-registration), specifying primary estimands, DAGs, covariate sets, and refutation plans for diabetes→cancer causal analyses. These protocols can be versioned and tied to specific NHANES/UK Biobank data snapshots, minimizing researcher degrees-of-freedom and agent drift.

Tools for decision-analytic modeling with DAGs show that explicit visual and structural representation of pathways improves modelers’ decisions; a protocol-generating agent can formalize this for KERI, ensuring that each structural assumption is documented and subject to reviewer-agent critique.[^27]

## Implications for KERI: Design Principles and Research Agenda

### Design Principles for the Diabetes–Cancer DAGs

1. **Explicit separation of shared causes vs mediators**: Model obesity, multi-morbidity, SES, and behavioral factors as upstream causes with clear arrows to both diabetes and cancer, rather than adjusting them away without structural representation.[^12][^3]
2. **Bidirectional diabetes–cancer pathways**: Include cancer→diabetes paths (especially pancreatic, liver, and haematologic malignancies), with time windows distinguishing pre- and post-diagnosis effects.[^7][^21]
3. **Treatment and care-process nodes**: Represent medications, screening intensity, and care trajectories (e.g., changes in diabetes management after cancer diagnosis) as distinct nodes with their own causal paths.[^29][^20]
4. **Multi-omics and metabolite mediators**: Integrate metabolite, lipid, liver enzyme, and inflammatory markers as candidate mediators in obesity→diabetes and obesity→cancer pathways, guided by MR and mechanistic evidence.[^18][^5][^1]
5. **Ancestry and sex stratification**: Allow for DAG variants or hierarchical SCMs that differ by ancestry and sex, reflecting observed heterogeneity in MR and cohort studies.[^31][^6]

### Leveraging NHANES and UK Biobank

NHANES offers repeated cross-sectional measures with rich lab and questionnaire data suitable for prototyping DAGs and testing cross-sectional conditional independencies. Agent-based structure-learning can combine algorithmic outputs with domain constraints to propose candidate DAGs for metabolic and behavioral factors.[^26][^8]

UK Biobank provides prospective longitudinal data with genetic instruments, enabling integration of MR-inspired constraints and time-varying models of diabetes, adiposity, metabolite levels, and cancer incidence. Multi-agent causal discovery frameworks can align DAGs inferred from NHANES with longitudinal constraints from UK Biobank, identifying which paths are stable and which are artefacts of cross-sectional data.[^19][^4]

### Research Agenda: Filling Blind Spots

Key priorities include:

- Developing ancestry- and sex-aware DAGs for the diabetes–cancer axis, incorporating multi-morbidity and care-process nodes.
- Integrating MR evidence on obesity-related metabolites and multi-omics signals as mediators, tested via mediation analysis in UK Biobank.[^30][^18]
- Systematically modeling reverse causality and prodromal cancer effects on diabetes diagnoses and trajectories.[^23][^7]
- Deploying multi-agent LLM systems to conduct continual literature surveillance and causal-graph critique, ensuring KERI’s DAGs remain aligned with evolving evidence.[^10][^9]
- Formalizing robustness metrics (e.g., proportion of refuters passed, alignment with MR/bias-analyses) as optimization targets for agentic DAG refinement.

By embedding MR and cohort evidence into a multi-agent SCM pipeline, KERI can aim to isolate the true causal component of diabetes→site-specific cancer risk, while explicitly quantifying the contribution of shared confounders, mediators, and reverse causation.

---

## References

1. [Diabetes and Cancer: A Twisted Bond - PubMed](https://pubmed.ncbi.nlm.nih.gov/38835644/) - This paper presents an overview of the interconnection between various factors related to both cance...
2. [Association of Type 2 Diabetes With Cancer: A Meta-analysis With ...](https://pubmed.ncbi.nlm.nih.gov/32910779/) - Our findings strongly suggest a causal association between T2D and liver, pancreatic, and endometria...
3. [Causal role of high body mass index in multiple chronic diseases: a systematic review and meta-analysis of Mendelian randomization studies](https://link.springer.com/article/10.1186/s12916-021-02188-x) - ## Background: Obesity is a worldwide epidemic that has been associated with a plurality of disease...
4. [Is Type 2 Diabetes Causally Associated With Cancer Risk? Evidence From a Two-Sample Mendelian Randomization Study - PubMed](https://pubmed.ncbi.nlm.nih.gov/32349989/) - We conducted a two-sample Mendelian randomization study to investigate the causal associations of ty...
5. [Association Between Diabetes and Site-Specific Cancer Risk: A Population-Based Cohort Study on the Differential Role of Metabolic Profiles - PubMed](https://pubmed.ncbi.nlm.nih.gov/40831756/) - This study is aimed at investigating (i) whether diabetes is associated with each site-specific canc...
6. [Is Type 2 Diabetes Causally Associated With Cancer Risk ...](https://www.repository.cam.ac.uk/items/4d837621-aa46-4ca2-892f-609f82b8a316) - by S Yuan · 2020 · Cited by 174 — We conducted a two-sample Mendelian randomization study to investi...
7. [Causal associations between site-specific cancer and ...](https://pubmed.ncbi.nlm.nih.gov/36860363/) - by R Xu · 2023 · Cited by 14 — Causal associations between site-specific cancer and diabetes risk: A...
8. [DoWhy: An End-to-End Library for Causal Inference - arXiv](https://arxiv.org/abs/2011.04216) - In addition to efficient statistical estimators of a treatment's effect, successful application of c...
9. [Multi-Agent Causal Discovery Using Large Language Models](https://arxiv.org/abs/2407.15073) - by HD Le · 2024 · Cited by 43 — Abstract:Causal discovery aims to identify causal relationships betw...
10. [Exploring Multi-Modal Integration with Tool-Augmented LLM Agents for Precise Causal Discovery](http://www.arxiv.org/abs/2412.13667) - Causal inference is an imperative foundation for decision-making across domains, such as smart healt...
11. [DoWhy – A library for causal inference - Microsoft Research](https://www.microsoft.com/en-us/research/blog/dowhy-a-library-for-causal-inference/?lang=ja) - For decades, causal inference methods have found wide applicability in the social and biomedical sci...
12. [Type 2 diabetes and cancer: umbrella review of meta-analyses of observational studies](https://www.bmj.com/content/350/bmj.g7607) - Objectives To summarise the evidence and evaluate the validity of the associations between type 2 di...
13. [Type 2 Diabetes and Cancer: An Umbrella Review of Observational ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC9398112/) - Type 2 diabetes mellitus (T2DM) has been associated with an increased risk of developing several com...
14. [Diabetes and cancer risk: A Mendelian randomization study](https://onlinelibrary.wiley.com/doi/abs/10.1002/ijc.32310) - Earlier cohort studies using conventional regression models have consistently shown an increased can...
15. [A Mendelian randomization study of type 2 diabetes and ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12320361/) - Our research aims to explore genetic correlation between T2D predisposition and risks of several can...
16. [Association Between Diabetes and Site‐Specific Cancer ...](https://onlinelibrary.wiley.com/doi/10.1155/jdr/1271189) - The associations between each studied factor and site-specific cancer risk were assessed using Cox p...
17. [Survival Tree Analysis of Interactions Among Factors ...](https://publichealth.jmir.org/2025/1/e62756) - by STY Yau · 2025 · Cited by 6 — This study suggests the interaction patterns among age, sex, waist-...
18. [Mendelian randomisation analysis to discover plasma metabolites mediating the effect of obesity on cancer risk](https://www.nature.com/articles/s41416-025-03170-7) - Obesity is a risk factor for several cancers, but the mechanistic basis is poorly understood. We sou
19. [Prevalent diabetes and risk of total, colorectal, prostate and breast cancers in an ageing population: meta-analysis of individual participant data from cohorts of the CHANCES consortium](https://www.nature.com/articles/s41416-021-01347-4) - We investigated whether associations between prevalent diabetes and cancer risk are pertinent to old
20. [Metformin Treatment and Cancer Risk: Cox Regression ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6768811/) - by R Dankner · 2019 · Cited by 50 — Our results do not support an association between metformin trea...
21. [Diabetes Mellitus and Pancreatic Cancer: Investigation of ...](https://www.scienceopen.com/hosted-document?doi=10.15212/bioi-2023-0014) - by Z Deng · 2023 · Cited by 10 — This study was aimed at investigating the association between diabe...
22. [Diabetes and tumor risk: a 23-year Danish national cohort ...](https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1725065/full) - IntroductionDiabetes mellitus is a recognized risk factor for cancer, yet the relationship between d...
23. [A Prospective Study of the Associations Between Treated Diabetes ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC3241297/) - To quantify the association of treated diabetes with cancer incidence and cancer mortality as well a...
24. [Unraveling the link between diabetes and cancer: separating signal from ...](https://academic.oup.com/jnci/article/117/12/2422/8300119?guestAccessKey=) - As the global prevalence of diabetes continues to rise, understanding its broader health implication...
25. [DAGs and GRaSP Causal Inference Algorithms Combined ...](https://www.mdpi.com/1099-4300/28/5/506) - by R Contreras-Jiménez · 2026 — This study evaluates a causal inference-based framework for insulin ...
26. [Investigating the validity of structure learning algorithms in identifying risk factors for intervention in patients with diabetes](http://arxiv.org/abs/2403.14327) - Diabetes, a pervasive and enduring health challenge, imposes significant global implications on heal...
27. [Directed Acyclic Graphs in Decision-Analytic Modeling](https://journals.sagepub.com/doi/10.1177/0272989X241310898) - by SW Dijk · 2025 · Cited by 12 — The DAG's ability to visually depict causal pathways helps modeler...
28. [Making causal inference easy — DoWhy documentation - PyWhy](https://www.pywhy.org/dowhy/v0.2/index.html)
29. [Diabetes and cancer: Optimising glycaemic control](https://onlinelibrary.wiley.com/doi/abs/10.1111/jhn.13051) - ## Abstract Diabetes and cancer are both common and increasingly prevalent conditions, but emerging...
30. [Graph Autoencoder and StrNN based Causal Analysis of ...](https://www.biorxiv.org/content/10.1101/2024.11.11.622921v3.full-text) - In this paper, based on the concepts of causal DAG extraction using Graph AutoEncoders and NCM learn...
31. [Associations of Type 2 Diabetes with risk of overall and site-specific ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12284803/) - The prevalence of type 2 diabetes (T2D) is higher in Black than white Americans, and individuals wit...
32. [[PDF] MULTI-AGENT CAUSAL DISCOVERY USING LARGE LANGUAGE ...](https://openreview.net/pdf?id=Idygh9MX0N)
33. [Published in Transactions on Machine Learning Research (11/2024)](https://openreview.net/pdf?id=EDHQDsqiSe)
