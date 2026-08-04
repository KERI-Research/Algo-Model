import { Notice } from "./common.jsx";

/**
 * "How the AI works" - a professor-facing explanation page.
 *
 * Reached from prominent entry points on Overview and Evidence & Methods rather
 * than from the sidebar, so the five working sections stay uncluttered.
 *
 * All content is static and deliberately plain-language. It must never describe
 * an output as a diagnosis, a disease probability or a cancer-type claim.
 */

/** The single sentence that governs every other statement on this page. */
export const LONGITUDINAL_LIMIT_STATEMENT =
	"The current model is not trained on longitudinal data. It cannot estimate a patient's probability of developing cancer over time, predict which cancer they will develop, or provide a diagnosis.";

const PIPELINE_STEPS = [
	{
		id: "inputs",
		label: "Inputs",
		title: "Prevention-safe clinical and metabolic inputs",
		body: "Twenty-five allowlisted fields: age, sex, ethnicity code, BMI and waist, HbA1c, glucose, insulin, C-peptide, lipids, hs-CRP, haemoglobin, platelets, liver and kidney chemistry, smoking, alcohol and weight change. Anything that reveals an outcome - a recorded diagnosis, a label-derived field, a post-diagnosis TCGA column - is refused as an input.",
	},
	{
		id: "preprocessing",
		label: "Preprocessing",
		title: "Train-only preprocessing and missingness indicators",
		body: "Medians, scaling centres and category levels were fitted on the training split alone, then frozen. A missing value is filled with the training median and, crucially, flagged with its own \"was missing\" indicator, so the model can see that a measurement was absent instead of being told a fabricated value is real.",
	},
	{
		id: "training",
		label: "Training",
		title: "Self-supervised denoising encoder, trained without labels",
		body: "The network was trained to rebuild its own inputs after they were noised and partly masked. No cancer or diabetes label took part in fitting, so the model never learned \"who has the disease\" - it learned what an ordinary adult metabolic profile looks like.",
	},
	{
		id: "representation",
		label: "Representation",
		title: "16-dimensional representation and a reconstruction-based deviation score",
		body: "Each record is compressed to sixteen numbers and rebuilt. What the model cannot rebuild is what it has rarely seen. The deviation score combines that reconstruction error (70%) with how far the sixteen-number encoding sits from the reference centre (30%), floored at zero.",
	},
	{
		id: "reference",
		label: "Reference",
		title: "Reference-cohort percentile and top contributing features",
		body: "The score is ranked inside the distribution of 63,041 scored adult NHANES records, giving a percentile, and the features with the largest reconstruction error are listed. Those features say which measurements the model could not reproduce - not what caused anything.",
	},
	{
		id: "clustering",
		label: "Clustering",
		title: "Exploratory clustering behind stability and negative-control gates",
		body: "Candidate groupings must survive bootstrap resampling, repeated random seeds and negative controls before they may be reported. On the current data every candidate tracked the NHANES survey cycle rather than metabolism, so the pipeline abstains.",
	},
	{
		id: "evidence",
		label: "Evidence",
		title: "Source-linked biomarker evidence for interpretation",
		body: "Interpretation is anchored to a catalogue of published biomarker findings, each with a resolvable source and an evidence grade. The catalogue describes what others have shown; it never reports this project's own performance, and causal status stays \"not established\".",
	},
	{
		id: "review",
		label: "Review",
		title: "Clinician review, and abstention when the data cannot support a claim",
		body: "Every output is a research signal for a clinician to interpret. Where the data cannot support a statement - future risk horizons, cluster meaning, causal direction - the system returns an explicit abstention instead of a weaker answer.",
	},
];

const COMPARISON_ROWS = [
	{
		dimension: "Study design",
		current: "Cross-sectional NHANES survey cycles.",
		future: "Prospective cohort with repeated pre-diagnosis measurements per participant.",
	},
	{
		dimension: "Measurements per person",
		current: "One set of measurements, taken at a single visit.",
		future: "Serial measurements over years, so a trajectory can be modelled.",
	},
	{
		dimension: "Outcome information",
		current: "Recorded conditions were already present when the blood was drawn.",
		future: "Dated incident cancer outcomes, recorded after the baseline measurement.",
	},
	{
		dimension: "Outcome detail",
		current: "None usable: no site, no stage, no date of onset.",
		future: "Cancer site and stage, with the diagnosis date.",
	},
	{
		dimension: "Model output",
		current: "Representation and deviation score only.",
		future: "Time-to-event risk over defined horizons, if and only if validation supports it.",
	},
	{
		dimension: "Associations",
		current: "Post-hoc and cross-sectional; they describe an already-present condition.",
		future: "Prespecified analyses on pre-diagnosis samples, blinded to outcome.",
	},
	{
		dimension: "Data splitting",
		current: "Random splits of independent records.",
		future: "Patient-level temporal splits, so no future information leaks backwards.",
	},
	{
		dimension: "Risk horizon",
		current: "None. No cancer horizon is estimated or reported.",
		future: "1, 3 and 5 year horizons, each gated on sufficient observed events.",
	},
	{
		dimension: "Calibration",
		current: "Not applicable: there is no probability to calibrate.",
		future: "Calibration curves and recalibration on held-out time periods.",
	},
	{
		dimension: "Clinical utility",
		current: "Not assessed.",
		future: "Decision-curve or net-benefit analysis at clinically agreed thresholds.",
	},
	{
		dimension: "External validation",
		current: "None.",
		future: "Independent cohorts, different sites and assay platforms, PRoBE-compliant specimens.",
	},
];

const CAPABILITIES = [
	{
		capability: "Metabolic deviation score for a submitted record",
		tier: "usable_now",
		state: "Available now",
		note: "Descriptive only: how unusual the profile is versus the reference cohort.",
	},
	{
		capability: "Reference-cohort percentile",
		tier: "usable_now",
		state: "Available now",
		note: "Rank within the 63,041-record NHANES adult reference distribution.",
	},
	{
		capability: "Top contributing features for a record",
		tier: "usable_now",
		state: "Available now",
		note: "Reconstruction diagnostics, not causes.",
	},
	{
		capability: "16-dimensional representation",
		tier: "usable_now",
		state: "Available now",
		note: "Useful for similarity and cohort description work.",
	},
	{
		capability: "Data reliability audit and feature tiers",
		tier: "usable_now",
		state: "Available now",
		note: "Coverage, plausibility and availability per feature.",
	},
	{
		capability: "De-identified CSV screening and batch deviation summaries",
		tier: "usable_now",
		state: "Available now",
		note: "Processed in memory and discarded; no retention.",
	},
	{
		capability: "Cross-sectional associations with already-present conditions",
		tier: "qualified_use",
		state: "Research only",
		note: "Descriptive, unadjusted for survey weights, not evidence of early detection.",
	},
	{
		capability: "Exploratory metabolic phenotype clustering",
		tier: "qualified_use",
		state: "Research only",
		note: "Currently abstains (no_stable_clusters); no cluster may be named as a disease.",
	},
	{
		capability: "Biomarker evidence catalogue for interpretation",
		tier: "qualified_use",
		state: "Research only",
		note: "Published findings from other groups, graded and source-linked.",
	},
	{
		capability: "Probability of developing cancer over time",
		tier: "unavailable",
		state: "Unavailable until longitudinal validation",
		note: "Requires dated incident outcomes after baseline measurement. Not attempted.",
	},
	{
		capability: "Which cancer type or site would develop",
		tier: "unavailable",
		state: "Unavailable until longitudinal validation",
		note: "Requires site-labelled incident outcomes. Not attempted.",
	},
	{
		capability: "1, 3 or 5 year risk horizons",
		tier: "unavailable",
		state: "Unavailable until longitudinal validation",
		note: "Fail-closed: no horizon passes the event-count gate on cross-sectional data.",
	},
	{
		capability: "Calibrated risk probabilities",
		tier: "unavailable",
		state: "Unavailable until longitudinal validation",
		note: "There is no outcome-time structure to calibrate against.",
	},
	{
		capability: "Diagnosis, screening or triage",
		tier: "unavailable",
		state: "Unavailable until longitudinal validation",
		note: "Out of scope for this project: a prospective, regulated study would be required.",
	},
];

const READING_OUTPUTS = [
	{
		id: "deviation",
		term: "A high deviation score",
		plain:
			"Says the profile is unusual compared with the reference cohort: the model rarely saw combinations like it. It does not mean a high chance of cancer. Unusualness can come from genuine metabolic strain, from an uncommon but healthy physiology, from a different population, or from a measurement or unit problem.",
	},
	{
		id: "percentile",
		term: "The percentile",
		plain:
			"Places that score in a queue of 63,041 scored adult records. A percentile of 90 means the profile is more unusual than 90% of that reference sample. It is a position in a distribution, not a probability and not a risk category.",
	},
	{
		id: "contributions",
		term: "Feature contributions",
		plain:
			"Are the measurements the model was least able to reproduce for this record. They point at where the unusualness sits, so a clinician knows what to look at. They are not causes, and they are not ranked by clinical importance.",
	},
	{
		id: "abstention",
		term: "no_stable_clusters",
		plain:
			"Is the clustering pipeline refusing to report groups. Candidate groupings dissolved when the data were resampled, and they lined up with which NHANES survey cycle a record came from - assay methods and availability changed between cycles. A boundary that follows collection cycles cannot be read as biology, so the honest output is no output.",
	},
];

export default function HowItWorks({ onNavigate }) {
	return (
		<>
			<Notice kind="blocked" title="What this model cannot do.">
				<span data-testid="text-longitudinal-limit">{LONGITUDINAL_LIMIT_STATEMENT}</span>
			</Notice>

			<p>
				MetaboGuard learns what ordinary adult metabolic profiles look like, then
				flags profiles that do not fit that pattern. Everything below follows from
				that one idea, and from the fact that the training data record each person
				only once.
			</p>

			<section className="card" aria-labelledby="pipeline-heading">
				<h2 id="pipeline-heading">The pipeline, step by step</h2>
				<ol className="flow" data-testid="pipeline-flow">
					{PIPELINE_STEPS.map((step, index) => (
						<li className="flow-step" key={step.id} data-testid={`flow-step-${step.id}`}>
							<div className="flow-marker" aria-hidden="true">
								<span className="flow-number">{index + 1}</span>
							</div>
							<div className="flow-body">
								<span className="flow-label">{step.label}</span>
								<h3>{step.title}</h3>
								<p>{step.body}</p>
							</div>
						</li>
					))}
				</ol>
				<p className="field-hint">
					Steps 1 to 5 run for every record you score. Step 6 runs only in research
					analysis and currently abstains. Steps 7 and 8 are human work, and they are
					not optional.
				</p>
			</section>

			<section className="card" aria-labelledby="reading-heading">
				<h2 id="reading-heading">Reading the outputs in plain language</h2>
				<dl className="plain-list">
					{READING_OUTPUTS.map((item) => (
						<div key={item.id} data-testid={`plain-${item.id}`}>
							<dt>{item.term}</dt>
							<dd>{item.plain}</dd>
						</div>
					))}
				</dl>
			</section>

			<section className="card" aria-labelledby="comparison-heading">
				<h2 id="comparison-heading">Current model vs a future longitudinal model</h2>
				<p>
					The right-hand column is not a roadmap and nothing in it is scheduled. It
					lists what a study would have to provide before any risk-over-time claim
					could be made at all.
				</p>
				<div className="table-wrap">
					<table className="stack-table" data-testid="comparison-table">
						<caption>
							Why longitudinal data is the blocker: every row on the right is
							missing from the data the current model was trained on.
						</caption>
						<thead>
							<tr>
								<th scope="col">Dimension</th>
								<th scope="col">Current model (available now)</th>
								<th scope="col">Required for a future longitudinal model</th>
							</tr>
						</thead>
						<tbody>
							{COMPARISON_ROWS.map((row) => (
								<tr key={row.dimension}>
									<th scope="row" className="wrap" data-label="Dimension">
										{row.dimension}
									</th>
									<td className="wrap" data-label="Current model">
										{row.current}
									</td>
									<td className="wrap" data-label="Required for a longitudinal model">
										{row.future}
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</section>

			<section className="card" aria-labelledby="tcga-heading">
				<h2 id="tcga-heading">Where TCGA fits, and where it does not</h2>
				<p>
					TCGA data come from people who already had a cancer diagnosis when their
					samples were taken. They are used for <strong>post-diagnosis biological
					context only</strong>: understanding pathways and marker behaviour in
					established disease, and sanity-checking what the literature describes.
				</p>
				<ul>
					<li>
						TCGA columns are never model inputs. Every <code>tcga_*</code> field is on
						the input denylist and is reported as <em>prohibited</em> during dataset
						screening.
					</li>
					<li>
						TCGA is never used for prevention scoring. Learning from samples taken
						after diagnosis and applying it to apparently healthy people is exactly
						the leakage this project refuses.
					</li>
					<li>
						TCGA cannot supply a risk horizon. Its follow-up starts at diagnosis, so
						it says nothing about the years before one.
					</li>
				</ul>
				<Notice kind="caution" title="Two separate worlds.">
					Prevention work uses label-free, pre-diagnosis measurements. TCGA describes
					established disease. Results from one are never presented as evidence about
					the other.
				</Notice>
			</section>

			<section className="card" aria-labelledby="capability-heading">
				<h2 id="capability-heading">Capability status</h2>
				<p>
					Three states only. Nothing moves between them automatically: an item in the
					third state stays there until a longitudinal study exists, is validated and
					is reviewed.
				</p>
				<div className="table-wrap">
					<table className="stack-table" data-testid="capability-table">
						<caption>
							Capability status for this deployment. "Unavailable until
							longitudinal validation" is a statement about missing data, not a
							release schedule.
						</caption>
						<thead>
							<tr>
								<th scope="col">Capability</th>
								<th scope="col">Status</th>
								<th scope="col">Note</th>
							</tr>
						</thead>
						<tbody>
							{CAPABILITIES.map((item) => (
								<tr key={item.capability}>
									<th scope="row" className="wrap" data-label="Capability">
										{item.capability}
									</th>
									<td data-label="Status">
										<span
											className="badge"
											data-tier={item.tier}
											data-testid={`capability-state-${item.tier}`}
										>
											{item.state}
										</span>
									</td>
									<td className="wrap" data-label="Note">
										{item.note}
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</section>

			<div className="actions">
				<button
					type="button"
					className="btn btn-secondary"
					onClick={() => onNavigate("overview")}
					data-testid="button-back-overview"
				>
					Back to overview
				</button>
				<button
					type="button"
					className="btn-quiet"
					onClick={() => onNavigate("evidence")}
					data-testid="button-to-evidence"
				>
					See the evidence catalogue and claim boundaries
				</button>
			</div>
		</>
	);
}

export { CAPABILITIES, COMPARISON_ROWS, PIPELINE_STEPS, READING_OUTPUTS };
