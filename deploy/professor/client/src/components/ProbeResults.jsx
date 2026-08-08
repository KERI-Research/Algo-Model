import { useState } from "react";
import { domainFor, formatProbeValue, labelFor } from "../lib/probe_fields.js";
import { DefinitionList, Notice, Stat, formatNumber } from "./common.jsx";

const EXPECTED_PATHWAY_IDS = [
	"diabetes_related_cancer",
	"lifestyle_related_cancer",
	"cancer_related_diabetes",
	"lifestyle_related_diabetes",
];

const EXPECTED_CANCER_OUTCOME_IDS = [
	"pan_cancer",
	"pancreatic_cancer",
	"other_site_specific_cancers",
];

const hasValidResearchContract = (research) =>
	Array.isArray(research?.cancer_outcomes) &&
	research.cancer_outcomes.length ===
		EXPECTED_CANCER_OUTCOME_IDS.length &&
	research.cancer_outcomes.every(
		(outcome, index) =>
			outcome.id === EXPECTED_CANCER_OUTCOME_IDS[index] &&
			outcome.probability === null &&
			["simulation_only", "not_estimable"].includes(
				outcome.status,
			) &&
			Array.isArray(outcome.available_horizons),
	) &&
	Array.isArray(research?.pathways) &&
	research.pathways.length === EXPECTED_PATHWAY_IDS.length &&
	research.pathways.every(
		(pathway, index) =>
			pathway.id === EXPECTED_PATHWAY_IDS[index] &&
			pathway.status === "not_estimable" &&
			pathway.probability === null &&
			Array.isArray(pathway.observed_standout_features),
	);

const bandState = (band) => {
	if (band === "within_reference_range") {
		return "ok";
	}
	if (band === "mild_deviation" || band === "elevated_deviation") {
		return "warn";
	}
	return "blocked";
};

const FactorRow = ({ entry, submittedForm, topContribution }) => {
	const contribution = Number(entry.reconstruction_error) || 0;
	const relativeWidth = topContribution
		? Math.max(4, (contribution / topContribution) * 100)
		: 0;

	return (
		<li className="probe-factor-row">
			<div className="probe-factor-name">
				<strong>{labelFor(entry.feature)}</strong>
				<span>{domainFor(entry.feature)}</span>
				<code>{entry.feature}</code>
			</div>
			<div className="probe-factor-value">
				<span>Submitted value</span>
				<strong>
					{formatProbeValue(
						entry.feature,
						submittedForm[entry.feature],
					)}
				</strong>
			</div>
			<div className="probe-factor-contribution">
				<span>Reconstruction contribution</span>
				<strong>{formatNumber(contribution, 4)}</strong>
				<span className="meter" aria-hidden="true">
					<span
						style={{
							width: `${relativeWidth}%`,
						}}
					/>
				</span>
			</div>
		</li>
	);
};

const PathwayFactor = ({ entry, submittedForm }) => (
	<li>
		<strong>{labelFor(entry.feature)}</strong>
		<span>
			{formatProbeValue(
				entry.feature,
				submittedForm[entry.feature],
			)}
		</span>
	</li>
);

const CancerOutcomeCapability = ({ outcomes, onNavigate }) => {
	const [selectedId, setSelectedId] = useState(outcomes[0]?.id || "");
	const selected =
		outcomes.find((outcome) => outcome.id === selectedId) ||
		outcomes[0];
	const simulationOnly = selected?.status === "simulation_only";

	return (
		<div
			className="probe-cancer-selector"
			data-testid="cancer-outcome-capability"
		>
			<div>
				<label htmlFor="probe-cancer-outcome">
					Cancer outcome
				</label>
				<select
					id="probe-cancer-outcome"
					value={selectedId}
					onChange={(event) =>
						setSelectedId(
							event.target.value,
						)
					}
					data-testid="select-cancer-outcome"
				>
					{outcomes.map((outcome) => (
						<option
							key={outcome.id}
							value={outcome.id}
						>
							{outcome.label}
						</option>
					))}
				</select>
				<p className="field-hint">
					Only outcomes represented by the
					deployed research artifacts are listed.
				</p>
			</div>
			<div
				className="probe-cancer-capability"
				data-status={selected?.status}
			>
				<div className="probe-pathway-head">
					<h4>{selected?.label}</h4>
					<span
						className="badge"
						data-tier={
							simulationOnly
								? "qualified_use"
								: "unavailable"
						}
					>
						{selected?.availability_label}
					</span>
				</div>
				<DefinitionList
					items={[
						[
							"Patient likelihood",
							"Not estimable from this record",
						],
						[
							"Synthetic horizons",
							selected
								?.available_horizons
								?.length
								? selected.available_horizons.join(
										", ",
									)
								: "None",
						],
					]}
				/>
				<p>{selected?.reason}</p>
				{simulationOnly && onNavigate ? (
					<button
						type="button"
						className="btn btn-secondary"
						onClick={() =>
							onNavigate("simulation")
						}
						data-testid="button-open-simulation"
					>
						Open synthetic simulation
					</button>
				) : null}
			</div>
		</div>
	);
};

export default function ProbeResults({ result, submittedForm, onNavigate }) {
	const score = result?.score;
	const assessment = result?.patient_assessment;
	const current = assessment?.current_profile_assessment;
	const standout = assessment?.standout_factors;
	const readiness = assessment?.data_readiness;
	const research = assessment?.research_association;

	if (!score || !current || !standout || !readiness || !research) {
		return (
			<div data-testid="probe-result-contract-error">
				<Notice
					kind="blocked"
					title="Result contract unavailable."
				>
					The response did not include the
					required assessment sections, so no
					interpretation is shown.
				</Notice>
			</div>
		);
	}

	const submittedValues = submittedForm || {};
	const featuresUsed = Array.isArray(result.features_used)
		? result.features_used
		: [];
	const featuresMissing = Array.isArray(result.features_missing)
		? result.features_missing
		: [];
	const topDeviationFeatures = Array.isArray(
		standout.top_deviation_features,
	)
		? standout.top_deviation_features
		: [];
	const missingFields = Array.isArray(readiness.missing_fields)
		? readiness.missing_fields
		: [];
	const evidenceBoundaries = Array.isArray(result.evidence_boundaries)
		? result.evidence_boundaries
		: [];
	const fieldMeanings = result.field_meanings || {};
	const hasPercentile =
		score.reference_percentile !== null &&
		score.reference_percentile !== undefined &&
		Number.isFinite(Number(score.reference_percentile));
	const percentile = hasPercentile
		? formatNumber(score.reference_percentile, 1)
		: null;
	const researchContractIsValid = hasValidResearchContract(research);
	const topContribution =
		Number(topDeviationFeatures[0]?.reconstruction_error) || 0;
	const totalFeatures = featuresUsed.length + featuresMissing.length;
	const deviationInterpretation = current.deviation_interpretation || {};
	const rangeReview = deviationInterpretation.range_review || {};
	const flaggedValues = Array.isArray(rangeReview.flagged_values)
		? rangeReview.flagged_values
		: [];

	return (
		<>
			<section
				className="probe-assessment"
				data-band={current.deviation_band}
				data-testid="panel-current-profile-assessment"
				aria-labelledby="current-profile-heading"
			>
				<div className="probe-assessment-copy">
					<span className="probe-band-label">
						{current.deviation_band_label}
					</span>
					<h3 id="current-profile-heading">
						{current.section_title}
					</h3>
					<p className="probe-assessment-summary">
						{hasPercentile ? (
							<>
								More unusual
								than{" "}
								<strong>
									{
										percentile
									}
									%
								</strong>{" "}
								of the NHANES
								adult reference.
							</>
						) : (
							"Reference percentile unavailable for this result."
						)}
					</p>
					<p>{current.note}</p>
				</div>
				<div className="probe-assessment-boundary">
					<strong>How to interpret this</strong>
					<span>
						This is an unusualness
						percentile, not a risk
						probability and not a good/bad
						diagnosis.
					</span>
				</div>
			</section>

			<section
				className="card probe-deviation-meaning"
				aria-labelledby="deviation-meaning-heading"
				data-testid="panel-deviation-meaning"
			>
				<h3 id="deviation-meaning-heading">
					What this deviation means
				</h3>
				<div className="probe-meaning-grid">
					<div>
						<span>Pattern rarity</span>
						<strong>
							{hasPercentile
								? `More unusual than ${percentile}% of the reference`
								: "Reference comparison unavailable"}
						</strong>
						<p>
							{
								deviationInterpretation.pattern_meaning
							}
						</p>
					</div>
					<div>
						<span>Better or worse?</span>
						<strong>
							{deviationInterpretation.health_direction_label ||
								"Cannot infer"}
						</strong>
						<p>
							{
								deviationInterpretation.health_direction_note
							}
						</p>
					</div>
					<div>
						<span>
							Do the values look
							plausible?
						</span>
						<strong>
							{rangeReview.status ===
							"review_flagged_values"
								? `Review ${flaggedValues.length} flagged value${flaggedValues.length === 1 ? "" : "s"}`
								: "No broad range flags"}
						</strong>
						<p>{rangeReview.note}</p>
					</div>
				</div>
				<p className="field-hint">
					{
						deviationInterpretation.record_validity_note
					}
				</p>
				{flaggedValues.length ? (
					<ul
						className="probe-range-flags"
						data-testid="list-range-flags"
					>
						{flaggedValues.map((entry) => (
							<li key={entry.feature}>
								<strong>
									{labelFor(
										entry.feature,
									)}
								</strong>
								:{" "}
								{formatProbeValue(
									entry.feature,
									entry.value,
								)}
								; broad range{" "}
								{entry.plausible_range.join(
									" to ",
								)}
							</li>
						))}
					</ul>
				) : null}
			</section>

			<div className="grid grid-3 probe-supporting-stats">
				<Stat
					label="Metabolic deviation score"
					value={formatNumber(
						score.metabolic_deviation_score,
						3,
					)}
					detail="Unitless and unbounded; use the percentile for reference context."
					state={bandState(
						current.deviation_band,
					)}
				/>
				<Stat
					label="Reference percentile"
					value={
						hasPercentile
							? `${percentile}%`
							: "Unavailable"
					}
					detail="Position in the NHANES adult reference, not disease risk."
					state={bandState(
						current.deviation_band,
					)}
				/>
				<Stat
					label="Data supplied"
					value={`${featuresUsed.length} of ${totalFeatures}`}
					detail={`${featuresMissing.length} model features left blank.`}
				/>
			</div>

			<section
				className="probe-context"
				aria-labelledby="submitted-context-heading"
			>
				<div>
					<h3 id="submitted-context-heading">
						Submitted diabetes context
					</h3>
					<p>
						Shown to help frame the research
						questions. These values were not
						sent to or scored by the model.
					</p>
				</div>
				<dl>
					<div>
						<dt>{labelFor("Diabetes")}</dt>
						<dd>
							{formatProbeValue(
								"Diabetes",
								submittedValues.Diabetes,
							)}
						</dd>
					</div>
					<div>
						<dt>
							{labelFor("DIQ_DID040")}
						</dt>
						<dd>
							{formatProbeValue(
								"DIQ_DID040",
								submittedValues.DIQ_DID040,
							)}
						</dd>
					</div>
				</dl>
			</section>

			<section
				className="card"
				data-testid="panel-standout-factors"
				aria-labelledby="standout-factors-heading"
			>
				<h3 id="standout-factors-heading">
					{standout.section_title}
				</h3>
				<p
					className="field-hint"
					style={{ marginTop: 0 }}
				>
					{standout.note} Larger contributions
					identify measurements the model found
					harder to reconstruct.
				</p>
				{topDeviationFeatures.length ? (
					<ol
						className="probe-factor-list"
						role="list"
					>
						{topDeviationFeatures.map(
							(entry) => (
								<FactorRow
									key={
										entry.feature
									}
									entry={
										entry
									}
									submittedForm={
										submittedValues
									}
									topContribution={
										topContribution
									}
								/>
							),
						)}
					</ol>
				) : (
					<p className="probe-pathway-empty">
						No supplied measurement appears
						among this profile's top
						reconstruction deviations.
					</p>
				)}
			</section>

			<section
				className="probe-research-section"
				data-testid="panel-research-association"
				aria-labelledby="research-pathways-heading"
			>
				<div className="probe-section-heading">
					<div>
						<h3 id="research-pathways-heading">
							{research.section_title}
						</h3>
						<p>{research.scope_note}</p>
						<p>{research.note}</p>
					</div>
					<span
						className="badge"
						data-tier="qualified_use"
					>
						Cross-sectional context only
					</span>
				</div>

				{researchContractIsValid ? (
					<>
						<Notice
							kind="info"
							title="Why no patient likelihood appears."
						>
							This probe receives one
							cross-sectional profile.
							It can show which
							measurements are
							unusual, but it cannot
							establish temporal
							order, a causal effect,
							or a future cancer
							likelihood from that
							record.
						</Notice>

						<CancerOutcomeCapability
							outcomes={
								research.cancer_outcomes
							}
							onNavigate={onNavigate}
						/>

						<div className="probe-pathway-grid">
							{research.pathways.map(
								(pathway) => (
									<article
										className="probe-pathway"
										key={
											pathway.id
										}
										data-testid={`pathway-${pathway.id}`}
									>
										<div className="probe-pathway-head">
											<h4>
												{
													pathway.title
												}
											</h4>
											<span
												className="badge"
												data-tier="unavailable"
											>
												Context
												only
											</span>
										</div>
										<p className="probe-pathway-question">
											{
												pathway.question
											}
										</p>
										<p className="probe-pathway-reason">
											{
												pathway.reason
											}
										</p>
										<strong className="probe-pathway-factor-label">
											Supplied
											standout
											measurements
											relevant
											to
											this
											question
										</strong>
										{pathway
											.observed_standout_features
											.length ? (
											<ul
												className="probe-pathway-factors"
												role="list"
											>
												{pathway.observed_standout_features.map(
													(
														entry,
													) => (
														<PathwayFactor
															key={
																entry.feature
															}
															entry={
																entry
															}
															submittedForm={
																submittedValues
															}
														/>
													),
												)}
											</ul>
										) : (
											<p className="probe-pathway-empty">
												No
												matching
												measurement
												appears
												among
												this
												profile's
												top
												deviations.
											</p>
										)}
									</article>
								),
							)}
						</div>
						<p className="field-hint">
							{research.factor_note}
						</p>
					</>
				) : (
					<Notice
						kind="blocked"
						title="Pathway contract unavailable."
					>
						No pathway probability or factor
						is shown because the response
						did not satisfy the fail-closed
						research contract.
					</Notice>
				)}
			</section>

			<section
				className="card"
				data-testid="panel-data-readiness"
				aria-labelledby="data-readiness-heading"
			>
				<div className="probe-section-heading">
					<div>
						<h3 id="data-readiness-heading">
							{
								readiness.section_title
							}
						</h3>
						<p>
							Dataset capability:{" "}
							<strong>
								{
									readiness.dataset_capability_state
								}
							</strong>
						</p>
					</div>
					<span
						className="badge"
						data-tier="qualified_use"
					>
						{featuresMissing.length} missing
					</span>
				</div>
				{missingFields.length ? (
					<ul
						className="probe-readiness-list"
						role="list"
					>
						{missingFields.map((item) => (
							<li key={item.field}>
								<strong>
									{labelFor(
										item.field,
									)}
								</strong>
								<span>
									{
										item.why_it_matters
									}
								</span>
							</li>
						))}
					</ul>
				) : (
					<Notice
						kind="ok"
						title="All model fields supplied."
					>
						No allowlisted model measurement
						was left blank.
					</Notice>
				)}
			</section>

			<div className="grid grid-2">
				<div className="card">
					<h3>How to read the model fields</h3>
					<DefinitionList
						items={[
							[
								"Deviation score",
								fieldMeanings.metabolic_deviation_score,
							],
							[
								"Percentile",
								fieldMeanings.reference_percentile,
							],
							[
								"Contributions",
								fieldMeanings.top_deviation_features,
							],
							[
								"Representation",
								fieldMeanings.latent_representation,
							],
						]}
					/>
				</div>
				<div className="card">
					<h3>Evidence boundaries</h3>
					<ul>
						{evidenceBoundaries.map(
							(item) => (
								<li key={item}>
									{item}
								</li>
							),
						)}
					</ul>
				</div>
			</div>

			<Notice kind="caution" title="Not a diagnosis.">
				{result.non_diagnostic_warning}
			</Notice>
		</>
	);
}
