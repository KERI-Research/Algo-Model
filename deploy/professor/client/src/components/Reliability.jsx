import { useEffect, useState } from "react";
import { apiGet } from "../lib/api.js";
import {
	DefinitionList,
	Empty,
	ErrorState,
	Loading,
	Notice,
	Stat,
	TierBadge,
	formatNumber,
	formatPercent,
} from "./common.jsx";

const TIER_ORDER = ["usable_now", "qualified_use", "unavailable", "prohibited"];

export default function Reliability({ onUnauthorised }) {
	const [reliability, setReliability] = useState(null);
	const [clusters, setClusters] = useState(null);
	const [variant, setVariant] = useState("complete_cases");
	const [error, setError] = useState(null);

	const load = (nextVariant = variant) => {
		setError(null);
		Promise.all([
			apiGet("/api/v1/reliability"),
			apiGet(`/api/v1/clusters?variant=${nextVariant}`),
		])
			.then(([reliabilityBody, clusterBody]) => {
				setReliability(reliabilityBody);
				setClusters(clusterBody);
			})
			.catch((failure) => {
				if (failure.status === 401) {
					onUnauthorised();
					return;
				}
				setError(failure.message);
			});
	};

	useEffect(() => {
		load(variant);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [variant]);

	if (error) {
		return <ErrorState message={error} onRetry={() => load()} />;
	}
	if (!reliability || !clusters) {
		return <Loading label="Loading reliability audit" rows={6} />;
	}

	const tiers = reliability.tiers || {};
	const eligibility = Object.entries(reliability.feature_eligibility || {});
	const abstained = clusters.status === "no_stable_clusters";

	return (
		<>
			<Notice kind="info" title="Data observation, not a model claim.">
				{reliability.explanation_classes.data_observation} The audit below is the
				report the research pipeline produced for{" "}
				<code>{reliability.source.dataset}</code>.
			</Notice>

			<div className="grid grid-4">
				{TIER_ORDER.map((tier) => (
					<Stat
						key={tier}
						label={tier.replace(/_/g, " ")}
						value={(tiers[tier] || []).length}
						detail={reliability.tier_definitions[tier]}
						state={
							tier === "usable_now"
								? "ok"
								: tier === "qualified_use"
									? "warn"
									: tier === "prohibited"
										? "blocked"
										: undefined
						}
					/>
				))}
			</div>

			<section className="card" aria-labelledby="failclosed-heading">
				<h2 id="failclosed-heading">Fail-closed controls</h2>
				<ul>
					{reliability.fail_closed_controls.map((item) => (
						<li key={item}>{item}</li>
					))}
				</ul>
				<DefinitionList
					items={[
						["Audit status", reliability.status],
						["Generated", reliability.source.generated_at],
						["Research run", reliability.source.research_run],
						["Blocking violations", (reliability.violations || []).length],
						[
							"Survey weights",
							reliability.sections.survey_weights
								? reliability.sections.survey_weights.note ||
									"Present in file; not applied in the model."
								: "Not applied in the model.",
						],
					]}
				/>
			</section>

			<section className="card" aria-labelledby="tiers-heading">
				<h2 id="tiers-heading">Feature tiers</h2>
				{eligibility.length === 0 ? (
					<Empty>The bundled report contains no feature eligibility table.</Empty>
				) : (
					<div className="table-wrap" style={{ maxHeight: 460, overflowY: "auto" }}>
						<table>
							<caption>
								Every allowlisted and denylisted column with its tier and the
								reason recorded by the pipeline.
							</caption>
							<thead>
								<tr>
									<th scope="col">Column</th>
									<th scope="col">Tier</th>
									<th scope="col" className="num">
										Coverage
									</th>
									<th scope="col">Reason</th>
								</tr>
							</thead>
							<tbody>
								{eligibility.map(([column, value]) => (
									<tr key={column}>
										<th scope="row">
											<code>{column}</code>
										</th>
										<td>
											<TierBadge tier={value.tier} />
										</td>
										<td className="num">
											{value.coverage === null ||
											value.coverage === undefined
												? "-"
												: formatPercent(value.coverage)}
										</td>
										<td className="wrap">
											{(value.reasons || []).join(" ")}
										</td>
									</tr>
								))}
							</tbody>
						</table>
					</div>
				)}
			</section>

			<section className="card" aria-labelledby="clusters-heading">
				<h2 id="clusters-heading">Exploratory phenotype clustering</h2>

				<div className="actions" style={{ marginTop: 0, marginBottom: "var(--space-4)" }}>
					<label htmlFor="cluster-variant" style={{ marginBottom: 0 }}>
						Variant
					</label>
					<select
						id="cluster-variant"
						value={variant}
						onChange={(event) => setVariant(event.target.value)}
						style={{ maxWidth: 220 }}
						data-testid="input-cluster-variant"
					>
						{clusters.available_variants.map((option) => (
							<option key={option} value={option}>
								{option.replace(/_/g, " ")}
							</option>
						))}
					</select>
				</div>

				{abstained ? (
					<Notice kind="blocked" title="Abstained: no_stable_clusters.">
						<span data-testid="text-cluster-status">
							{clusters.abstain.reason}
						</span>
					</Notice>
				) : (
					<Notice kind="caution" title="Solution reported.">
						A candidate passed the gates. It is still a metabolic phenotype, never a
						disease or cancer type.
					</Notice>
				)}

				{abstained ? (
					<>
						<h3>Why survey-cycle effects prevent interpretation</h3>
						<p>{clusters.abstain.survey_cycle_explanation}</p>
						<h3>Gate failures by candidate</h3>
						<div className="table-wrap">
							<table>
								<thead>
									<tr>
										<th scope="col">Candidate</th>
										<th scope="col">Failed gates</th>
									</tr>
								</thead>
								<tbody>
									{Object.entries(
										clusters.abstain.gate_failure_summary || {},
									).map(([candidate, failures]) => (
										<tr key={candidate}>
											<th scope="row">
												<code>{candidate}</code>
											</th>
											<td className="wrap">
												{failures.map((failure) => (
													<span
														className="badge"
														data-tier="prohibited"
														key={failure}
														style={{
															marginRight: 6,
															marginBottom: 4,
														}}
													>
														{failure.replace(/_/g, " ")}
													</span>
												))}
											</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>
					</>
				) : null}

				<h3 style={{ marginTop: "var(--space-5)" }}>Candidate metrics</h3>
				<div className="table-wrap">
					<table>
						<caption>
							Internal metrics with stability and negative-control results. A
							candidate must pass all gates to be reported.
						</caption>
						<thead>
							<tr>
								<th scope="col">Method</th>
								<th scope="col" className="num">
									k
								</th>
								<th scope="col" className="num">
									Silhouette
								</th>
								<th scope="col" className="num">
									Bootstrap ARI
								</th>
								<th scope="col" className="num">
									Seed ARI
								</th>
								<th scope="col">Dominated by</th>
								<th scope="col">Passes gates</th>
							</tr>
						</thead>
						<tbody>
							{clusters.candidate_summary.map((candidate) => (
								<tr key={`${candidate.method}-${candidate.k}`}>
									<th scope="row">{candidate.method}</th>
									<td className="num">{candidate.k}</td>
									<td className="num">
										{formatNumber(candidate.silhouette, 3)}
									</td>
									<td className="num">
										{formatNumber(candidate.bootstrap_mean_ari, 3)}
									</td>
									<td className="num">
										{formatNumber(candidate.seed_mean_ari, 3)}
									</td>
									<td>{candidate.dominated_by || "-"}</td>
									<td>
										<span
											className="badge"
											data-tier={
												candidate.passes_gates
													? "usable_now"
													: "prohibited"
											}
										>
											{candidate.passes_gates ? "yes" : "no"}
										</span>
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>

				<h3 style={{ marginTop: "var(--space-5)" }}>Standing warnings</h3>
				<ul>
					{clusters.warnings.map((item) => (
						<li key={item}>{item}</li>
					))}
				</ul>

				<Notice kind="info" title="Uploaded datasets.">
					Clustering is not run on uploaded data in this deployment. The validated
					pipeline needs bootstrap and seed stability runs plus survey-cycle negative
					controls that exceed the hosting compute budget, and an unvalidated cluster
					must never be presented as a disease group.
				</Notice>
			</section>
		</>
	);
}
