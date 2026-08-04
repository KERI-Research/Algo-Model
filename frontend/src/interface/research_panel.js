/**
 * Research panel: data reliability, exploratory phenotypes and published evidence.
 *
 * Terminology contract enforced in this component:
 *   - "data observation"  = measured in the file (coverage, tiers, counts)
 *   - "model association" = produced by our model on our sample; not validated, not causal
 *   - "published evidence" = catalogued source with URL, design and evidence grade
 *   - "causal claim not established" = default for every mechanism statement
 *
 * A cluster is a patient/metabolic phenotype. It is never labelled as a cancer type,
 * cancer site or disease subtype, and no future-risk probability is displayed.
 */

import React, { useCallback, useEffect, useState } from "react";
import {
	fetchDataReliability,
	fetchEvidenceCatalogue,
	fetchResearchClusters,
} from "./service";

const CLASS_LABELS = {
	data_observation: "Data observation",
	model_association: "Model association",
	published_evidence: "Published evidence",
	causal_claim_not_established: "Causal claim not established",
};

const ClassBadge = ({ kind }) => (
	<span className={`evidence-badge evidence-${kind}`}>
		{CLASS_LABELS[kind] || kind}
	</span>
);

const ResearchPanel = () => {
	const [reliability, setReliability] = useState(null);
	const [clusters, setClusters] = useState(null);
	const [evidence, setEvidence] = useState(null);
	const [variant, setVariant] = useState("complete_cases");
	const [messages, setMessages] = useState([]);
	const [loading, setLoading] = useState(false);

	const load = useCallback(async () => {
		setLoading(true);
		const problems = [];
		try {
			setReliability(await fetchDataReliability());
		} catch (error) {
			problems.push(`Reliability: ${error.message}`);
		}
		try {
			setEvidence(await fetchEvidenceCatalogue());
		} catch (error) {
			problems.push(`Evidence: ${error.message}`);
		}
		try {
			setClusters(await fetchResearchClusters(variant));
		} catch (error) {
			setClusters(null);
			problems.push(`Clusters: ${error.message}`);
		}
		setMessages(problems);
		setLoading(false);
	}, [variant]);

	useEffect(() => {
		load();
	}, [load]);

	return (
		<section className="research-panel">
			<header className="research-header">
				<h2>Research surfaces</h2>
				<p className="research-warning">
					Research use only, non-diagnostic. This panel shows data
					reliability, exploratory metabolic phenotypes and published
					evidence. It does not show a diagnosis, a future disease
					probability or a cancer site. Clinician review is required.
				</p>
				<div className="research-controls">
					<label htmlFor="cluster-variant">Cluster variant</label>
					<select
						id="cluster-variant"
						value={variant}
						onChange={(event) => setVariant(event.target.value)}
					>
						<option value="complete_cases">
							complete cases (missingness controlled)
						</option>
						<option value="all_adults">all adults</option>
					</select>
					<button type="button" onClick={load} disabled={loading}>
						{loading ? "Loading…" : "Refresh"}
					</button>
				</div>
			</header>

			{messages.length > 0 && (
				<ul className="research-messages">
					{messages.map((message) => (
						<li key={message}>{message}</li>
					))}
				</ul>
			)}

			{reliability && (
				<article className="research-card">
					<h3>
						Data reliability <ClassBadge kind="data_observation" />
					</h3>
					<p>
						Dataset <code>{reliability.dataset}</code> — status{" "}
						<strong>{reliability.status}</strong>. Feature eligibility
						tiers gate what may be modelled.
					</p>
					<ul className="tier-list">
						{Object.entries(reliability.tiers || {}).map(
							([tier, features]) => (
								<li key={tier}>
									<strong>{tier}</strong>: {features.length}
									{features.length > 0 && (
										<span className="tier-features">
											{" "}
											— {features.slice(0, 6).join(", ")}
											{features.length > 6
												? ` (+${features.length - 6})`
												: ""}
										</span>
									)}
								</li>
							),
						)}
					</ul>
					{reliability.assay_cycle_drift_summary && (
						<p className="research-note">
							Cycle availability gaps:{" "}
							{(
								reliability.assay_cycle_drift_summary
									.availability_gap_features || []
							).join(", ") || "none"}
							. Level drift:{" "}
							{(
								reliability.assay_cycle_drift_summary
									.level_drift_features || []
							).join(", ") || "none"}
							.
						</p>
					)}
					{reliability.sections?.survey_weights && (
						<p className="research-note">
							Survey weights applied in modelling:{" "}
							{String(
								reliability.sections.survey_weights
									.weights_applied_in_modelling,
							)}
							. Results describe the analytic sample, not the
							population.
						</p>
					)}
				</article>
			)}

			<article className="research-card">
				<h3>
					Exploratory phenotypes <ClassBadge kind="model_association" />
				</h3>
				{!clusters && (
					<p className="research-note">
						No cluster run is available. Run{" "}
						<code>python api/run_research_pass.py</code> to generate
						one. Nothing is shown in place of a real result.
					</p>
				)}
				{clusters && (
					<>
						<p>
							Run <code>{clusters.run}</code>, variant{" "}
							<code>{clusters.variant}</code>, status{" "}
							<strong>{clusters.status}</strong>.
						</p>
						<p className="research-warning">
							{clusters.cluster_naming_policy}
						</p>
						{clusters.status === "no_stable_clusters" && (
							<div className="abstain-box">
								<h4>No stable phenotypes reported</h4>
								<p>{clusters.abstain?.reason}</p>
								<p className="research-note">
									{clusters.abstain?.interpretation}
								</p>
								<ul>
									{Object.entries(
										clusters.abstain?.gate_failure_summary || {},
									).map(([candidate, failures]) => (
										<li key={candidate}>
											<code>{candidate}</code>:{" "}
											{(failures || []).join(", ") || "none"}
										</li>
									))}
								</ul>
							</div>
						)}
						{clusters.status === "stable_clusters_found" && (
							<>
								<p>
									Selected {clusters.selected?.method}, k=
									{clusters.selected?.k} — silhouette{" "}
									{clusters.selected?.train_metrics?.silhouette},
									bootstrap ARI{" "}
									{
										clusters.selected?.bootstrap_stability
											?.mean_ari
									}
									.
								</p>
								<ul className="cluster-list">
									{(clusters.clusters || []).map((cluster) => (
										<li key={cluster.cluster_id}>
											<strong>{cluster.cluster_id}</strong> —{" "}
											{cluster.rows} rows. Panel:{" "}
											{(cluster.top_distinguishing_panel || [])
												.slice(0, 4)
												.map(
													(item) =>
														`${item.feature} ${item.direction}`,
												)
												.join(", ")}
										</li>
									))}
								</ul>
								{clusters.posthoc_label_summary && (
									<p className="research-note">
										{clusters.posthoc_label_summary.warning}
									</p>
								)}
							</>
						)}
						<table className="candidate-table">
							<caption>
								Candidate solutions with stability and negative
								controls. A control above 0.30 marks a data
								artefact.
							</caption>
							<thead>
								<tr>
									<th>Method</th>
									<th>k</th>
									<th>Silhouette</th>
									<th>Bootstrap ARI</th>
									<th>Dominated by</th>
									<th>Gate failures</th>
								</tr>
							</thead>
							<tbody>
								{(clusters.candidate_summary || []).map((item) => (
									<tr key={`${item.method}-${item.k}`}>
										<td>{item.method}</td>
										<td>{item.k}</td>
										<td>{item.silhouette ?? "n.a."}</td>
										<td>{item.bootstrap_mean_ari ?? "n.a."}</td>
										<td>
											{(item.dominated_by || []).join(", ") ||
												"—"}
										</td>
										<td>
											{(item.gate_failures || []).join(", ") ||
												"none"}
										</td>
									</tr>
								))}
							</tbody>
						</table>
					</>
				)}
			</article>

			{evidence && (
				<article className="research-card">
					<h3>
						Published evidence <ClassBadge kind="published_evidence" />{" "}
						<ClassBadge kind="causal_claim_not_established" />
					</h3>
					<p className="research-note">{evidence.panel_framing}</p>
					<ul className="evidence-list">
						{(evidence.clinician_ready_entries || []).map((entry) => (
							<li key={entry.entry_id}>
								<strong>{entry.marker_or_panel}</strong> (
								{entry.cancer_site}) — grade{" "}
								<em>{entry.evidence_grade}</em>, design{" "}
								{entry.study_design}.{" "}
								<span className="evidence-limitation">
									{entry.limitations}
								</span>{" "}
								{entry.primary_source_url &&
									entry.primary_source_url !== "unknown" && (
										<a
											href={entry.primary_source_url}
											target="_blank"
											rel="noreferrer"
										>
											source
										</a>
									)}
							</li>
						))}
					</ul>
					{(evidence.research_only_entries || []).length > 0 && (
						<p className="research-note">
							{evidence.research_only_entries.length} catalogue rows are
							research-only (missing source or ungraded) and are not shown
							to clinicians.
						</p>
					)}
				</article>
			)}
		</section>
	);
};

export default ResearchPanel;