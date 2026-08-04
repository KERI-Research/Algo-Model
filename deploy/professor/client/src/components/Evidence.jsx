import { useEffect, useMemo, useState } from "react";
import { apiGet } from "../lib/api.js";
import {
	Empty,
	ErrorState,
	Loading,
	Notice,
	Stat,
	TierBadge,
} from "./common.jsx";

export default function Evidence({ onUnauthorised, onNavigate }) {
	const [data, setData] = useState(null);
	const [error, setError] = useState(null);
	const [query, setQuery] = useState("");

	const load = () => {
		setError(null);
		setData(null);
		apiGet("/api/v1/evidence")
			.then(setData)
			.catch((failure) => {
				if (failure.status === 401) {
					onUnauthorised();
					return;
				}
				setError(failure.message);
			});
	};

	useEffect(load, []);

	const entries = useMemo(() => {
		if (!data) {
			return [];
		}
		const needle = query.trim().toLowerCase();
		if (!needle) {
			return data.clinician_ready_entries;
		}
		return data.clinician_ready_entries.filter((entry) =>
			[
				entry.marker_or_panel,
				entry.cancer_site,
				entry.evidence_grade,
				entry.current_data_column,
			]
				.filter(Boolean)
				.some((value) => String(value).toLowerCase().includes(needle)),
		);
	}, [data, query]);

	if (error) {
		return <ErrorState message={error} onRetry={load} />;
	}
	if (!data) {
		return <Loading label="Loading evidence catalogue" rows={6} />;
	}

	const summary = data.summary;

	return (
		<>
			<Notice kind="info" title="Published evidence, not our own performance.">
				{summary.disclaimer}
			</Notice>

			<div className="grid grid-4">
				<Stat label="Catalogue entries" value={summary.entry_count} state="ok" />
				<Stat
					label="Source-linked"
					value={summary.doctor_facing_ready}
					detail="Entry has a resolvable URL or DOI and a graded strength."
					state="ok"
				/>
				<Stat
					label="Research only"
					value={summary.research_only}
					detail="Missing source or ungraded: not presentable to a clinician."
					state="warn"
				/>
				<Stat
					label="Available in current data"
					value={summary.entries_available_in_current_data}
					detail="Marker has a matching column in the analytic dataset."
				/>
			</div>

			<section className="card" aria-labelledby="panel-heading">
				<h2 id="panel-heading">Multi-marker rationale</h2>
				<p>{data.panel_framing}</p>
				<ul className="tag-list">
					{Object.entries(summary.evidence_grades).map(([grade, count]) => (
						<li key={grade}>
							<span className="badge">
								{grade.replace(/_/g, " ")}: {count}
							</span>
						</li>
					))}
				</ul>
			</section>

			<section className="card" aria-labelledby="catalogue-heading">
				<h2 id="catalogue-heading">Biomarker catalogue</h2>
				<label htmlFor="evidence-filter">Filter by marker, site, grade or column</label>
				<input
					id="evidence-filter"
					type="text"
					value={query}
					onChange={(event) => setQuery(event.target.value)}
					placeholder="e.g. CA19-9, pancreas, HbA1c"
					style={{ maxWidth: 360 }}
					data-testid="input-evidence-filter"
				/>
				{entries.length === 0 ? (
					<div style={{ marginTop: "var(--space-4)" }}>
						<Empty testId="empty-evidence">
							No catalogue entry matches "{query}". Clear the filter to see all
							source-linked entries.
						</Empty>
					</div>
				) : (
					<div
						className="table-wrap"
						style={{ marginTop: "var(--space-4)", maxHeight: 520, overflowY: "auto" }}
					>
						<table>
							<caption>
								Every row shown here carries a source link and an evidence
								grade. Causal status is "not established" for all of them.
							</caption>
							<thead>
								<tr>
									<th scope="col">Marker or panel</th>
									<th scope="col">Site</th>
									<th scope="col">Evidence grade</th>
									<th scope="col">Study design</th>
									<th scope="col">In current data</th>
									<th scope="col">Source</th>
								</tr>
							</thead>
							<tbody>
								{entries.map((entry) => (
									<tr key={entry.entry_id}>
										<th scope="row" className="wrap">
											{entry.marker_or_panel}
										</th>
										<td>{entry.cancer_site || "-"}</td>
										<td className="wrap">
											{String(entry.evidence_grade || "").replace(
												/_/g,
												" ",
											)}
										</td>
										<td className="wrap">{entry.study_design || "-"}</td>
										<td>
											<TierBadge
												tier={
													entry.available_in_current_data
														? "usable_now"
														: "unavailable"
												}
											/>
										</td>
										<td>
											{entry.primary_source_url ? (
												<a
													href={entry.primary_source_url}
													target="_blank"
													rel="noreferrer noopener"
												>
													Source
												</a>
											) : entry.doi ? (
												<a
													href={`https://doi.org/${entry.doi}`}
													target="_blank"
													rel="noreferrer noopener"
												>
													{entry.doi}
												</a>
											) : (
												"-"
											)}
										</td>
									</tr>
								))}
							</tbody>
						</table>
					</div>
				)}

				{data.research_only_entries.length ? (
					<Notice kind="caution" title="Research-only rows.">
						{data.research_only_entries.length} entr
						{data.research_only_entries.length === 1 ? "y" : "ies"} lack a
						resolvable source or a graded strength, so they are excluded from any
						clinician-facing statement:{" "}
						{data.research_only_entries
							.map((entry) => entry.marker_or_panel)
							.join(", ")}
						.
					</Notice>
				) : null}
			</section>

			<section className="explainer-cta" aria-labelledby="explainer-evidence-heading">
				<div>
					<h2 id="explainer-evidence-heading">How the AI works</h2>
					<p>
						How the model is trained and scored, the capability status table, and
						why longitudinal data is the blocker.
					</p>
				</div>
				<button
					type="button"
					className="btn btn-secondary"
					onClick={() => onNavigate && onNavigate("how")}
					data-testid="button-open-how-it-works-evidence"
				>
					Read the explanation
				</button>
			</section>

			<section className="card" aria-labelledby="methods-heading">
				<h2 id="methods-heading">Methods and reporting standards</h2>
				<div className="table-wrap">
					<table>
						<thead>
							<tr>
								<th scope="col">Standard</th>
								<th scope="col">Why it matters here</th>
								<th scope="col">Reference</th>
							</tr>
						</thead>
						<tbody>
							{data.method_references.map((reference) => (
								<tr key={reference.id}>
									<th scope="row" className="wrap">
										{reference.title}
										<div
											className="field-hint"
											style={{ marginTop: 2, fontWeight: 400 }}
										>
											{reference.citation}
										</div>
									</th>
									<td className="wrap">{reference.why_it_matters}</td>
									<td>
										<a
											href={reference.url}
											target="_blank"
											rel="noreferrer noopener"
										>
											Open
										</a>
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</section>

			<div className="grid grid-2">
				<section className="card" aria-labelledby="supported-heading">
					<h2 id="supported-heading">Claims this deployment supports</h2>
					<ul className="claim-list" data-kind="supported">
						{data.supported_claims.map((claim) => (
							<li key={claim}>
								<span className="claim-mark" aria-hidden="true">
									OK
								</span>
								<span>{claim}</span>
							</li>
						))}
					</ul>
				</section>
				<section className="card" aria-labelledby="prohibited-heading">
					<h2 id="prohibited-heading">Claims that are prohibited</h2>
					<ul className="claim-list" data-kind="prohibited">
						{data.prohibited_claims.map((claim) => (
							<li key={claim}>
								<span className="claim-mark" aria-hidden="true">
									NO
								</span>
								<span>{claim}</span>
							</li>
						))}
					</ul>
				</section>
			</div>

			{data.denied_statements && data.denied_statements.length ? (
				<section className="card" aria-labelledby="denied-heading">
					<h2 id="denied-heading">Denied wording from the evidence contract</h2>
					<p className="field-hint" style={{ marginTop: 0 }}>
						Phrases the catalogue explicitly forbids, with the entry that forbids
						them.
					</p>
					<div className="table-wrap" style={{ maxHeight: 320, overflowY: "auto" }}>
						<table>
							<thead>
								<tr>
									<th scope="col">Entry</th>
									<th scope="col">Statement that must not be made</th>
								</tr>
							</thead>
							<tbody>
								{data.denied_statements.slice(0, 40).map((item, index) => (
									<tr key={`${item.entry_id}-${index}`}>
										<th scope="row">
											<code>{item.entry_id}</code>
										</th>
										<td className="wrap">{item.statement}</td>
									</tr>
								))}
							</tbody>
						</table>
					</div>
				</section>
			) : null}
		</>
	);
}
