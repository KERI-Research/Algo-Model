/**
 * Simulation-only future-risk panel.
 *
 * Rules enforced in this component:
 *   - Every estimate is labelled SIMULATION ONLY and is never called a patient's risk.
 *   - A horizon row is either a simulated calibrated estimate or an explicit abstention.
 *   - Cross-sectional input is refused: future risk needs at least two distinct visit times.
 *   - The real patient future-risk endpoint stays disabled (409) and is reported as such.
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";

import {
	fetchFutureRiskCapability,
	fetchSimulatedFutureRisk,
} from "./service";
import {
	generateSyntheticHistory,
	HISTORY_FEATURES,
	HISTORY_SAFETY_NOTE,
	isLongitudinal,
} from "./synthetic_history";

const HORIZON_ROWS = [
	{ key: "1y", label: "1 year" },
	{ key: "3y", label: "3 years" },
	{ key: "5y", label: "5 years" },
];

const OUTCOMES = [
	{ id: "type2_diabetes", label: "Type 2 diabetes" },
	{ id: "pan_cancer", label: "Cancer (pan-cancer composite)" },
];

const percent = (value) =>
	typeof value === "number" && Number.isFinite(value)
		? `${(value * 100).toFixed(2)}%`
		: "n.a.";


/**
 * Pure horizon-row builder: each row is either a simulated calibrated estimate or an explicit
 * abstention. Kept pure so the abstention contract is unit-testable without a DOM.
 */
export const buildHorizonRows = ({ scores, capability, outcome }) => {
	const selection =
		(capability && capability.artifact && capability.artifact.selection) || {};
	return HORIZON_ROWS.map((row) => {
		const gated = Boolean(selection[`${outcome}:${row.key}`]);
		const entry = scores && scores.horizons ? scores.horizons[row.key] : null;
		const selected = entry ? entry.selected_model : null;
		const model = selected && entry.models ? entry.models[selected] : null;
		if (!gated || !model) {
			return {
				...row,
				status: "abstained",
				model: null,
				raw: null,
				calibrated: null,
				note: gated
					? "Abstained: no simulated estimate available for this horizon."
					: "Abstained: this horizon did not pass the 50-event gate, so no estimate is produced.",
			};
		}
		return {
			...row,
			status: "simulated_estimate",
			model: selected,
			raw: model.raw_cumulative_incidence,
			calibrated: model.calibrated_cumulative_incidence,
			note: "Simulated estimate (synthetic data only)",
		};
	});
};

export const FutureRiskPanel = () => {
	const [capability, setCapability] = useState(null);
	const [history, setHistory] = useState(() => generateSyntheticHistory({ seed: 1 }));
	const [seed, setSeed] = useState(1);
	const [outcome, setOutcome] = useState("type2_diabetes");
	const [scores, setScores] = useState(null);
	const [messages, setMessages] = useState([]);
	const [loading, setLoading] = useState(false);

	useEffect(() => {
		let active = true;
		fetchFutureRiskCapability()
			.then((payload) => {
				if (active) setCapability(payload);
			})
			.catch((error) => {
				if (active) setMessages([`Capability report unavailable: ${error.message}`]);
			});
		return () => {
			active = false;
		};
	}, []);

	const regenerate = useCallback(() => {
		const nextSeed = seed + 1;
		setSeed(nextSeed);
		setHistory(generateSyntheticHistory({ seed: nextSeed }));
		setScores(null);
	}, [seed]);

	const longitudinal = useMemo(() => isLongitudinal(history.visits), [history]);

	const score = useCallback(async () => {
		if (!longitudinal) {
			setMessages([
				"Refused: a single visit is cross-sectional. Future risk requires at least two distinct visit times.",
			]);
			return;
		}
		setLoading(true);
		setMessages([]);
		try {
			const payload = await fetchSimulatedFutureRisk({
				simulation_mode: true,
				outcome,
				patient_history: history.visits,
			});
			setScores(payload);
		} catch (error) {
			setScores(null);
			setMessages([`Simulated scoring unavailable: ${error.message}`]);
		} finally {
			setLoading(false);
		}
	}, [history, longitudinal, outcome]);

	const horizonRows = useMemo(
		() => buildHorizonRows({ scores, capability, outcome }),
		[scores, capability, outcome],
	);

	return (
		<section className="future-risk-panel" aria-labelledby="future-risk-heading">
			<h2 id="future-risk-heading">Future-risk simulation</h2>

			<p className="simulation-banner" role="note">
				SIMULATION ONLY — synthetic longitudinal data, software verification only. These
				estimates are not validated for patient risk, are not a diagnosis, and are not
				evidence of early detection.
			</p>

			{capability && (
				<dl className="future-risk-capability">
					<dt>Real patient future risk</dt>
					<dd data-testid="clinical-status">
						Disabled (HTTP 409).{" "}
						{capability.clinical_future_risk_blocker}
					</dd>
					<dt>Simulated future risk</dt>
					<dd>
						{capability.simulated_future_risk_enabled ? "Enabled" : "Disabled"} —
						requires {capability.simulated_future_risk_requires}.
					</dd>
					<dt>Event gate</dt>
					<dd>
						{capability.event_gate.minimum_events} events and{" "}
						{capability.event_gate.minimum_non_events} non-events per horizon.
					</dd>
					<dt>Permanently disabled outcomes</dt>
					<dd>{(capability.disabled_outcomes || []).join(", ") || "none"}</dd>
				</dl>
			)}

			<div className="future-risk-controls">
				<label htmlFor="future-risk-outcome">Outcome</label>
				<select
					id="future-risk-outcome"
					value={outcome}
					onChange={(event) => {
						setOutcome(event.target.value);
						setScores(null);
					}}
				>
					{OUTCOMES.map((entry) => (
						<option key={entry.id} value={entry.id}>
							{entry.label}
						</option>
					))}
				</select>
				<button type="button" onClick={regenerate}>
					Generate synthetic history
				</button>
				<button type="button" onClick={score} disabled={loading}>
					{loading ? "Scoring…" : "Score simulated history"}
				</button>
				<span className="synthetic-indicator" data-testid="synthetic-indicator">
					Synthetic research data
				</span>
			</div>

			<p className="research-note">{HISTORY_SAFETY_NOTE}</p>

			<table className="future-risk-history" data-testid="history-table">
				<caption>
					Synthetic visit history — archetype {history.archetypeLabel}, seed{" "}
					{history.seed}, {history.visits.length} visits
				</caption>
				<thead>
					<tr>
						<th scope="col">Years before index</th>
						{HISTORY_FEATURES.map((feature) => (
							<th key={feature} scope="col">
								{feature}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{history.visits.map((visit) => (
						<tr key={visit.visit_index}>
							<th scope="row">{visit.years_before_index}</th>
							{HISTORY_FEATURES.map((feature) => (
								<td key={feature}>
									{visit[feature] === null || visit[feature] === undefined
										? "not measured"
										: visit[feature]}
								</td>
							))}
						</tr>
					))}
				</tbody>
			</table>

			{!longitudinal && (
				<p className="future-risk-abstain" role="alert">
					Cross-sectional input refused: at least two distinct visit times are required.
				</p>
			)}

			<table className="future-risk-estimates" data-testid="estimates-table">
				<caption>
					Simulated cumulative incidence by horizon — each row is either a simulated
					calibrated estimate or an explicit abstention
				</caption>
				<thead>
					<tr>
						<th scope="col">Horizon</th>
						<th scope="col">Simulated calibrated estimate</th>
						<th scope="col">Raw (uncalibrated)</th>
						<th scope="col">Model</th>
						<th scope="col">Status</th>
					</tr>
				</thead>
				<tbody>
					{horizonRows.map((row) => (
						<tr key={row.key} data-testid={`horizon-${row.key}`}>
							<th scope="row">{row.label}</th>
							<td>
								{row.status === "abstained" ? "abstained" : percent(row.calibrated)}
							</td>
							<td>{row.status === "abstained" ? "abstained" : percent(row.raw)}</td>
							<td>{row.model || "—"}</td>
							<td>{row.note}</td>
						</tr>
					))}
				</tbody>
			</table>

			{scores && (
				<div className="future-risk-context">
					<p>
						<strong>Competing outcomes:</strong> {scores.competing_outcomes_note}
					</p>
					<p>
						<strong>Calibration state:</strong> {scores.calibration_state}
					</p>
					<p>
						<strong>Uncertainty:</strong> point estimates only. Confidence intervals for
						discrimination and calibration are reported per horizon in the run artifact
						(<code>results.json</code>), not per individual synthetic history.
					</p>
					<p className="simulation-banner">{scores.banner}</p>
				</div>
			)}

			{messages.length > 0 && (
				<ul className="future-risk-messages" role="status">
					{messages.map((message) => (
						<li key={message}>{message}</li>
					))}
				</ul>
			)}
		</section>
	);
};

export default FutureRiskPanel;