import { useCallback, useEffect, useMemo, useState } from "react";
import { scoreSimulationHistory, simulationCapability } from "../lib/api.js";
import {
	generateSyntheticHistory,
	HISTORY_FEATURES,
	HISTORY_SAFETY_NOTE,
	isLongitudinal,
} from "../lib/synthetic_history.js";
import { ErrorState, formatPercent, Loading, Notice, Stat } from "./common.jsx";

const HORIZONS = [
	{ id: "1y", label: "1 year" },
	{ id: "3y", label: "3 years" },
	{ id: "5y", label: "5 years" },
];

const formatModel = (value) =>
	value ? String(value).replace(/_/g, " ") : "Not selected";

const formatDifference = (value) =>
	typeof value === "number" && Number.isFinite(value)
		? value.toExponential(2)
		: "Not reported";

export const buildSimulationRows = ({ capability, scores, outcome }) =>
	HORIZONS.map((horizon) => {
		const key = `${outcome}:${horizon.id}`;
		const supported = capability?.supported_horizons?.[key];
		const capabilityAbstention =
			capability?.abstained_horizons?.[key];
		const scored =
			scores?.outcomes?.[outcome]?.horizons?.[horizon.id];

		if (!supported || scored?.status === "abstained") {
			return {
				...horizon,
				status: "abstained",
				model: null,
				raw: null,
				calibrated: null,
				note:
					scored?.reason ||
					capabilityAbstention?.reason ||
					"No selected model passed the deployment gate for this horizon.",
			};
		}

		if (!scored) {
			return {
				...horizon,
				status: "ready",
				model: supported.selected_model,
				raw: null,
				calibrated: null,
				note: "Selected simulation model is ready; no history has been scored yet.",
			};
		}

		const modelName = scored.selected_model;
		const model = scored.models?.[modelName];
		if (!model) {
			return {
				...horizon,
				status: "abstained",
				model: null,
				raw: null,
				calibrated: null,
				note: "The selected model returned no portable estimate, so this row abstains.",
			};
		}

		return {
			...horizon,
			status: "simulated_estimate",
			model: modelName,
			raw: model.raw_cumulative_incidence,
			calibrated: model.calibrated_cumulative_incidence,
			note: `Synthetic-data calibration (${model.calibration_method || "reported method"}).`,
		};
	});

export default function FutureRiskSimulation({ onUnauthorised }) {
	const [capability, setCapability] = useState(null);
	const [capabilityError, setCapabilityError] = useState(null);
	const [scoreError, setScoreError] = useState(null);
	const [scoring, setScoring] = useState(false);
	const [seed, setSeed] = useState(1);
	const [visitCount, setVisitCount] = useState(5);
	const [outcome, setOutcome] = useState("type2_diabetes");
	const [history, setHistory] = useState(() =>
		generateSyntheticHistory({ seed: 1, visits: 5 }),
	);
	const [scores, setScores] = useState(null);

	const loadCapability = useCallback(() => {
		setCapabilityError(null);
		setCapability(null);
		simulationCapability()
			.then(setCapability)
			.catch((failure) => {
				if (failure.status === 401) {
					onUnauthorised();
					return;
				}
				setCapabilityError(failure.message);
			});
	}, [onUnauthorised]);

	useEffect(loadCapability, [loadCapability]);

	const longitudinal = useMemo(
		() => isLongitudinal(history.visits),
		[history],
	);
	const rows = useMemo(
		() => buildSimulationRows({ capability, scores, outcome }),
		[capability, scores, outcome],
	);

	const generateHistory = () => {
		const nextSeed = Math.max(1, Math.trunc(Number(seed) || 1));
		const nextVisitCount = Math.min(
			8,
			Math.max(2, Math.trunc(Number(visitCount) || 5)),
		);
		setSeed(nextSeed);
		setVisitCount(nextVisitCount);
		setHistory(
			generateSyntheticHistory({
				seed: nextSeed,
				visits: nextVisitCount,
			}),
		);
		setScores(null);
		setScoreError(null);
	};

	const runSimulation = async () => {
		if (!longitudinal) {
			setScoreError(
				"A future-risk simulation requires at least two distinct visit times.",
			);
			return;
		}
		setScoring(true);
		setScoreError(null);
		try {
			const payload = await scoreSimulationHistory(
				history.visits,
				{
					seed: history.seed,
					archetype: history.archetype,
				},
			);
			setScores(payload);
		} catch (failure) {
			if (failure.status === 401) {
				onUnauthorised();
				return;
			}
			setScores(null);
			setScoreError(failure.message);
		} finally {
			setScoring(false);
		}
	};

	if (capabilityError) {
		return (
			<ErrorState
				message={capabilityError}
				onRetry={loadCapability}
			/>
		);
	}
	if (!capability) {
		return (
			<Loading
				label="Loading simulation capability"
				rows={5}
			/>
		);
	}

	const parity = capability.parity || {};
	const outcomeOptions = capability.outcomes || [];

	return (
		<>
			<Notice kind="blocked" title="Simulation only.">
				Synthetic longitudinal data for software
				verification. Estimates are not validated for
				patient risk, are not a diagnosis, and must not
				inform care.
			</Notice>

			<div className="grid grid-4">
				<Stat
					label="Clinical use"
					value="Prohibited"
					state="blocked"
				/>
				<Stat
					label="Inference"
					value="NumPy portable"
					detail="No sklearn or Torch in the request path."
					state="ok"
				/>
				<Stat
					label="Parity"
					value={parity.verdict || "Not reported"}
					detail={`${parity.histories_compared || 0} histories; max difference ${formatDifference(
						parity.max_abs_difference,
					)}.`}
					state={
						parity.verdict === "parity"
							? "ok"
							: "warn"
					}
				/>
				<Stat
					label="Artifact"
					value={
						capability.portable_artifact_available
							? "Available"
							: "Unavailable"
					}
					detail="Selected synthetic models only."
					state={
						capability.portable_artifact_available
							? "ok"
							: "blocked"
					}
				/>
			</div>

			<section
				className="card"
				aria-labelledby="simulation-input-heading"
			>
				<h2 id="simulation-input-heading">
					Synthetic history
				</h2>
				<p className="field-hint">
					{HISTORY_SAFETY_NOTE}
				</p>

				<div className="simulation-toolbar">
					<div>
						<label htmlFor="simulation-outcome">
							Outcome
						</label>
						<select
							id="simulation-outcome"
							value={outcome}
							onChange={(event) =>
								setOutcome(
									event
										.target
										.value,
								)
							}
						>
							{outcomeOptions.map(
								(entry) => (
									<option
										key={
											entry.id
										}
										value={
											entry.id
										}
										disabled={
											!entry.enabled
										}
									>
										{
											entry.label
										}
									</option>
								),
							)}
						</select>
					</div>
					<div>
						<label htmlFor="simulation-seed">
							History seed
						</label>
						<input
							id="simulation-seed"
							type="number"
							min="1"
							max="999999"
							step="1"
							value={seed}
							onChange={(event) =>
								setSeed(
									event
										.target
										.value,
								)
							}
						/>
					</div>
					<div>
						<label htmlFor="simulation-visits">
							Visits
						</label>
						<input
							id="simulation-visits"
							type="number"
							min="2"
							max="8"
							step="1"
							value={visitCount}
							onChange={(event) =>
								setVisitCount(
									event
										.target
										.value,
								)
							}
						/>
					</div>
					<div className="simulation-actions">
						<button
							type="button"
							className="btn btn-secondary"
							onClick={
								generateHistory
							}
						>
							Generate history
						</button>
						<button
							type="button"
							className="btn"
							onClick={runSimulation}
							disabled={
								scoring ||
								!capability.portable_artifact_available ||
								!longitudinal
							}
						>
							{scoring
								? "Running..."
								: "Run simulation"}
						</button>
					</div>
				</div>

				{scoreError ? (
					<Notice
						kind="blocked"
						title="Simulation unavailable."
					>
						{scoreError}
					</Notice>
				) : null}

				<div className="table-wrap simulation-history-table">
					<table data-testid="simulation-history">
						<caption>
							Synthetic{" "}
							{history.archetypeLabel}{" "}
							history, seed{" "}
							{history.seed},{" "}
							{history.visits.length}{" "}
							visits
						</caption>
						<thead>
							<tr>
								<th scope="col">
									Years
									before
									index
								</th>
								{HISTORY_FEATURES.map(
									(
										feature,
									) => (
										<th
											scope="col"
											key={
												feature
											}
										>
											{
												feature
											}
										</th>
									),
								)}
							</tr>
						</thead>
						<tbody>
							{history.visits.map(
								(visit) => (
									<tr
										key={
											visit.visit_index
										}
									>
										<th
											scope="row"
											className="num"
										>
											{
												visit.years_before_index
											}
										</th>
										{HISTORY_FEATURES.map(
											(
												feature,
											) => (
												<td
													className="num"
													key={
														feature
													}
												>
													{visit[
														feature
													] ??
														"not measured"}
												</td>
											),
										)}
									</tr>
								),
							)}
						</tbody>
					</table>
				</div>
			</section>

			<section
				className="card"
				aria-labelledby="simulation-results-heading"
			>
				<h2 id="simulation-results-heading">
					Simulated cumulative incidence
				</h2>
				<div className="table-wrap">
					<table
						className="stack-table"
						data-testid="simulation-results"
					>
						<caption>
							Each horizon is a
							synthetic estimate, an
							explicit abstention, or
							waiting for a simulation
							run.
						</caption>
						<thead>
							<tr>
								<th scope="col">
									Horizon
								</th>
								<th scope="col">
									Calibrated
								</th>
								<th scope="col">
									Raw
								</th>
								<th scope="col">
									Selected
									model
								</th>
								<th scope="col">
									Status
								</th>
								<th scope="col">
									Boundary
								</th>
							</tr>
						</thead>
						<tbody>
							{rows.map((row) => (
								<tr
									key={
										row.id
									}
									data-testid={`simulation-horizon-${row.id}`}
								>
									<th scope="row">
										{
											row.label
										}
									</th>
									<td
										data-label="Calibrated"
										className="num"
									>
										{row.status ===
										"simulated_estimate"
											? formatPercent(
													row.calibrated,
													2,
												)
											: "-"}
									</td>
									<td
										data-label="Raw"
										className="num"
									>
										{row.status ===
										"simulated_estimate"
											? formatPercent(
													row.raw,
													2,
												)
											: "-"}
									</td>
									<td data-label="Selected model">
										{formatModel(
											row.model,
										)}
									</td>
									<td data-label="Status">
										<span
											className="simulation-status"
											data-state={
												row.status
											}
										>
											{row.status.replace(
												/_/g,
												" ",
											)}
										</span>
									</td>
									<td
										data-label="Boundary"
										className="wrap"
									>
										{
											row.note
										}
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</section>

			{scores ? (
				<>
					<div className="grid grid-3">
						<Stat
							label="History span"
							value={`${(scores.history.history_days / 365.25).toFixed(1)} years`}
							state="ok"
						/>
						<Stat
							label="Visit density"
							value={`${scores.history.visit_density_per_year.toFixed(2)} / year`}
							state="ok"
						/>
						<Stat
							label="Missingness"
							value={formatPercent(
								scores.history
									.missingness_burden,
								1,
							)}
							state="warn"
						/>
					</div>
					<Notice
						kind="info"
						title="Interpretation boundary."
					>
						{scores.interpretation}{" "}
						{scores.persistence}
					</Notice>
				</>
			) : null}

			<section
				className="card"
				aria-labelledby="simulation-caveats-heading"
			>
				<h2 id="simulation-caveats-heading">
					Evaluation caveats
				</h2>
				<ul className="simulation-caveats">
					{(
						capability.evaluation_caveats ||
						[]
					).map((caveat) => (
						<li key={caveat}>{caveat}</li>
					))}
				</ul>
				<p className="field-hint">
					{capability.competing_outcomes}
				</p>
			</section>
		</>
	);
}
