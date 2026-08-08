import { useEffect, useState } from "react";
import { apiGet } from "../lib/api.js";
import {
	DefinitionList,
	Empty,
	ErrorState,
	Loading,
	Notice,
	Stat,
} from "./common.jsx";

export default function Overview({ onUnauthorised, onNavigate }) {
	const [data, setData] = useState(null);
	const [error, setError] = useState(null);

	const load = () => {
		setError(null);
		setData(null);
		apiGet("/api/v1/overview")
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

	if (error) {
		return <ErrorState message={error} onRetry={load} />;
	}
	if (!data) {
		return <Loading label="Loading deployment status" rows={5} />;
	}

	const model = data.model;
	const architecture = model.architecture || {};

	return (
		<>
			<Notice
				kind="info"
				title="Non-diagnostic discovery research."
			>
				{data.posture.statement}
			</Notice>

			<section
				className="explainer-cta"
				aria-labelledby="explainer-heading"
			>
				<div>
					<h2 id="explainer-heading">
						How the AI works
					</h2>
					<p>
						The pipeline step by step, what
						a deviation score does and does
						not mean, and why synthetic
						simulation does not remove the
						longitudinal validation barrier
						for patient risk claims.
					</p>
				</div>
				<button
					type="button"
					className="btn"
					onClick={() =>
						onNavigate && onNavigate("how")
					}
					data-testid="button-open-how-it-works"
				>
					Read the explanation
				</button>
			</section>

			<div className="grid grid-4">
				{data.status_cards.map((card) => (
					<Stat
						key={card.id}
						label={card.label}
						value={card.value}
						detail={card.detail}
						state={card.state}
					/>
				))}
			</div>

			<section
				className="card"
				aria-labelledby="model-heading"
			>
				<h2 id="model-heading">Model and capability</h2>
				<DefinitionList
					items={[
						["Model", model.model_name],
						[
							"Model version",
							model.model_version,
						],
						[
							"Code version",
							model.code_version,
						],
						[
							"Deployment",
							model.deployment_version,
						],
						["Trained", model.created_at],
						[
							"Inference backend",
							`${model.inference_backend} (exported weights)`,
						],
						[
							"Training rows",
							(
								model.training_rows ||
								0
							).toLocaleString(
								"en-GB",
							),
						],
						[
							"Reference rows scored",
							(
								model.reference_rows_scored ||
								0
							).toLocaleString(
								"en-GB",
							),
						],
						[
							"Intended use",
							model.intended_use,
						],
					]}
				/>
			</section>

			<div className="grid grid-2">
				<section
					className="card"
					aria-labelledby="architecture-heading"
				>
					<h2 id="architecture-heading">
						Architecture
					</h2>
					<DefinitionList
						items={[
							[
								"Type",
								architecture.type,
							],
							[
								"Input features",
								architecture.input_features,
							],
							[
								"Encoded dimension",
								architecture.transformed_dimension,
							],
							[
								"Latent dimension",
								architecture.latent_dimension,
							],
							[
								"Hidden layers",
								Array.isArray(
									architecture.hidden_layers,
								)
									? architecture.hidden_layers.join(
											" - ",
										)
									: architecture.hidden_layers,
							],
							[
								"Activation",
								architecture.activation,
							],
							[
								"Objective",
								architecture.objective,
							],
							[
								"Deviation score",
								architecture.score_definition,
							],
						]}
					/>
					<p
						className="field-hint"
						style={{
							marginTop: "var(--space-3)",
						}}
					>
						{architecture.reference_rows}
					</p>
				</section>

				<section
					className="card"
					aria-labelledby="claims-heading"
				>
					<h2 id="claims-heading">
						What this deployment reports
					</h2>
					<h4>Supported outputs</h4>
					<ul>
						{model.supported_outputs.map(
							(item) => (
								<li key={item}>
									<code>
										{
											item
										}
									</code>
								</li>
							),
						)}
					</ul>
					<h4>Never reported</h4>
					<ul>
						{model.prohibited_outputs.map(
							(item) => (
								<li key={item}>
									{item}
								</li>
							),
						)}
					</ul>
				</section>
			</div>

			<section
				className="card"
				aria-labelledby="limits-heading"
			>
				<h2 id="limits-heading">
					Current data limitations
				</h2>
				{data.data_limitations.length === 0 ? (
					<Empty>No limitations recorded.</Empty>
				) : (
					<div className="table-wrap">
						<table>
							<caption>
								These
								constraints set
								the ceiling on
								every claim
								below.
							</caption>
							<thead>
								<tr>
									<th scope="col">
										Limitation
									</th>
									<th scope="col">
										Consequence
									</th>
								</tr>
							</thead>
							<tbody>
								{data.data_limitations.map(
									(
										item,
									) => (
										<tr
											key={
												item.id
											}
										>
											<th
												scope="row"
												className="wrap"
											>
												{
													item.title
												}
											</th>
											<td className="wrap">
												{
													item.detail
												}
											</td>
										</tr>
									),
								)}
							</tbody>
						</table>
					</div>
				)}
			</section>

			<Notice
				kind="caution"
				title="Clinician review required."
			>
				{model.non_diagnostic_warning}
			</Notice>
		</>
	);
}
