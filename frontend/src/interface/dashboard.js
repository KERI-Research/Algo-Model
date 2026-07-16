/**
 * Causal Inference Dashboard Interface
 * ====================================
 * This component establishes the primary user interface for the causal inference pipeline.
 * Taking a forward-thinking view, this is designed to be data-agnostic—whether it's fed
 * NHANES today or UK Biobank tomorrow, the UI dynamically renders the assumed DAG,
 * the Average Treatment Effect (ATE), and the refutation stress-test results.
 * By decoupling the React frontend from the Python backend, we ensure the heavy ML
 * computation doesn't block the main thread, maintaining a fluid user experience.
 */

import React, { useEffect, useState } from "react";
import {
	fetchDatasetCatalog,
	fetchDatasetPreview,
	fetchModelResults,
	fetchPredictiveBaseline,
} from "./service";

const CausalDashboard = () => {
	const [datasets, setDatasets] = useState([]);
	const [selectedDataset, setSelectedDataset] =
		useState("nhanes_merged.csv");
	const [datasetPreview, setDatasetPreview] = useState(null);
	const [directionResults, setDirectionResults] = useState(null);
	const [loading, setLoading] = useState(false);
	const [previewLoading, setPreviewLoading] = useState(false);
	const [error, setError] = useState(null);
	const [previewError, setPreviewError] = useState(null);
	const [strictCausalMode, setStrictCausalMode] = useState(false);
	const [predictiveResults, setPredictiveResults] = useState(null);

	useEffect(() => {
		const loadDatasets = async () => {
			try {
				const data = await fetchDatasetCatalog();
				const availableDatasets = data.datasets || [];
				setDatasets(availableDatasets);

				if (availableDatasets.length > 0) {
					setSelectedDataset(
						(currentSelection) => {
							const hasCurrent =
								availableDatasets.some(
									(
										dataset,
									) =>
										dataset.name ===
										currentSelection,
								);

							return hasCurrent
								? currentSelection
								: availableDatasets[0]
										.name;
						},
					);
				}
			} catch (catalogError) {
				setError(catalogError.message);
			}
		};

		loadDatasets();
	}, []);

	useEffect(() => {
		const loadPreview = async () => {
			if (!selectedDataset) {
				return;
			}

			setPreviewLoading(true);
			setPreviewError(null);
			try {
				const data =
					await fetchDatasetPreview(
						selectedDataset,
					);
				setDatasetPreview(data);
			} catch (previewRequestError) {
				setDatasetPreview(null);
				setPreviewError(previewRequestError.message);
			} finally {
				setPreviewLoading(false);
			}
		};

		loadPreview();
	}, [selectedDataset]);

	const runAnalysis = async () => {
		setLoading(true);
		setError(null);
		try {
			const [
				diabetesToCancer,
				cancerToDiabetes,
				predictiveBaseline,
			] = await Promise.all([
				fetchModelResults(selectedDataset, {
					treatment: "Diabetes",
					outcome: "Cancer",
					allowFallback: !strictCausalMode,
				}),
				fetchModelResults(selectedDataset, {
					treatment: "Cancer",
					outcome: "Diabetes",
					allowFallback: !strictCausalMode,
				}),
				fetchPredictiveBaseline(selectedDataset),
			]);

			setDirectionResults({
				diabetesToCancer,
				cancerToDiabetes,
			});
			setPredictiveResults(predictiveBaseline);
		} catch (err) {
			setError(err.message);
			setPredictiveResults(null);
		} finally {
			setLoading(false);
		}
	};

	const formatMetric = (value) => {
		const numericValue = Number(value);
		if (Number.isNaN(numericValue)) {
			return "N/A";
		}

		return numericValue.toFixed(3);
	};

	const modeLabel = (result) => {
		if (!result) {
			return "Unknown mode";
		}

		return result.execution_mode === "dowhy_causal"
			? "DoWhy causal estimate"
			: "Fallback association estimate";
	};

	const modeClassName = (result) => {
		if (!result) {
			return "mode-badge mode-neutral";
		}

		return result.execution_mode === "dowhy_causal"
			? "mode-badge mode-causal"
			: "mode-badge mode-fallback";
	};

	const interpretEstimate = (result) => {
		if (!result) {
			return "";
		}

		const estimateValue = Number(result.estimate);
		const treatment = result.treatment || "Treatment";
		const outcome = result.outcome || "Outcome";

		if (Number.isNaN(estimateValue)) {
			return `${treatment} -> ${outcome}: estimate could not be converted to a numeric probability shift.`;
		}

		const percentagePoints = (estimateValue * 100).toFixed(2);
		const absPoints = Math.abs(Number(percentagePoints)).toFixed(2);

		if (Math.abs(estimateValue) < 0.001) {
			return `${treatment} -> ${outcome}: near-zero change (${percentagePoints}% points), suggesting little directional effect in this model.`;
		}

		if (estimateValue > 0) {
			return `${treatment} -> ${outcome}: when ${treatment} is present, ${outcome} is estimated to be ${absPoints}% points more likely.`;
		}

		return `${treatment} -> ${outcome}: when ${treatment} is present, ${outcome} is estimated to be ${absPoints}% points less likely.`;
	};

	return (
		<div className="app-shell">
			<div
				className="background-orb orb-one"
				aria-hidden="true"
			/>
			<div
				className="background-orb orb-two"
				aria-hidden="true"
			/>
			<main className="dashboard-layout">
				<section className="hero-panel">
					<p className="eyebrow">
						KERI RESEARCH CONSOLE
					</p>
					<h1>Causal Inference Engine</h1>
					<p className="hero-copy">
						Choose a dataset, inspect sample
						rows, then execute a causal run
						with transparent estimates and
						refutation checks.
					</p>
				</section>

				<section className="content-grid">
					<div className="left-column">
						<article className="panel control-panel">
							<div>
								<h2>
									Dataset
									Selector
								</h2>
								<p>
									Pick a
									CSV
									source
									for
									preview
									and
									analysis.
								</p>
							</div>
							<div className="controls-row">
								<select
									value={
										selectedDataset
									}
									onChange={(
										event,
									) =>
										setSelectedDataset(
											event
												.target
												.value,
										)
									}
								>
									{datasets.length ===
									0 ? (
										<option value="nhanes_merged.csv">
											nhanes_merged.csv
										</option>
									) : (
										datasets.map(
											(
												dataset,
											) => (
												<option
													key={
														dataset.name
													}
													value={
														dataset.name
													}
												>
													{
														dataset.name
													}
												</option>
											),
										)
									)}
								</select>
								<button
									type="button"
									onClick={
										runAnalysis
									}
									disabled={
										loading
									}
								>
									{loading
										? "Running analysis..."
										: "Execute Causal Model"}
								</button>
							</div>
							<label className="strict-toggle">
								<input
									type="checkbox"
									checked={
										strictCausalMode
									}
									onChange={(
										event,
									) =>
										setStrictCausalMode(
											event
												.target
												.checked,
										)
									}
								/>
								<span>
									Strict
									causal
									mode
									(disable
									fallback)
								</span>
							</label>
							{error && (
								<p className="alert alert-error">
									Analysis
									error:{" "}
									{error}
								</p>
							)}
						</article>

						<article className="panel preview-panel">
							<div className="panel-header">
								<div>
									<h2>
										Dataset
										Preview
									</h2>
									<p>
										Validate
										available
										variables
										before
										you
										estimate
										effects.
									</p>
								</div>
								{previewLoading && (
									<span className="status">
										Loading
										preview...
									</span>
								)}
							</div>
							{previewError && (
								<p className="alert alert-warning">
									Preview
									error:{" "}
									{
										previewError
									}
								</p>
							)}
							{datasetPreview && (
								<div className="preview-content">
									<div className="stats-grid">
										<div className="stat-card">
											<span>
												Dataset
											</span>
											<strong>
												{
													datasetPreview.dataset
												}
											</strong>
										</div>
										<div className="stat-card">
											<span>
												Columns
											</span>
											<strong>
												{
													datasetPreview
														.columns
														.length
												}
											</strong>
										</div>
										<div className="stat-card">
											<span>
												Sample
												Rows
											</span>
											<strong>
												{
													datasetPreview.sample_size
												}
											</strong>
										</div>
									</div>

									<div>
										<p className="section-label">
											Visible
											Columns
										</p>
										<div className="chip-grid">
											{datasetPreview.columns
												.slice(
													0,
													18,
												)
												.map(
													(
														column,
													) => (
														<span
															key={
																column
															}
															className="chip"
														>
															{
																column
															}
														</span>
													),
												)}
										</div>
									</div>

									<div>
										<p className="section-label">
											Sample
											Records
										</p>
										<div className="table-wrap">
											<table>
												<thead>
													<tr>
														{datasetPreview.columns
															.slice(
																0,
																6,
															)
															.map(
																(
																	column,
																) => (
																	<th
																		key={
																			column
																		}
																	>
																		{
																			column
																		}
																	</th>
																),
															)}
													</tr>
												</thead>
												<tbody>
													{datasetPreview.preview.map(
														(
															row,
															rowIndex,
														) => (
															<tr
																key={
																	rowIndex
																}
															>
																{datasetPreview.columns
																	.slice(
																		0,
																		6,
																	)
																	.map(
																		(
																			column,
																		) => (
																			<td
																				key={
																					column
																				}
																			>
																				{String(
																					row[
																						column
																					] ??
																						"",
																				)}
																			</td>
																		),
																	)}
															</tr>
														),
													)}
												</tbody>
											</table>
										</div>
									</div>
								</div>
							)}
						</article>
					</div>

					<div className="right-column">
						{directionResults && (
							<article className="panel result-panel">
								<h2>
									Direction
									Summary
								</h2>
								<div className="direction-card-grid">
									<div className="direction-card">
										<p className="section-label">
											Diabetes
											-&gt;
											Cancer
										</p>
										<p
											className={modeClassName(
												directionResults.diabetesToCancer,
											)}
										>
											{modeLabel(
												directionResults.diabetesToCancer,
											)}
										</p>
										<p className="ate-value">
											{(
												directionResults
													.diabetesToCancer
													.estimate *
												100
											).toFixed(
												2,
											)}

											%
										</p>
										<p className="meta-text">
											{interpretEstimate(
												directionResults.diabetesToCancer,
											)}
										</p>
										{directionResults
											.diabetesToCancer
											.warnings
											?.length >
											0 && (
											<p className="alert alert-warning compact-alert">
												{
													directionResults
														.diabetesToCancer
														.warnings[0]
												}
											</p>
										)}
									</div>
									<div className="direction-card">
										<p className="section-label">
											Cancer
											-&gt;
											Diabetes
										</p>
										<p
											className={modeClassName(
												directionResults.cancerToDiabetes,
											)}
										>
											{modeLabel(
												directionResults.cancerToDiabetes,
											)}
										</p>
										<p className="ate-value">
											{(
												directionResults
													.cancerToDiabetes
													.estimate *
												100
											).toFixed(
												2,
											)}

											%
										</p>
										<p className="meta-text">
											{interpretEstimate(
												directionResults.cancerToDiabetes,
											)}
										</p>
										{directionResults
											.cancerToDiabetes
											.warnings
											?.length >
											0 && (
											<p className="alert alert-warning compact-alert">
												{
													directionResults
														.cancerToDiabetes
														.warnings[0]
												}
											</p>
										)}
									</div>
								</div>
								<p className="meta-text">
									Dataset:{" "}
									{
										selectedDataset
									}
								</p>
								<p className="meta-text">
									A
									positive
									value
									means
									the
									outcome
									is more
									likely
									when the
									treatment
									is
									present;
									a
									negative
									value
									means
									less
									likely.
								</p>
								<p className="meta-text">
									These
									estimates
									show
									directional
									models,
									not
									guaranteed
									clinical
									causation.
								</p>
							</article>
						)}

						{directionResults && (
							<article className="panel predictive-panel">
								<h2>
									Predictive
									Baseline
								</h2>
								<p className="meta-text">
									Imbalance-aware
									metrics
									for the
									70%+
									target.
									Prioritize
									AUPRC,
									recall,
									and
									balanced
									accuracy
									over raw
									accuracy.
								</p>
								{predictiveResults && (
									<>
										<div className="direction-card-grid">
											{[
												{
													label: "Diabetes",
													payload: predictiveResults
														.results
														.diabetes,
												},
												{
													label: "Cancer",
													payload: predictiveResults
														.results
														.cancer,
												},
											].map(
												({
													label,
													payload,
												}) => (
													<div
														key={
															label
														}
														className="direction-card"
													>
														<p className="section-label">
															Predict
															{
																label
															}
														</p>
														<p className="meta-text">
															Rows:
															{
																payload.rows_used
															}

															|
															Features:
															{
																payload
																	.features
																	.length
															}
														</p>
														<div className="metric-grid">
															<div className="metric-item">
																<span>
																	AUROC
																</span>
																<strong>
																	{formatMetric(
																		payload
																			.metrics
																			.auroc,
																	)}
																</strong>
															</div>
															<div className="metric-item">
																<span>
																	AUPRC
																</span>
																<strong>
																	{formatMetric(
																		payload
																			.metrics
																			.auprc,
																	)}
																</strong>
															</div>
															<div className="metric-item">
																<span>
																	Recall
																</span>
																<strong>
																	{formatMetric(
																		payload
																			.metrics
																			.recall,
																	)}
																</strong>
															</div>
															<div className="metric-item">
																<span>
																	Balanced
																	Acc
																</span>
																<strong>
																	{formatMetric(
																		payload
																			.metrics
																			.balanced_accuracy,
																	)}
																</strong>
															</div>
														</div>
														<p className="meta-text">
															Baseline
															AUPRC:
															{formatMetric(
																payload
																	.baseline_metrics
																	.auprc,
															)}
														</p>
													</div>
												),
											)}
										</div>
										{predictiveResults
											.notes?.[0] && (
											<p className="meta-text">
												{
													predictiveResults
														.notes[0]
												}
											</p>
										)}
									</>
								)}
							</article>
						)}

						{directionResults && (
							<article className="panel refute-panel">
								<h2>
									Refutation
									Stress
									Tests
								</h2>
								<p className="section-label">
									Diabetes
									-&gt;
									Cancer
								</p>
								<div className="refute-grid">
									{Object.entries(
										directionResults
											.diabetesToCancer
											.refutations,
									).map(
										([
											label,
											value,
										]) => (
											<div
												key={
													label
												}
												className="refute-item"
											>
												<p>
													{label.replace(
														"_",
														" ",
													)}
												</p>
												<strong>
													{
														value
													}
												</strong>
											</div>
										),
									)}
								</div>
								<p className="section-label">
									Cancer
									-&gt;
									Diabetes
								</p>
								<div className="refute-grid">
									{Object.entries(
										directionResults
											.cancerToDiabetes
											.refutations,
									).map(
										([
											label,
											value,
										]) => (
											<div
												key={`reverse-${label}`}
												className="refute-item"
											>
												<p>
													{label.replace(
														"_",
														" ",
													)}
												</p>
												<strong>
													{
														value
													}
												</strong>
											</div>
										),
									)}
								</div>
							</article>
						)}
					</div>
				</section>
			</main>
		</div>
	);
};

export default CausalDashboard;
