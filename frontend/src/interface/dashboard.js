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
	fetchBiomarkerDiscovery,
	fetchDatasetCatalog,
	fetchDatasetPreview,
	fetchModelResults,
	fetchPredictiveBaseline,
} from "./service";

const INITIAL_PATIENT_FORM = {
	Diabetes: "",
	DEMO_RIDAGEYR: "",
	DEMO_RIAGENDR: "",
	BMX_BMXBMI: "",
	BMX_BMXWAIST: "",
	DIQ_DID040: "",
	GHB_LBXGH: "",
	GLU_LBXGLU: "",
	INS_LBXIN: "",
};

const PATIENT_FIELD_ALIASES = {
	age: "DEMO_RIDAGEYR",
	age_years: "DEMO_RIDAGEYR",
	sex: "DEMO_RIAGENDR",
	gender: "DEMO_RIAGENDR",
	bmi: "BMX_BMXBMI",
	waist: "BMX_BMXWAIST",
	waist_circumference: "BMX_BMXWAIST",
	diabetes: "Diabetes",
	diabetes_onset_age: "DIQ_DID040",
	hba1c: "GHB_LBXGH",
	a1c: "GHB_LBXGH",
	glucose: "GLU_LBXGLU",
	fasting_glucose: "GLU_LBXGLU",
	insulin: "INS_LBXIN",
};

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
	const [biomarkerResults, setBiomarkerResults] = useState(null);
	const [biomarkerLoading, setBiomarkerLoading] = useState(false);
	const [biomarkerError, setBiomarkerError] = useState(null);
	const [patientForm, setPatientForm] = useState(INITIAL_PATIENT_FORM);
	const [uploadStatus, setUploadStatus] = useState(null);

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
			const runDirectionalAnalysis = async (
				treatment,
				outcome,
			) => {
				const strictRequest = {
					treatment,
					outcome,
					allowFallback: !strictCausalMode,
				};

				try {
					return await fetchModelResults(
						selectedDataset,
						strictRequest,
					);
				} catch (analysisError) {
					const isStrictCompatibilityFailure =
						strictCausalMode &&
						typeof analysisError?.message ===
							"string" &&
						analysisError.message.includes(
							"fallback was disabled",
						);

					if (!isStrictCompatibilityFailure) {
						throw analysisError;
					}

					const fallbackResult =
						await fetchModelResults(
							selectedDataset,
							{
								treatment,
								outcome,
								allowFallback: true,
							},
						);

					return {
						...fallbackResult,
						warnings: [
							...(fallbackResult.warnings ||
								[]),
							"Strict causal mode failed due to DoWhy runtime compatibility. Automatically switched to fallback estimate for this run.",
						],
					};
				}
			};

			const [
				diabetesToCancer,
				cancerToDiabetes,
				predictiveBaseline,
			] = await Promise.all([
				runDirectionalAnalysis("Diabetes", "Cancer"),
				runDirectionalAnalysis("Cancer", "Diabetes"),
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

	const formatModelLabel = (modelName) =>
		String(modelName)
			.replaceAll("_", " ")
			.replaceAll(" v1", "")
			.replace(/\b\w/g, (character) =>
				character.toUpperCase(),
			);

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

	const handlePatientFieldChange = (field, value) => {
		setPatientForm((current) => ({
			...current,
			[field]: value,
		}));
	};

	const buildPatientRecord = () => {
		const payload = {};
		Object.entries(patientForm).forEach(([field, rawValue]) => {
			if (rawValue === "") {
				return;
			}

			const numericValue = Number(rawValue);
			payload[field] = Number.isNaN(numericValue)
				? rawValue
				: numericValue;
		});
		return payload;
	};

	const normalizeFieldName = (field) =>
		PATIENT_FIELD_ALIASES[String(field).trim().toLowerCase()] ||
		String(field).trim();

	const normalizeFieldValue = (field, value) => {
		if (value === null || value === undefined) {
			return "";
		}

		const textValue = String(value).trim();
		if (textValue === "") {
			return "";
		}

		if (field === "DEMO_RIAGENDR") {
			const lowered = textValue.toLowerCase();
			if (lowered === "male") {
				return "1";
			}
			if (lowered === "female") {
				return "2";
			}
		}

		if (field === "Diabetes") {
			const lowered = textValue.toLowerCase();
			if (["yes", "true", "positive"].includes(lowered)) {
				return "1";
			}
			if (["no", "false", "negative"].includes(lowered)) {
				return "0";
			}
		}

		return textValue;
	};

	const parseCsvLine = (line) => {
		const values = [];
		let current = "";
		let inQuotes = false;

		for (let index = 0; index < line.length; index += 1) {
			const character = line[index];
			if (character === '"') {
				if (inQuotes && line[index + 1] === '"') {
					current += '"';
					index += 1;
				} else {
					inQuotes = !inQuotes;
				}
				continue;
			}

			if (character === "," && !inQuotes) {
				values.push(current);
				current = "";
				continue;
			}

			current += character;
		}

		values.push(current);
		return values;
	};

	const parseUploadedRecord = (rawText, fileName) => {
		const trimmed = rawText.trim();
		if (!trimmed) {
			throw new Error("Uploaded file is empty.");
		}

		const lowerName = fileName.toLowerCase();
		if (
			lowerName.endsWith(".json") ||
			trimmed.startsWith("{") ||
			trimmed.startsWith("[")
		) {
			const parsed = JSON.parse(trimmed);
			if (Array.isArray(parsed)) {
				if (
					parsed.length === 0 ||
					typeof parsed[0] !== "object"
				) {
					throw new Error(
						"JSON array upload must contain at least one object record.",
					);
				}
				return parsed[0];
			}
			if (typeof parsed !== "object") {
				throw new Error(
					"JSON upload must be an object or an array of objects.",
				);
			}
			return parsed;
		}

		const lines = trimmed.split(/\r?\n/).filter(Boolean);
		if (lines.length < 2) {
			throw new Error(
				"CSV upload must include a header row and at least one data row.",
			);
		}

		const headers = parseCsvLine(lines[0]).map((header) =>
			header.trim(),
		);
		const values = parseCsvLine(lines[1]);
		const record = {};
		headers.forEach((header, index) => {
			record[header] = values[index] ?? "";
		});
		return record;
	};

	const applyUploadedRecord = (record) => {
		setPatientForm((current) => {
			const next = { ...current };
			Object.entries(record).forEach(([field, value]) => {
				const normalizedField =
					normalizeFieldName(field);
				if (!(normalizedField in next)) {
					return;
				}
				next[normalizedField] = normalizeFieldValue(
					normalizedField,
					value,
				);
			});
			return next;
		});
	};

	const handlePatientUpload = async (event) => {
		const file = event.target.files?.[0];
		if (!file) {
			return;
		}

		setUploadStatus(null);
		setBiomarkerError(null);
		try {
			const rawText = await file.text();
			const record = parseUploadedRecord(rawText, file.name);
			applyUploadedRecord(record);
			setUploadStatus(
				`Loaded patient record from ${file.name}. Recognized fields were mapped into the biomarker form.`,
			);
		} catch (uploadError) {
			setUploadStatus(null);
			setBiomarkerError(uploadError.message);
		} finally {
			event.target.value = "";
		}
	};

	const runBiomarkerDiscovery = async (forceRetrain = false) => {
		setBiomarkerLoading(true);
		setBiomarkerError(null);
		try {
			const result = await fetchBiomarkerDiscovery(
				selectedDataset,
				{
					patientRecord: buildPatientRecord(),
					topK: 8,
					forceRetrain,
				},
			);
			setBiomarkerResults(result);
		} catch (biomarkerRequestError) {
			setBiomarkerError(biomarkerRequestError.message);
			setBiomarkerResults(null);
		} finally {
			setBiomarkerLoading(false);
		}
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
						rows, then run causal analysis
						alongside a biomarker discovery
						workflow for local NHANES risk
						scoring.
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

						<article className="panel biomarker-input-panel">
							<div className="panel-header">
								<div>
									<h2>
										Biomarker
										Probe
									</h2>
									<p>
										Enter
										a
										patient-style
										record.
										Missing
										required
										fields
										trigger
										mandatory
										follow-up
										questions;
										optional
										fields
										are
										only
										requested
										when
										confidence
										is
										too
										low.
									</p>
								</div>
								{biomarkerLoading && (
									<span className="status">
										Running
										biomarker
										model...
									</span>
								)}
							</div>
							<label className="upload-control">
								<span>
									Upload
									CSV or
									JSON
									record
								</span>
								<input
									type="file"
									accept=".csv,.json,application/json,text/csv"
									onChange={
										handlePatientUpload
									}
								/>
							</label>
							{uploadStatus && (
								<p className="alert alert-success">
									{
										uploadStatus
									}
								</p>
							)}
							<div className="patient-form-grid">
								<label>
									<span>
										Diabetes
									</span>
									<select
										value={
											patientForm.Diabetes
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"Diabetes",
												event
													.target
													.value,
											)
										}
									>
										<option value="">
											Unknown
										</option>
										<option value="1">
											Yes
										</option>
										<option value="0">
											No
										</option>
									</select>
								</label>
								<label>
									<span>
										Age
									</span>
									<input
										type="number"
										value={
											patientForm.DEMO_RIDAGEYR
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"DEMO_RIDAGEYR",
												event
													.target
													.value,
											)
										}
									/>
								</label>
								<label>
									<span>
										Sex
										Code
									</span>
									<select
										value={
											patientForm.DEMO_RIAGENDR
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"DEMO_RIAGENDR",
												event
													.target
													.value,
											)
										}
									>
										<option value="">
											Unknown
										</option>
										<option value="1">
											Male
										</option>
										<option value="2">
											Female
										</option>
									</select>
								</label>
								<label>
									<span>
										BMI
									</span>
									<input
										type="number"
										step="0.1"
										value={
											patientForm.BMX_BMXBMI
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"BMX_BMXBMI",
												event
													.target
													.value,
											)
										}
									/>
								</label>
								<label>
									<span>
										Waist
									</span>
									<input
										type="number"
										step="0.1"
										value={
											patientForm.BMX_BMXWAIST
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"BMX_BMXWAIST",
												event
													.target
													.value,
											)
										}
									/>
								</label>
								<label>
									<span>
										Diabetes
										Onset
										Age
									</span>
									<input
										type="number"
										value={
											patientForm.DIQ_DID040
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"DIQ_DID040",
												event
													.target
													.value,
											)
										}
									/>
								</label>
								<label>
									<span>
										HbA1c
									</span>
									<input
										type="number"
										step="0.1"
										value={
											patientForm.GHB_LBXGH
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"GHB_LBXGH",
												event
													.target
													.value,
											)
										}
									/>
								</label>
								<label>
									<span>
										Fasting
										Glucose
									</span>
									<input
										type="number"
										step="0.1"
										value={
											patientForm.GLU_LBXGLU
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"GLU_LBXGLU",
												event
													.target
													.value,
											)
										}
									/>
								</label>
								<label>
									<span>
										Insulin
									</span>
									<input
										type="number"
										step="0.1"
										value={
											patientForm.INS_LBXIN
										}
										onChange={(
											event,
										) =>
											handlePatientFieldChange(
												"INS_LBXIN",
												event
													.target
													.value,
											)
										}
									/>
								</label>
							</div>
							<div className="controls-row">
								<button
									type="button"
									onClick={() =>
										runBiomarkerDiscovery(
											false,
										)
									}
									disabled={
										biomarkerLoading
									}
								>
									{biomarkerLoading
										? "Running biomarker model..."
										: "Run Biomarker Model"}
								</button>
								<button
									type="button"
									className="secondary-button"
									onClick={() =>
										runBiomarkerDiscovery(
											true,
										)
									}
									disabled={
										biomarkerLoading
									}
								>
									Retrain
									Artifact
								</button>
							</div>
							{biomarkerError && (
								<p className="alert alert-error">
									Biomarker
									error:{" "}
									{
										biomarkerError
									}
								</p>
							)}
						</article>
					</div>

					<div className="right-column">
						{biomarkerResults && (
							<article className="panel biomarker-result-panel">
								<h2>
									Biomarker
									Discovery
								</h2>
								<p className="meta-text">
									Local
									tabular
									model
									trained
									on
									NHANES
									with
									ChromaDB-backed
									case
									retrieval.
								</p>
								<div className="metric-grid">
									<div className="metric-item">
										<span>
											AUROC
										</span>
										<strong>
											{formatMetric(
												biomarkerResults
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
												biomarkerResults
													.metrics
													.auprc,
											)}
										</strong>
									</div>
									<div className="metric-item">
										<span>
											Cases
											in
											Memory
										</span>
										<strong>
											{
												biomarkerResults
													.memory
													.stored_cases
											}
										</strong>
									</div>
									<div className="metric-item">
										<span>
											Rows
											Used
										</span>
										<strong>
											{
												biomarkerResults
													.cohort_summary
													.rows_used
											}
										</strong>
									</div>
								</div>

								{Object.keys(
									biomarkerResults.benchmarks ||
										{},
								).length >
									0 && (
									<div className="biomarker-list">
										<p className="section-label">
											Model
											Benchmarks
										</p>
										<div className="direction-card-grid">
											{Object.entries(
												biomarkerResults.benchmarks,
											).map(
												([
													modelName,
													metrics,
												]) => (
													<div
														key={
															modelName
														}
														className="direction-card"
													>
														<p className="section-label">
															{formatModelLabel(
																modelName,
															)}
														</p>
														<p
															className={
																modelName ===
																biomarkerResults.model
																	? "mode-badge mode-causal"
																	: "mode-badge mode-neutral"
															}
														>
															{modelName ===
															biomarkerResults.model
																? "Selected model"
																: "Benchmark candidate"}
														</p>
														<div className="metric-grid">
															<div className="metric-item">
																<span>
																	AUROC
																</span>
																<strong>
																	{formatMetric(
																		metrics.auroc,
																	)}
																</strong>
															</div>
															<div className="metric-item">
																<span>
																	AUPRC
																</span>
																<strong>
																	{formatMetric(
																		metrics.auprc,
																	)}
																</strong>
															</div>
														</div>
													</div>
												),
											)}
										</div>
									</div>
								)}

								<div className="biomarker-list">
									<p className="section-label">
										Top
										Biomarkers
									</p>
									{biomarkerResults.biomarker_ranking.map(
										(
											item,
										) => (
											<div
												key={
													item.feature
												}
												className="biomarker-item"
											>
												<div>
													<strong>
														{
															item.feature
														}
													</strong>
													<p className="meta-text">
														{item.direction.replaceAll(
															"_",
															" ",
														)}
													</p>
												</div>
												<div className="biomarker-stats">
													<span>
														Importance{" "}
														{formatMetric(
															item.importance,
														)}
													</span>
													<span>
														Shift{" "}
														{formatMetric(
															item.mean_shift,
														)}
													</span>
												</div>
											</div>
										),
									)}
								</div>

								{biomarkerResults.patient_assessment && (
									<div className="patient-assessment">
										<p className="section-label">
											Patient
											Assessment
										</p>
										<p className="meta-text">
											Status:{" "}
											{
												biomarkerResults
													.patient_assessment
													.status
											}
										</p>
										{biomarkerResults
											.patient_assessment
											.cancer_risk_probability !==
											undefined && (
											<p className="ate-value biomarker-score">
												{(
													biomarkerResults
														.patient_assessment
														.cancer_risk_probability *
													100
												).toFixed(
													1,
												)}

												%
											</p>
										)}
										<p className="meta-text">
											Confidence:{" "}
											{
												biomarkerResults
													.patient_assessment
													.confidence_label
											}
										</p>
										{biomarkerResults
											.patient_assessment
											.explanation && (
											<p className="meta-text">
												{
													biomarkerResults
														.patient_assessment
														.explanation
												}
											</p>
										)}
										{biomarkerResults
											.patient_assessment
											.follow_up_questions
											?.length >
											0 && (
											<div className="follow-up-list">
												{biomarkerResults.patient_assessment.follow_up_questions.map(
													(
														question,
													) => (
														<p
															key={
																question
															}
															className="alert alert-warning compact-alert"
														>
															{
																question
															}
														</p>
													),
												)}
											</div>
										)}
									</div>
								)}
							</article>
						)}

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
