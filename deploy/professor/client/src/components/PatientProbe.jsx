import { useMemo, useState } from "react";
import { apiPostJson } from "../lib/api.js";
import {
	REQUIRED_PROBE_FIELDS,
	SYNTHETIC_FIELD_SCHEMA,
} from "../lib/synthetic_patient.js";
import {
	generateFullSyntheticProfile,
	PREVENTION_FIELD_SCHEMA,
	SYNTHETIC_NOTE,
} from "../lib/synthetic_prevention.js";
import {
	DefinitionList,
	Empty,
	Notice,
	Stat,
	formatNumber,
} from "./common.jsx";

/** Field order and labels for the probe form. `Diabetes` stays an input, never an output. */
const FIELD_GROUPS = [
	{
		id: "demographics",
		legend: "Demographics and reported status",
		fields: ["DEMO_RIDAGEYR", "DEMO_RIAGENDR", "DEMO_RIDRETH3", "Diabetes", "DIQ_DID040"],
	},
	{
		id: "anthropometry",
		legend: "Anthropometry",
		fields: ["BMX_BMXBMI", "BMX_BMXWAIST", "weight_loss_1yr_lb", "weight_loss_10yr_lb"],
	},
	{
		id: "glycaemia",
		legend: "Glycaemia and insulin",
		fields: ["GHB_LBXGH", "GLU_LBXGLU", "INS_LBXIN", "CPEP_LBXCPSI", "homa_ir"],
	},
	{
		id: "lipids",
		legend: "Lipids and inflammation",
		fields: ["TRIGLY_LBXTR", "TRIGLY_LBDLDL", "HDL_LBDHDD", "TCHOL_LBXTC", "HSCRP_LBXHSCRP"],
	},
	{
		id: "haematology",
		legend: "Haematology and biochemistry",
		fields: [
			"CBC_LBXHGB",
			"CBC_LBXPLTSI",
			"BIOPRO_LBXSATSI",
			"BIOPRO_LBXSAPSI",
			"BIOPRO_LBXSCR",
		],
	},
	{
		id: "lifestyle",
		legend: "Lifestyle",
		fields: ["smoking_status", "alcohol_status", "average_drinks_per_day"],
	},
];

const LABELS = {
	DEMO_RIDAGEYR: "Age",
	DEMO_RIAGENDR: "Sex (NHANES code)",
	Diabetes: "Reported diabetes (input)",
	DIQ_DID040: "Age at diabetes onset",
	BMX_BMXBMI: "BMI",
	BMX_BMXWAIST: "Waist circumference",
	GHB_LBXGH: "HbA1c",
	GLU_LBXGLU: "Fasting glucose",
	INS_LBXIN: "Insulin",
	CPEP_LBXCPSI: "C-peptide",
};

const ALL_FIELDS = FIELD_GROUPS.flatMap((group) => group.fields);

const schemaFor = (field) =>
	PREVENTION_FIELD_SCHEMA[field] || SYNTHETIC_FIELD_SCHEMA[field] || { kind: "number" };

const labelFor = (field) => {
	const schema = schemaFor(field);
	return LABELS[field] || schema.label || field;
};

const emptyForm = () =>
	ALL_FIELDS.reduce((accumulator, field) => ({ ...accumulator, [field]: "" }), {});

/** Fields the model actually consumes: `Diabetes` and onset age are context only. */
const MODEL_FIELDS = ALL_FIELDS.filter(
	(field) => field !== "Diabetes" && field !== "DIQ_DID040",
);

export default function PatientProbe({ onUnauthorised }) {
	const [form, setForm] = useState(emptyForm);
	const [generated, setGenerated] = useState(null);
	const [pending, setPending] = useState(false);
	const [result, setResult] = useState(null);
	const [error, setError] = useState(null);

	const missingRequired = useMemo(
		() => REQUIRED_PROBE_FIELDS.filter((field) => !String(form[field] ?? "").trim()),
		[form],
	);

	const update = (field, value) => {
		setForm((current) => ({ ...current, [field]: value }));
		setResult(null);
	};

	const generate = () => {
		const profile = generateFullSyntheticProfile({});
		setForm({ ...emptyForm(), ...profile.values });
		setGenerated(profile);
		setResult(null);
		setError(null);
	};

	const reset = () => {
		setForm(emptyForm());
		setGenerated(null);
		setResult(null);
		setError(null);
	};

	const score = async () => {
		setPending(true);
		setError(null);
		setResult(null);
		const record = {};
		MODEL_FIELDS.forEach((field) => {
			const value = String(form[field] ?? "").trim();
			if (value !== "") {
				record[field] = Number.isNaN(Number(value)) ? value : Number(value);
			}
		});
		try {
			const response = await apiPostJson("/api/v1/probe/score", {
				patient_record: record,
				confirm_explicit_scoring: true,
			});
			setResult(response);
		} catch (failure) {
			if (failure.status === 401) {
				onUnauthorised();
				return;
			}
			setError(failure.message);
		} finally {
			setPending(false);
		}
	};

	const score_ = result ? result.score : null;

	return (
		<>
			<Notice kind="info" title="Explicit scoring only.">
				Nothing is scored while you type. The model runs once, when you press
				<strong> Score this record</strong>, and no submitted record is stored.
			</Notice>

			<section className="card" aria-labelledby="probe-form-heading">
				<h2 id="probe-form-heading">Record under test</h2>
				<div className="actions" style={{ marginTop: 0, marginBottom: "var(--space-4)" }}>
					<button
						type="button"
						className="btn btn-secondary"
						onClick={generate}
						data-testid="button-generate-synthetic"
					>
						Generate synthetic patient
					</button>
					<button
						type="button"
						className="btn-quiet"
						onClick={reset}
						data-testid="button-reset-probe"
					>
						Clear form
					</button>
				</div>

				{generated ? (
					<Notice kind="ok" title={`Synthetic profile: ${generated.archetype.label}.`}>
						{generated.archetype.description} {SYNTHETIC_NOTE}
					</Notice>
				) : null}

				<form
					onSubmit={(event) => {
						event.preventDefault();
						score();
					}}
					noValidate
				>
					{FIELD_GROUPS.map((group) => (
						<fieldset className="fieldset" key={group.id}>
							<legend>{group.legend}</legend>
							<div className="form-grid">
								{group.fields.map((field) => {
									const schema = schemaFor(field);
									const required = REQUIRED_PROBE_FIELDS.includes(field);
									const inputId = `probe-${field}`;
									return (
										<div key={field}>
											<label htmlFor={inputId}>
												{labelFor(field)}
												{required ? (
													<span aria-hidden="true"> *</span>
												) : null}
												{schema.unit ? (
													<span
														style={{
															color: "var(--ink-400)",
															fontWeight: 400,
														}}
													>
														{" "}
														({schema.unit})
													</span>
												) : null}
											</label>
											{schema.kind === "categorical" ? (
												<select
													id={inputId}
													value={form[field] ?? ""}
													required={required}
													onChange={(event) =>
														update(field, event.target.value)
													}
													data-testid={`input-${field}`}
												>
													<option value="">Not supplied</option>
													{schema.options.map((option) => (
														<option key={option} value={option}>
															{(schema.optionLabels || {})[
																option
															] || option}
														</option>
													))}
												</select>
											) : (
												<input
													id={inputId}
													type="number"
													inputMode="decimal"
													step={schema.step || "any"}
													min={schema.min}
													max={schema.max}
													value={form[field] ?? ""}
													required={required}
													onChange={(event) =>
														update(field, event.target.value)
													}
													data-testid={`input-${field}`}
												/>
											)}
										</div>
									);
								})}
							</div>
						</fieldset>
					))}

					{missingRequired.length ? (
						<p className="field-hint" data-testid="text-missing-required">
							Required before scoring: {missingRequired.map(labelFor).join(", ")}.
						</p>
					) : null}

					<div className="actions">
						<button
							type="submit"
							className="btn"
							disabled={pending || missingRequired.length > 0}
							data-testid="button-score-record"
						>
							{pending ? (
								<>
									<span className="spinner" aria-hidden="true" />
									Scoring
								</>
							) : (
								"Score this record"
							)}
						</button>
						<span className="field-hint" style={{ marginTop: 0 }}>
							Reported diabetes and onset age are context fields; they are not
							model inputs and not outputs.
						</span>
					</div>
				</form>
			</section>

			<section aria-labelledby="probe-result-heading" aria-live="polite">
				<h2 id="probe-result-heading" style={{ marginBottom: "var(--space-3)" }}>
					Research output
				</h2>
				{error ? (
					<Notice kind="blocked" title="Scoring failed.">
						<span data-testid="text-probe-error">{error}</span>
					</Notice>
				) : null}

				{!score_ && !error ? (
					<Empty testId="empty-probe-result">
						No record has been scored yet. Fill the form or generate a synthetic
						profile, then press Score this record.
					</Empty>
				) : null}

				{score_ ? (
					<>
						<div className="grid grid-3">
							<Stat
								label="Metabolic deviation score"
								value={formatNumber(score_.metabolic_deviation_score, 3)}
								detail="Unitless, floored at zero, unbounded above."
								state="abstain"
							/>
							<Stat
								label="Reference percentile"
								value={`${formatNumber(score_.reference_percentile, 1)}`}
								detail="Rank within the NHANES adult reference distribution."
								state="abstain"
							/>
							<Stat
								label="Features supplied"
								value={`${result.features_used.length} of 25`}
								detail={`${result.features_missing.length} allowlisted features left blank.`}
							/>
						</div>

						<div className="card">
							<h3>Feature contributions</h3>
							<p className="field-hint" style={{ marginTop: 0 }}>
								Reconstruction error by feature. This shows what the model could
								not reproduce, not a cause and not a diagnosis.
							</p>
							<div className="table-wrap">
								<table>
									<thead>
										<tr>
											<th scope="col">Feature</th>
											<th scope="col" className="num">
												Reconstruction error
											</th>
											<th scope="col">Share of top contribution</th>
										</tr>
									</thead>
									<tbody>
										{score_.top_deviation_features.map((entry) => {
											const top =
												score_.top_deviation_features[0]
													.reconstruction_error || 1;
											return (
												<tr key={entry.feature}>
													<th scope="row">
														<code>{entry.feature}</code>
													</th>
													<td className="num">
														{formatNumber(
															entry.reconstruction_error,
															4,
														)}
													</td>
													<td style={{ minWidth: 140 }}>
														<span className="meter">
															<span
																style={{
																	width: `${Math.max(4, (entry.reconstruction_error / top) * 100)}%`,
																}}
															/>
														</span>
													</td>
												</tr>
											);
										})}
									</tbody>
								</table>
							</div>
						</div>

						<div className="grid grid-2">
							<div className="card">
								<h3>How to read this</h3>
								<DefinitionList
									items={[
										[
											"Deviation score",
											result.field_meanings.metabolic_deviation_score,
										],
										[
											"Percentile",
											result.field_meanings.reference_percentile,
										],
										[
											"Contributions",
											result.field_meanings.top_deviation_features,
										],
										[
											"Representation",
											result.field_meanings.latent_representation,
										],
									]}
								/>
							</div>
							<div className="card">
								<h3>Evidence boundaries</h3>
								<ul>
									{result.evidence_boundaries.map((item) => (
										<li key={item}>{item}</li>
									))}
								</ul>
							</div>
						</div>

						<Notice kind="caution" title="Not a diagnosis.">
							{result.non_diagnostic_warning}
						</Notice>
					</>
				) : null}
			</section>
		</>
	);
}
