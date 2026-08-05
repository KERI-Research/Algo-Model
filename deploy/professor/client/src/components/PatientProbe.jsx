import { useMemo, useState } from "react";
import { apiPostJson } from "../lib/api.js";
import {
	MODEL_FIELDS,
	PROBE_FIELD_GROUPS,
	emptyProbeForm,
	labelFor,
	schemaFor,
} from "../lib/probe_fields.js";
import {
	generateFullSyntheticProfile,
	REQUIRED_PROBE_FIELDS,
	SYNTHETIC_NOTE,
} from "../lib/synthetic_prevention.js";
import { Empty, Notice } from "./common.jsx";
import ProbeResults from "./ProbeResults.jsx";
export default function PatientProbe({ onUnauthorised }) {
	const [form, setForm] = useState(emptyProbeForm);
	const [generated, setGenerated] = useState(null);
	const [pending, setPending] = useState(false);
	const [result, setResult] = useState(null);
	const [submittedForm, setSubmittedForm] = useState(null);
	const [error, setError] = useState(null);

	const missingRequired = useMemo(
		() =>
			REQUIRED_PROBE_FIELDS.filter(
				(field) => !String(form[field] ?? "").trim(),
			),
		[form],
	);

	const update = (field, value) => {
		setForm((current) => ({ ...current, [field]: value }));
		setResult(null);
		setSubmittedForm(null);
	};

	const generate = () => {
		const profile = generateFullSyntheticProfile({});
		setForm({ ...emptyProbeForm(), ...profile.values });
		setGenerated(profile);
		setResult(null);
		setSubmittedForm(null);
		setError(null);
	};

	const reset = () => {
		setForm(emptyProbeForm());
		setGenerated(null);
		setResult(null);
		setSubmittedForm(null);
		setError(null);
	};

	const score = async () => {
		setPending(true);
		setError(null);
		setResult(null);
		setSubmittedForm(null);
		const formSnapshot = { ...form };
		const record = {};
		MODEL_FIELDS.forEach((field) => {
			const value = String(formSnapshot[field] ?? "").trim();
			if (value !== "") {
				record[field] = Number.isNaN(Number(value))
					? value
					: Number(value);
			}
		});
		try {
			const response = await apiPostJson(
				"/api/v1/probe/score",
				{
					patient_record: record,
					confirm_explicit_scoring: true,
				},
			);
			setResult(response);
			setSubmittedForm(formSnapshot);
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
				Nothing is scored while you type. The model runs
				once, when you press
				<strong> Score this record</strong>, and no
				submitted record is stored.
			</Notice>

			<section
				className="card"
				aria-labelledby="probe-form-heading"
			>
				<h2 id="probe-form-heading">
					Record under test
				</h2>
				<div
					className="actions"
					style={{
						marginTop: 0,
						marginBottom: "var(--space-4)",
					}}
				>
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
					<Notice
						kind="ok"
						title={`Synthetic profile: ${generated.archetype.label}.`}
					>
						{
							generated.archetype
								.description
						}{" "}
						{SYNTHETIC_NOTE}
					</Notice>
				) : null}

				<form
					onSubmit={(event) => {
						event.preventDefault();
						score();
					}}
					noValidate
				>
					{PROBE_FIELD_GROUPS.map((group) => (
						<fieldset
							className="fieldset"
							key={group.id}
						>
							<legend>
								{group.legend}
							</legend>
							<div className="form-grid">
								{group.fields.map(
									(
										field,
									) => {
										const schema =
											schemaFor(
												field,
											);
										const required =
											REQUIRED_PROBE_FIELDS.includes(
												field,
											);
										const inputId = `probe-${field}`;
										return (
											<div
												key={
													field
												}
											>
												<label
													htmlFor={
														inputId
													}
												>
													{labelFor(
														field,
													)}
													{required ? (
														<span aria-hidden="true">
															{" "}
															*
														</span>
													) : null}
													{schema.unit ? (
														<span
															style={{
																color: "var(--ink-400)",
																fontWeight: 400,
															}}
														>
															{" "}
															(
															{
																schema.unit
															}
															)
														</span>
													) : null}
												</label>
												{schema.kind ===
												"categorical" ? (
													<select
														id={
															inputId
														}
														value={
															form[
																field
															] ??
															""
														}
														required={
															required
														}
														onChange={(
															event,
														) =>
															update(
																field,
																event
																	.target
																	.value,
															)
														}
														data-testid={`input-${field}`}
													>
														<option value="">
															Not
															supplied
														</option>
														{schema.options.map(
															(
																option,
															) => (
																<option
																	key={
																		option
																	}
																	value={
																		option
																	}
																>
																	{(schema.optionLabels ||
																		{})[
																		option
																	] ||
																		option}
																</option>
															),
														)}
													</select>
												) : (
													<input
														id={
															inputId
														}
														type="number"
														inputMode="decimal"
														step={
															schema.step ||
															"any"
														}
														min={
															schema.min
														}
														max={
															schema.max
														}
														value={
															form[
																field
															] ??
															""
														}
														required={
															required
														}
														onChange={(
															event,
														) =>
															update(
																field,
																event
																	.target
																	.value,
															)
														}
														data-testid={`input-${field}`}
													/>
												)}
											</div>
										);
									},
								)}
							</div>
						</fieldset>
					))}

					{missingRequired.length ? (
						<p
							className="field-hint"
							data-testid="text-missing-required"
						>
							Required before scoring:{" "}
							{missingRequired
								.map(labelFor)
								.join(", ")}
							.
						</p>
					) : null}

					<div className="actions">
						<button
							type="submit"
							className="btn"
							disabled={
								pending ||
								missingRequired.length >
									0
							}
							data-testid="button-score-record"
						>
							{pending ? (
								<>
									<span
										className="spinner"
										aria-hidden="true"
									/>
									Scoring
								</>
							) : (
								"Score this record"
							)}
						</button>
						<span
							className="field-hint"
							style={{ marginTop: 0 }}
						>
							Reported diabetes and
							onset age are context
							fields; they are not
							model inputs and not
							outputs.
						</span>
					</div>
				</form>
			</section>

			<section
				aria-labelledby="probe-result-heading"
				aria-live="polite"
			>
				<h2
					id="probe-result-heading"
					style={{
						marginBottom: "var(--space-3)",
					}}
				>
					Research output
				</h2>
				{error ? (
					<Notice
						kind="blocked"
						title="Scoring failed."
					>
						<span data-testid="text-probe-error">
							{error}
						</span>
					</Notice>
				) : null}

				{!score_ && !error ? (
					<Empty testId="empty-probe-result">
						No record has been scored yet.
						Fill the form or generate a
						synthetic profile, then press
						Score this record.
					</Empty>
				) : null}

				{score_ ? (
					<ProbeResults
						result={result}
						submittedForm={submittedForm}
					/>
				) : null}
			</section>
		</>
	);
}
