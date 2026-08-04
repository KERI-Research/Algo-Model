import { useRef, useState } from "react";
import { apiPostForBlob, apiPostForm } from "../lib/api.js";
import {
	DefinitionList,
	Empty,
	Notice,
	Stat,
	TierBadge,
	formatNumber,
	formatPercent,
} from "./common.jsx";

const MAX_BYTES = 15 * 1024 * 1024;
const MAX_ROWS = 20000;

const readableSize = (bytes) =>
	bytes >= 1024 * 1024
		? `${(bytes / (1024 * 1024)).toFixed(2)} MB`
		: `${Math.max(1, Math.round(bytes / 1024))} KB`;

export default function DatasetAnalysis({ onUnauthorised }) {
	const [file, setFile] = useState(null);
	const [confirmed, setConfirmed] = useState(false);
	const [dragging, setDragging] = useState(false);
	const [stage, setStage] = useState("idle"); // idle | inspecting | inspected | analysing | analysed
	const [intake, setIntake] = useState(null);
	const [analysis, setAnalysis] = useState(null);
	const [error, setError] = useState(null);
	const [downloadNote, setDownloadNote] = useState(null);
	const inputRef = useRef(null);

	const resetResults = () => {
		setIntake(null);
		setAnalysis(null);
		setError(null);
		setDownloadNote(null);
		setStage("idle");
	};

	const chooseFile = (candidate) => {
		resetResults();
		if (!candidate) {
			setFile(null);
			return;
		}
		if (!candidate.name.toLowerCase().endsWith(".csv")) {
			setFile(null);
			setError({ message: "Only .csv files are accepted." });
			return;
		}
		if (candidate.size > MAX_BYTES) {
			setFile(null);
			setError({
				message: `That file is ${readableSize(candidate.size)}. The limit is 15 MB.`,
			});
			return;
		}
		setFile(candidate);
	};

	const handleDrop = (event) => {
		event.preventDefault();
		setDragging(false);
		chooseFile(event.dataTransfer.files && event.dataTransfer.files[0]);
	};

	const buildForm = (extra = {}) => {
		const body = new FormData();
		body.append("file", file);
		body.append("deidentified_confirmed", "true");
		Object.entries(extra).forEach(([key, value]) => body.append(key, value));
		return body;
	};

	const failed = (failure) => {
		if (failure.status === 401) {
			onUnauthorised();
			return;
		}
		const detail = failure.detail;
		setError({
			message: failure.message,
			identifierColumns:
				detail && typeof detail === "object" ? detail.identifier_columns : null,
			blockers: detail && typeof detail === "object" ? detail.blockers : null,
		});
	};

	const inspect = async () => {
		if (!file || !confirmed) {
			return;
		}
		setStage("inspecting");
		setError(null);
		setAnalysis(null);
		try {
			const report = await apiPostForm("/api/v1/dataset/inspect", buildForm());
			setIntake(report);
			setStage("inspected");
		} catch (failure) {
			failed(failure);
			setStage("idle");
		}
	};

	const analyse = async () => {
		if (!file || !confirmed || !intake) {
			return;
		}
		setStage("analysing");
		setError(null);
		try {
			const report = await apiPostForm(
				"/api/v1/dataset/analyse",
				buildForm({ analysis_confirmed: "true" }),
			);
			setAnalysis(report);
			setStage("analysed");
		} catch (failure) {
			failed(failure);
			setStage("inspected");
		}
	};

	const download = async () => {
		if (!analysis) {
			return;
		}
		try {
			const blob = await apiPostForBlob("/api/v1/dataset/export", {
				rows: analysis.rows,
			});
			const href = URL.createObjectURL(blob);
			const anchor = document.createElement("a");
			anchor.href = href;
			anchor.download = "metaboguard_research_results.csv";
			document.body.appendChild(anchor);
			anchor.click();
			anchor.remove();
			URL.revokeObjectURL(href);
			setDownloadNote("Results CSV downloaded. Nothing was retained on the server.");
		} catch (failure) {
			failed(failure);
		}
	};

	const busy = stage === "inspecting" || stage === "analysing";

	return (
		<>
			<Notice kind="caution" title="De-identified research data only.">
				Files are parsed in memory, screened for direct identifiers, analysed and
				discarded. Nothing is written to disk, nothing is sent to a third party and
				no upload history is kept. Do not upload identifiable or clinical patient
				data to this prototype.
			</Notice>

			<section className="card" aria-labelledby="upload-heading">
				<h2 id="upload-heading">1. Upload a CSV</h2>
				<div
					className="dropzone"
					data-active={dragging}
					onDragOver={(event) => {
						event.preventDefault();
						setDragging(true);
					}}
					onDragLeave={() => setDragging(false)}
					onDrop={handleDrop}
				>
					<p>
						Drag a CSV here, or choose a file. Maximum 15 MB and{" "}
						{MAX_ROWS.toLocaleString("en-GB")} rows.
					</p>
					<input
						ref={inputRef}
						id="dataset-file"
						type="file"
						accept=".csv,text/csv"
						className="visually-hidden"
						onChange={(event) => chooseFile(event.target.files[0])}
						data-testid="input-dataset-file"
					/>
					<button
						type="button"
						className="btn btn-secondary"
						onClick={() => inputRef.current && inputRef.current.click()}
						data-testid="button-choose-file"
					>
						Choose CSV file
					</button>
					{file ? (
						<span className="file-chip" data-testid="text-selected-file">
							{file.name} - {readableSize(file.size)}
						</span>
					) : (
						<span className="field-hint" style={{ marginTop: 0 }}>
							No file selected.
						</span>
					)}
				</div>

				<div className="checkbox" style={{ marginTop: "var(--space-4)" }}>
					<input
						id="deid-confirm"
						type="checkbox"
						checked={confirmed}
						onChange={(event) => {
							setConfirmed(event.target.checked);
							resetResults();
						}}
						data-testid="input-deidentified-confirm"
					/>
					<label htmlFor="deid-confirm">
						I confirm this file contains de-identified research data. It holds no
						names, contact details, NHS or insurance numbers, medical record
						numbers, exact dates of birth or postcodes. Age and anonymous row
						identifiers are acceptable.
					</label>
				</div>

				<div className="actions">
					<button
						type="button"
						className="btn"
						onClick={inspect}
						disabled={!file || !confirmed || busy}
						data-testid="button-screen-dataset"
					>
						{stage === "inspecting" ? (
							<>
								<span className="spinner" aria-hidden="true" />
								Screening
							</>
						) : (
							"Screen file"
						)}
					</button>
					{!confirmed ? (
						<span className="field-hint" style={{ marginTop: 0 }}>
							Confirm de-identification to continue.
						</span>
					) : null}
				</div>

				{error ? (
					<div style={{ marginTop: "var(--space-4)" }}>
						<Notice kind="blocked" title="File rejected.">
							<span data-testid="text-dataset-error">{error.message}</span>
							{error.identifierColumns && error.identifierColumns.length ? (
								<ul style={{ marginTop: "var(--space-2)", marginBottom: 0 }}>
									{error.identifierColumns.map((item) => (
										<li key={item.column}>
											<code>{item.column}</code> - detected as a
											direct identifier ({item.identifier_type})
										</li>
									))}
								</ul>
							) : null}
							{error.blockers && error.blockers.length ? (
								<ul style={{ marginTop: "var(--space-2)", marginBottom: 0 }}>
									{error.blockers.map((item) => (
										<li key={item}>{item}</li>
									))}
								</ul>
							) : null}
						</Notice>
					</div>
				) : null}
			</section>

			{stage === "inspecting" ? (
				<section className="card" aria-busy="true">
					<h4>Screening file</h4>
					<div className="skeleton" />
					<div className="skeleton" />
					<div className="skeleton" />
				</section>
			) : null}

			{intake ? <IntakeReport intake={intake} /> : null}

			{intake ? (
				<section className="card" aria-labelledby="confirm-heading">
					<h2 id="confirm-heading">3. Confirm and run the model</h2>
					{intake.model_ready ? (
						<p>
							{intake.rows.will_be_scored.toLocaleString("en-GB")} row
							{intake.rows.will_be_scored === 1 ? "" : "s"} will be scored with
							the deployed NumPy inference artifact. The model is never trained
							or updated on uploaded data.
						</p>
					) : (
						<Notice kind="blocked" title="Not analysable.">
							{intake.blockers.join(" ")}
						</Notice>
					)}
					<div className="actions">
						<button
							type="button"
							className="btn"
							onClick={analyse}
							disabled={!intake.model_ready || busy}
							data-testid="button-run-analysis"
						>
							{stage === "analysing" ? (
								<>
									<span className="spinner" aria-hidden="true" />
									Scoring rows
								</>
							) : (
								"Run model analysis"
							)}
						</button>
					</div>
				</section>
			) : null}

			{stage === "analysing" ? (
				<section className="card" aria-busy="true">
					<h4>Scoring accepted rows</h4>
					<div className="skeleton" />
					<div className="skeleton" />
				</section>
			) : null}

			{analysis ? (
				<AnalysisResult
					analysis={analysis}
					onDownload={download}
					downloadNote={downloadNote}
				/>
			) : null}
		</>
	);
}

function IntakeReport({ intake }) {
	const eligibility = Object.entries(intake.feature_eligibility);
	const tierCounts = eligibility.reduce((accumulator, [, value]) => {
		accumulator[value.tier] = (accumulator[value.tier] || 0) + 1;
		return accumulator;
	}, {});

	return (
		<section className="card" aria-labelledby="intake-heading" data-testid="dataset-intake">
			<h2 id="intake-heading">2. Screening report</h2>

			<div className="grid grid-4">
				<Stat
					label="Rows accepted"
					value={intake.rows.accepted.toLocaleString("en-GB")}
					detail={`${intake.rows.rejected.toLocaleString("en-GB")} rejected of ${intake.rows.total.toLocaleString("en-GB")}.`}
					state={intake.rows.accepted > 0 ? "ok" : "blocked"}
				/>
				<Stat
					label="Allowlisted features mapped"
					value={`${intake.schema.mapped_features.length} of ${intake.schema.allowlist.length}`}
					detail={`${tierCounts.usable_now || 0} usable now, ${tierCounts.qualified_use || 0} qualified.`}
				/>
				<Stat
					label="Prohibited columns"
					value={intake.schema.prohibited_columns.length}
					detail="Outcome, label-derived or post-diagnosis TCGA fields, excluded from the model."
					state={intake.schema.prohibited_columns.length ? "warn" : "ok"}
				/>
				<Stat
					label="Range violations"
					value={Object.keys(intake.range_violations).length}
					detail="Values outside the project's plausibility windows."
					state={Object.keys(intake.range_violations).length ? "warn" : "ok"}
				/>
			</div>

			<h3>Preview</h3>
			<p className="field-hint" style={{ marginTop: 0 }}>
				First rows, restricted to allowlisted feature columns.
			</p>
			{intake.preview && intake.preview.columns.length ? (
				<div className="table-wrap">
					<table>
						<thead>
							<tr>
								{intake.preview.columns.map((column) => (
									<th scope="col" key={column}>
										{column}
									</th>
								))}
							</tr>
						</thead>
						<tbody>
							{intake.preview.rows.map((row, index) => (
								<tr key={index}>
									{row.map((value, cell) => (
										<td className="num" key={cell}>
											{value === null ? "-" : value}
										</td>
									))}
								</tr>
							))}
						</tbody>
					</table>
				</div>
			) : (
				<Empty>No allowlisted feature column was found to preview.</Empty>
			)}

			<h3 style={{ marginTop: "var(--space-5)" }}>Schema mapping and missingness</h3>
			<div className="table-wrap">
				<table>
					<caption>
						Tier definitions: {Object.entries(intake.tier_definitions)
							.map(([tier, definition]) => `${tier.replace(/_/g, " ")} - ${definition}`)
							.join(" ")}
					</caption>
					<thead>
						<tr>
							<th scope="col">Column</th>
							<th scope="col">Tier</th>
							<th scope="col" className="num">
								Coverage
							</th>
							<th scope="col" className="num">
								Missing
							</th>
							<th scope="col">Notes</th>
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
									{value.coverage === null
										? "-"
										: formatPercent(value.coverage)}
								</td>
								<td className="num">
									{value.coverage === null
										? "-"
										: formatPercent(1 - value.coverage)}
								</td>
								<td className="wrap">{value.reasons.join(" ")}</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>

			{Object.keys(intake.range_violations).length ? (
				<>
					<h3 style={{ marginTop: "var(--space-5)" }}>Unit and range violations</h3>
					<div className="table-wrap">
						<table>
							<thead>
								<tr>
									<th scope="col">Feature</th>
									<th scope="col">Plausible range</th>
									<th scope="col" className="num">
										Values outside
									</th>
									<th scope="col" className="num">
										Share
									</th>
									<th scope="col">Exceeds tolerance</th>
								</tr>
							</thead>
							<tbody>
								{Object.entries(intake.range_violations).map(
									([feature, detail]) => (
										<tr key={feature}>
											<th scope="row">
												<code>{feature}</code>
											</th>
											<td className="num">
												{detail.plausible_range.join(" to ")}
											</td>
											<td className="num">
												{detail.values_outside_range}
											</td>
											<td className="num">
												{formatPercent(
													detail.fraction_of_present_values,
													2,
												)}
											</td>
											<td>
												{detail.exceeds_tolerance ? (
													<span className="badge" data-tier="qualified_use">
														yes
													</span>
												) : (
													<span className="badge">no</span>
												)}
											</td>
										</tr>
									),
								)}
							</tbody>
						</table>
					</div>
				</>
			) : null}

			<h3 style={{ marginTop: "var(--space-5)" }}>Dataset capability</h3>
			<DefinitionList
				items={[
					["Rows", (intake.dataset_capability.rows || 0).toLocaleString("en-GB")],
					[
						"Repeated measurements per subject",
						intake.dataset_capability.has_repeated_patient_measurements
							? "yes"
							: "no",
					],
					[
						"Future-development prediction",
						intake.dataset_capability.supports_future_development_prediction
							? "reported as supported - verify gates"
							: "not supported",
					],
					["Supported output", intake.dataset_capability.supported_output],
					["Clustering", intake.dataset_capability.clustering_note],
					["Row rule", intake.rows.rejection_rule],
					["Retention", intake.persistence],
				]}
			/>

			{intake.schema.prohibited_columns.length ? (
				<Notice kind="caution" title="Excluded columns.">
					<ul style={{ marginBottom: 0, marginTop: "var(--space-2)" }}>
						{intake.schema.prohibited_columns.map((item) => (
							<li key={item.column}>
								<code>{item.column}</code> - {item.reason}
							</li>
						))}
					</ul>
				</Notice>
			) : null}
		</section>
	);
}

function AnalysisResult({ analysis, onDownload, downloadNote }) {
	const aggregate = analysis.aggregate;
	const bands = aggregate.percentile_bands;
	const maxBand = Math.max(...Object.values(bands), 1);

	return (
		<section className="card" aria-labelledby="analysis-heading" data-testid="dataset-analysis">
			<h2 id="analysis-heading">4. Research outputs</h2>

			<div className="grid grid-4">
				<Stat
					label="Rows scored"
					value={aggregate.rows_scored.toLocaleString("en-GB")}
					detail={
						aggregate.row_cap_applied
							? `Capped at ${aggregate.row_cap.toLocaleString("en-GB")} rows for this deployment.`
							: "All accepted rows were scored."
					}
					state="ok"
				/>
				<Stat
					label="Median deviation score"
					value={formatNumber(aggregate.deviation_score.median, 3)}
					detail={`Range ${formatNumber(aggregate.deviation_score.min, 2)} to ${formatNumber(aggregate.deviation_score.max, 2)}.`}
				/>
				<Stat
					label="Median reference percentile"
					value={formatNumber(aggregate.reference_percentile.median, 1)}
					detail="Versus the NHANES adult reference distribution."
				/>
				<Stat
					label="At or above p90"
					value={formatPercent(aggregate.reference_percentile.share_at_or_above_p90)}
					detail="Share of scored rows that are unusual for the reference sample."
					state="abstain"
				/>
			</div>

			<h3>Percentile distribution</h3>
			<div className="table-wrap">
				<table>
					<caption>
						Counts of scored rows by reference-percentile band. Bands are
						positions in a distribution, not risk categories.
					</caption>
					<thead>
						<tr>
							<th scope="col">Band</th>
							<th scope="col" className="num">
								Rows
							</th>
							<th scope="col">Share</th>
						</tr>
					</thead>
					<tbody>
						{Object.entries(bands).map(([band, count]) => (
							<tr key={band}>
								<th scope="row">{band.replace(/_/g, " ")}</th>
								<td className="num">{count.toLocaleString("en-GB")}</td>
								<td style={{ minWidth: 160 }}>
									<span className="meter">
										<span style={{ width: `${(count / maxBand) * 100}%` }} />
									</span>
								</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>

			<h3 style={{ marginTop: "var(--space-5)" }}>Most frequent contributing features</h3>
			<div className="table-wrap">
				<table>
					<thead>
						<tr>
							<th scope="col">Feature</th>
							<th scope="col" className="num">
								Rows in top three
							</th>
							<th scope="col" className="num">
								Share of rows
							</th>
						</tr>
					</thead>
					<tbody>
						{aggregate.top_deviation_features.map((item) => (
							<tr key={item.feature}>
								<th scope="row">
									<code>{item.feature}</code>
								</th>
								<td className="num">
									{item.rows_in_top_three.toLocaleString("en-GB")}
								</td>
								<td className="num">{formatPercent(item.share_of_rows)}</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>

			<h3 style={{ marginTop: "var(--space-5)" }}>Per-row outputs</h3>
			<p className="field-hint" style={{ marginTop: 0 }}>
				First 50 of {aggregate.rows_scored.toLocaleString("en-GB")} scored rows. The
				full set is in the CSV export.
			</p>
			<div className="table-wrap" style={{ maxHeight: 420, overflowY: "auto" }}>
				<table>
					<thead>
						<tr>
							<th scope="col" className="num">
								Row
							</th>
							<th scope="col" className="num">
								Deviation score
							</th>
							<th scope="col" className="num">
								Percentile
							</th>
							<th scope="col">Top contributing features</th>
						</tr>
					</thead>
					<tbody>
						{analysis.rows.slice(0, 50).map((row) => (
							<tr key={row.row_number}>
								<td className="num">{row.row_number}</td>
								<td className="num">
									{formatNumber(row.metabolic_deviation_score, 3)}
								</td>
								<td className="num">
									{formatNumber(row.reference_percentile, 1)}
								</td>
								<td className="wrap">
									{row.top_deviation_features
										.map((entry) => entry.feature)
										.join(", ")}
								</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>

			<div className="actions">
				<button
					type="button"
					className="btn"
					onClick={onDownload}
					data-testid="button-download-results"
				>
					Download results CSV
				</button>
				<span className="field-hint" style={{ marginTop: 0 }}>
					Generated in memory on request. No server-side retention or history.
				</span>
			</div>

			{downloadNote ? (
				<Notice kind="ok" title="Export ready.">
					<span data-testid="text-download-note">{downloadNote}</span>
				</Notice>
			) : null}

			<Notice kind="info" title="Clustering unavailable in this deployment.">
				{analysis.clustering.reason}
			</Notice>

			<Notice kind="caution" title="Interpretation limits.">
				{analysis.interpretation} {analysis.non_diagnostic_warning}
			</Notice>
		</section>
	);
}
