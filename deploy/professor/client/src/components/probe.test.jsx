import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DatasetAnalysis from "./DatasetAnalysis.jsx";
import PatientProbe from "./PatientProbe.jsx";

const json = (body, status = 200) => ({
	ok: status >= 200 && status < 300,
	status,
	headers: { get: () => "application/json" },
	json: async () => body,
	text: async () => JSON.stringify(body),
});

const SCORE_RESPONSE = {
	score: {
		metabolic_deviation_score: 1.234567,
		reference_percentile: 87.31,
		latent_representation: Array.from({ length: 16 }, () => 0.1),
		top_deviation_features: [
			{ feature: "homa_ir", reconstruction_error: 0.51 },
			{ feature: "GLU_LBXGLU", reconstruction_error: 0.22 },
		],
	},
	features_used: ["DEMO_RIDAGEYR", "BMX_BMXBMI"],
	features_missing: ["CPEP_LBXCPSI"],
	field_meanings: {
		metabolic_deviation_score: "How unusual this profile is.",
		reference_percentile: "Rank in the reference distribution.",
		top_deviation_features: "Largest reconstruction error.",
		latent_representation: "Learned encoding.",
	},
	evidence_boundaries: ["Cross-sectional training data only."],
	dataset_capability_state: "Cross-sectional only",
	patient_assessment: {
		current_profile_assessment: {
			section_title: "Current profile assessment",
			deviation_band: "within_reference_range",
			deviation_band_label: "Within reference range",
			reference_percentile: 87.31,
			warning_label:
				"Within reference range (not diagnostic)",
			note: "This profile differs from the reference and may warrant clinician review.",
		},
		standout_factors: {
			section_title: "Standout factors in this profile",
			top_deviation_features: [
				{
					feature: "homa_ir",
					reconstruction_error: 0.51,
				},
				{
					feature: "GLU_LBXGLU",
					reconstruction_error: 0.22,
				},
			],
		},
		data_readiness: {
			section_title: "Data readiness and missing information",
			dataset_capability_state: "Cross-sectional only",
			missing_fields: [
				{
					field: "CPEP_LBXCPSI",
					priority: "medium",
					why_it_matters:
						"Additional features improve interpretation depth.",
					expected_impact_bucket:
						"moderate_confidence_gain",
				},
			],
		},
		research_association: {
			section_title:
				"Research-only cancer/diabetes association",
			status: "disabled_on_this_route",
			note: "No validated future cancer or diabetes risk model is currently deployed.",
		},
	},
	non_diagnostic_warning: "Research use only, non-diagnostic.",
};

describe("PatientProbe", () => {
	beforeEach(() => {
		global.fetch = vi.fn(() =>
			Promise.resolve(json(SCORE_RESPONSE)),
		);
	});
	afterEach(() => vi.restoreAllMocks());

	it("starts empty and does not score anything automatically", async () => {
		render(<PatientProbe onUnauthorised={() => {}} />);
		expect(
			screen.getByTestId("empty-probe-result"),
		).toBeInTheDocument();
		expect(
			screen.getByTestId("button-score-record"),
		).toBeDisabled();
		expect(global.fetch).not.toHaveBeenCalled();
	});

	it("fills every required field from the synthetic generator", async () => {
		render(<PatientProbe onUnauthorised={() => {}} />);
		const user = userEvent.setup();
		await user.click(
			screen.getByTestId("button-generate-synthetic"),
		);
		expect(
			screen.getByTestId("input-DEMO_RIDAGEYR"),
		).not.toHaveValue(null);
		expect(screen.getByTestId("input-BMX_BMXBMI")).not.toHaveValue(
			null,
		);
		expect(
			screen.getByTestId("input-DEMO_RIAGENDR"),
		).not.toHaveValue("");
		expect(screen.getByTestId("button-score-record")).toBeEnabled();
		expect(global.fetch).not.toHaveBeenCalled();
	});

	it("scores explicitly and shows deviation, percentile and contributions", async () => {
		render(<PatientProbe onUnauthorised={() => {}} />);
		const user = userEvent.setup();
		await user.click(
			screen.getByTestId("button-generate-synthetic"),
		);
		await user.click(screen.getByTestId("button-score-record"));
		await waitFor(() =>
			expect(global.fetch).toHaveBeenCalledTimes(1),
		);
		const [, options] = global.fetch.mock.calls[0];
		const payload = JSON.parse(options.body);
		expect(payload.confirm_explicit_scoring).toBe(true);
		expect(payload.patient_record.Diabetes).toBeUndefined();
		expect(await screen.findByText("1.235")).toBeInTheDocument();
		expect(screen.getByText("87.3")).toBeInTheDocument();
		expect(screen.getByText("homa_ir")).toBeInTheDocument();
		expect(
			screen.getByText(/Research use only/),
		).toBeInTheDocument();
		expect(
			screen.getByTestId("panel-current-profile-assessment"),
		).toBeInTheDocument();
		expect(
			screen.getByTestId("panel-standout-factors"),
		).toBeInTheDocument();
		expect(
			screen.getByTestId("panel-data-readiness"),
		).toBeInTheDocument();
		expect(
			screen.getByTestId("panel-research-association"),
		).toBeInTheDocument();
		expect(
			screen.getAllByText(
				/No validated future cancer or diabetes risk model/i,
			).length,
		).toBeGreaterThan(0);
	});

	it("clears the form on reset", async () => {
		render(<PatientProbe onUnauthorised={() => {}} />);
		const user = userEvent.setup();
		await user.click(
			screen.getByTestId("button-generate-synthetic"),
		);
		await user.click(screen.getByTestId("button-reset-probe"));
		expect(screen.getByTestId("input-DEMO_RIDAGEYR")).toHaveValue(
			null,
		);
		expect(
			screen.getByTestId("button-score-record"),
		).toBeDisabled();
	});

	it("surfaces a scoring error without losing the form", async () => {
		global.fetch = vi.fn(() =>
			Promise.resolve(
				json(
					{
						error: "No allowlisted feature values.",
					},
					422,
				),
			),
		);
		render(<PatientProbe onUnauthorised={() => {}} />);
		const user = userEvent.setup();
		await user.click(
			screen.getByTestId("button-generate-synthetic"),
		);
		await user.click(screen.getByTestId("button-score-record"));
		expect(
			await screen.findByTestId("text-probe-error"),
		).toBeInTheDocument();
		expect(
			screen.getByTestId("input-DEMO_RIDAGEYR"),
		).not.toHaveValue(null);
	});
});

describe("DatasetAnalysis", () => {
	beforeEach(() => {
		global.fetch = vi.fn(() => Promise.resolve(json({})));
	});
	afterEach(() => vi.restoreAllMocks());

	const csvFile = (
		name = "cohort.csv",
		content = "DEMO_RIDAGEYR\n55\n",
	) => new File([content], name, { type: "text/csv" });

	it("blocks screening until a file and the de-identification confirmation are present", async () => {
		render(<DatasetAnalysis onUnauthorised={() => {}} />);
		const user = userEvent.setup();
		expect(
			screen.getByTestId("button-screen-dataset"),
		).toBeDisabled();
		await user.upload(
			screen.getByTestId("input-dataset-file"),
			csvFile(),
		);
		expect(
			screen.getByTestId("button-screen-dataset"),
		).toBeDisabled();
		await user.click(
			screen.getByTestId("input-deidentified-confirm"),
		);
		expect(
			screen.getByTestId("button-screen-dataset"),
		).toBeEnabled();
		expect(global.fetch).not.toHaveBeenCalled();
	});

	it("rejects a non-CSV drop in the browser before any upload", async () => {
		render(<DatasetAnalysis onUnauthorised={() => {}} />);
		const dropzone = document.querySelector(".dropzone");
		fireEvent.drop(dropzone, {
			dataTransfer: {
				files: [
					new File(["a"], "notes.txt", {
						type: "text/plain",
					}),
				],
			},
		});
		expect(
			await screen.findByTestId("text-dataset-error"),
		).toHaveTextContent("Only .csv files are accepted.");
		expect(global.fetch).not.toHaveBeenCalled();
	});

	it("rejects a file above the 15 MB limit before any upload", async () => {
		render(<DatasetAnalysis onUnauthorised={() => {}} />);
		const big = new File(["x"], "big.csv", { type: "text/csv" });
		Object.defineProperty(big, "size", { value: 16 * 1024 * 1024 });
		const user = userEvent.setup();
		await user.upload(
			screen.getByTestId("input-dataset-file"),
			big,
		);
		expect(
			await screen.findByTestId("text-dataset-error"),
		).toHaveTextContent("The limit is 15 MB.");
		expect(global.fetch).not.toHaveBeenCalled();
	});

	it("shows identifier rejections column by column", async () => {
		global.fetch = vi.fn(() =>
			Promise.resolve(
				json(
					{
						error: {
							message: "The file contains direct identifiers.",
							identifier_columns: [
								{
									column: "full_name",
									identifier_type:
										"personal name",
								},
							],
						},
					},
					422,
				),
			),
		);
		render(<DatasetAnalysis onUnauthorised={() => {}} />);
		const user = userEvent.setup();
		await user.upload(
			screen.getByTestId("input-dataset-file"),
			csvFile(),
		);
		await user.click(
			screen.getByTestId("input-deidentified-confirm"),
		);
		await user.click(screen.getByTestId("button-screen-dataset"));
		expect(
			await screen.findByTestId("text-dataset-error"),
		).toHaveTextContent("direct identifiers");
		expect(screen.getByText("full_name")).toBeInTheDocument();
	});
});
