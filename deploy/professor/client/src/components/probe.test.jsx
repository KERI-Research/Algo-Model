import {
	fireEvent,
	render,
	screen,
	waitFor,
	within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DatasetAnalysis from "./DatasetAnalysis.jsx";
import PatientProbe from "./PatientProbe.jsx";
import ProbeResults from "./ProbeResults.jsx";

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
	features_used: ["DEMO_RIDAGEYR", "BMX_BMXBMI", "GLU_LBXGLU", "homa_ir"],
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
			note: "No model warning; not evidence disease is absent.",
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
			note: "Supplied-measurement reconstruction diagnostics only; not causality, disease attribution or diagnosis.",
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
			section_title: "Cancer and diabetes research questions",
			status: "future_and_causal_probabilities_unavailable",
			note: "No validated future cancer or diabetes risk model is currently deployed.",
			cancer_scope: [
				{
					id: "pancreatic_cancer",
					label: "Pancreatic cancer",
					status: "research_scope_only",
				},
				{
					id: "general_cancers",
					label: "General cancers",
					status: "research_scope_only",
				},
			],
			scope_note: "Pancreatic cancer and general cancers have equal research emphasis here. This model classifies neither scope.",
			factor_note:
				"Supplied standout measurements are grouped by relevance to each research question. A measurement may appear in more than one group; grouping does not establish direction or causality.",
			pathways: [
				{
					id: "diabetes_related_cancer",
					title: "Diabetes measurements and cancer",
					question: "Can this model determine temporal direction between diabetes-related measurements and cancer?",
					status: "not_estimable",
					probability: null,
					reason: "Cross-sectional records cannot establish that diabetes-related changes occurred before cancer or estimate future cancer incidence.",
					observed_standout_features: [
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
				{
					id: "lifestyle_related_cancer",
					title: "Anthropometry, reported exposures and cancer",
					question: "Can this model separate anthropometry, weight change and reported exposure measurements from diabetes-related pathways?",
					status: "not_estimable",
					probability: null,
					reason: "Cross-sectional records cannot establish that lifestyle factors occurred before cancer, separate them from diabetes pathways, or estimate future cancer incidence.",
					observed_standout_features: [],
				},
				{
					id: "cancer_related_diabetes",
					title: "Cancer and diabetes direction",
					question: "Can this model determine temporal direction between cancer and diabetes-related changes?",
					status: "not_estimable",
					probability: null,
					reason: "Cross-sectional records cannot determine whether cancer preceded diabetes or estimate future diabetes onset.",
					observed_standout_features: [
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
				{
					id: "lifestyle_related_diabetes",
					title: "Anthropometry, reported exposures and diabetes",
					question: "Can this model determine temporal direction between these measurements and diabetes?",
					status: "not_estimable",
					probability: null,
					reason: "Cross-sectional records cannot establish that lifestyle factors preceded diabetes or estimate future diabetes onset.",
					observed_standout_features: [],
				},
			],
		},
	},
	non_diagnostic_warning: "Research use only, non-diagnostic.",
};

describe("ProbeResults contract", () => {
	afterEach(() => vi.restoreAllMocks());

	it("fails closed when a pathway probability is not null", async () => {
		const invalidResponse = JSON.parse(
			JSON.stringify(SCORE_RESPONSE),
		);
		invalidResponse.patient_assessment.research_association.pathways[0].probability = 0.42;
		global.fetch = vi.fn(() =>
			Promise.resolve(json(invalidResponse)),
		);
		render(<PatientProbe onUnauthorised={() => {}} />);
		const user = userEvent.setup();
		await user.click(
			screen.getByTestId("button-generate-synthetic"),
		);
		await user.click(screen.getByTestId("button-score-record"));
		expect(
			await screen.findByText(
				"Pathway contract unavailable.",
			),
		).toBeInTheDocument();
		expect(
			screen.queryByTestId("pathway-diabetes_related_cancer"),
		).not.toBeInTheDocument();
		expect(screen.queryByText("42%")).not.toBeInTheDocument();
	});

	it("handles a missing percentile and no supplied standout factors", () => {
		const sparseResponse = JSON.parse(
			JSON.stringify(SCORE_RESPONSE),
		);
		sparseResponse.score.reference_percentile = null;
		sparseResponse.patient_assessment.current_profile_assessment.reference_percentile =
			null;
		sparseResponse.patient_assessment.standout_factors.top_deviation_features =
			[];
		sparseResponse.patient_assessment.research_association.pathways.forEach(
			(pathway) => {
				pathway.observed_standout_features = [];
			},
		);
		render(
			<ProbeResults
				result={sparseResponse}
				submittedForm={{}}
			/>,
		);
		expect(
			screen.getByText(
				"Reference percentile unavailable for this result.",
			),
		).toBeInTheDocument();
		expect(screen.getByText("Unavailable")).toBeInTheDocument();
		expect(
			screen.getByText(/No supplied measurement appears/i),
		).toBeInTheDocument();
		expect(
			screen.getAllByText(/No matching measurement appears/i),
		).toHaveLength(4);
	});

	it("blocks a response missing required assessment sections", () => {
		render(
			<ProbeResults
				result={{ score: {} }}
				submittedForm={{}}
			/>,
		);
		expect(
			screen.getByTestId("probe-result-contract-error"),
		).toHaveTextContent("Result contract unavailable");
	});
});

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
		await user.selectOptions(
			screen.getByTestId("input-Diabetes"),
			"1",
		);
		const age = Number(
			screen.getByTestId("input-DEMO_RIDAGEYR").value,
		);
		const onsetAge = String(Math.max(1, age - 10));
		await user.clear(screen.getByTestId("input-DIQ_DID040"));
		await user.type(
			screen.getByTestId("input-DIQ_DID040"),
			onsetAge,
		);
		await user.clear(screen.getByTestId("input-homa_ir"));
		await user.type(screen.getByTestId("input-homa_ir"), "6.2");
		await user.click(screen.getByTestId("button-score-record"));
		await waitFor(() =>
			expect(global.fetch).toHaveBeenCalledTimes(1),
		);
		const [, options] = global.fetch.mock.calls[0];
		const payload = JSON.parse(options.body);
		expect(payload.confirm_explicit_scoring).toBe(true);
		expect(payload.patient_record.Diabetes).toBeUndefined();
		expect(payload.patient_record.DIQ_DID040).toBeUndefined();
		expect(await screen.findByText("1.235")).toBeInTheDocument();
		expect(screen.getAllByText("87.3%")).toHaveLength(2);
		expect(screen.getByText(/More unusual than/)).toHaveTextContent(
			"More unusual than 87.3% of the NHANES adult reference.",
		);
		expect(
			screen.getByText(/not a risk probability/i),
		).toBeInTheDocument();
		expect(screen.getByText("homa_ir")).toBeInTheDocument();
		expect(
			screen.getAllByText("HOMA-IR (derived)").length,
		).toBeGreaterThan(1);
		expect(screen.getAllByText("6.2 index").length).toBeGreaterThan(
			1,
		);
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
			screen.getAllByText("Probability unavailable"),
		).toHaveLength(4);
		const researchPanel = screen.getByTestId(
			"panel-research-association",
		);
		expect(
			within(researchPanel).getByText("Pancreatic cancer"),
		).toBeInTheDocument();
		expect(
			within(researchPanel).getByText("General cancers"),
		).toBeInTheDocument();
		expect(
			within(researchPanel).getByText(
				/equal research emphasis/i,
			),
		).toBeInTheDocument();
		expect(
			within(researchPanel).getByText(
				/No validated future cancer or diabetes risk model/i,
			),
		).toBeInTheDocument();
		expect(
			within(researchPanel).getByText(
				/may appear in more than one group/i,
			),
		).toBeInTheDocument();
		expect(
			within(researchPanel).getAllByText(
				/No matching measurement appears/i,
			),
		).toHaveLength(2);
		[
			"diabetes_related_cancer",
			"lifestyle_related_cancer",
			"cancer_related_diabetes",
			"lifestyle_related_diabetes",
		].forEach((pathway) => {
			const pathwayPanel = screen.getByTestId(
				`pathway-${pathway}`,
			);
			expect(pathwayPanel).toBeInTheDocument();
			expect(
				within(pathwayPanel).getByText(
					"Probability unavailable",
				),
			).toBeInTheDocument();
			expect(
				within(pathwayPanel).getByText(
					/Cross-sectional records cannot/i,
				),
			).toBeInTheDocument();
		});
		const contextPanel = screen
			.getByRole("heading", {
				name: "Submitted diabetes context",
			})
			.closest("section");
		expect(
			within(contextPanel).getByText(
				/not sent to or scored/i,
			),
		).toBeInTheDocument();
		expect(
			within(contextPanel).getByText("Yes"),
		).toBeInTheDocument();
		expect(
			within(contextPanel).getByText(
				`${onsetAge} years (age at diabetes onset)`,
			),
		).toBeInTheDocument();
		expect(
			within(
				screen.getByTestId("panel-data-readiness"),
			).getByText("C-peptide"),
		).toBeInTheDocument();
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
