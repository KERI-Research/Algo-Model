/**
 * Static render checks for the synthetic-patient control in the Biomarker Probe.
 *
 * Uses `react-dom/server` so no extra testing dependency is required. Effects do not run
 * during server rendering, so no API call is made.
 */

import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

jest.mock("./service", () => ({
	fetchDatasetCatalog: jest.fn(() => Promise.resolve({ datasets: [] })),
	fetchDatasetPreview: jest.fn(() => Promise.resolve({})),
	fetchAnalysis: jest.fn(() => Promise.resolve({})),
	fetchPredictiveBaseline: jest.fn(() => Promise.resolve({})),
	fetchBiomarkerDiscovery: jest.fn(() => Promise.resolve({})),
	fetchPreventionCapabilities: jest.fn(() => Promise.resolve({})),
	fetchPreventionScore: jest.fn(() => Promise.resolve({})),
	fetchDataReliability: jest.fn(() => Promise.resolve({})),
	fetchEvidenceCatalogue: jest.fn(() => Promise.resolve({})),
	fetchResearchClusters: jest.fn(() => Promise.resolve({})),
}));

// eslint-disable-next-line import/first
import CausalDashboard from "./dashboard";
// eslint-disable-next-line import/first
import * as service from "./service";

describe("synthetic patient control in the Biomarker Probe", () => {
	const markup = renderToStaticMarkup(<CausalDashboard />);

	test("renders a real button with concise visible text", () => {
		expect(markup).toContain('type="button"');
		expect(markup).toContain("Generate synthetic");
		expect(markup).toContain("patient");
		expect(markup).toContain('class="synthetic-button"');
	});

	test("button carries a descriptive accessible label and description", () => {
		expect(markup).toContain(
			'aria-label="Generate synthetic patient: fill every biomarker probe input with fabricated test values"',
		);
		expect(markup).toContain('aria-describedby="synthetic-patient-note"');
		expect(markup).toContain('id="synthetic-patient-note"');
	});

	test("the note states it does not run the model and fields stay editable", () => {
		const text = markup.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
		expect(text).toContain("fabricated values for interface testing");
		expect(text).toContain("does not run the model");
		expect(text).toContain("every field stays editable");
	});

	test("no synthetic indicator is shown before the button is used", () => {
		expect(markup).not.toContain("synthetic-indicator");
	});

	test("the probe inputs the generator targets are present", () => {
		expect(markup).toContain("patient-form-grid");
		["Age", "Sex", "BMI", "Waist", "HbA1c", "Glucose", "Insulin"].forEach(
			(label) => {
				expect(markup.replace(/<[^>]+>/g, " ")).toContain(label);
			},
		);
	});

	test("rendering does not call the scoring API", () => {
		expect(service.fetchBiomarkerDiscovery).not.toHaveBeenCalled();
		expect(service.fetchPreventionScore).not.toHaveBeenCalled();
	});
});