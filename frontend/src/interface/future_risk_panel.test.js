/**
 * Tests for the simulation-only future-risk panel.
 *
 * Uses `react-dom/server` for rendering so no extra testing dependency is required. Effects do
 * not run during server rendering, so no API call is made. The abstention contract is tested
 * directly against the pure `buildHorizonRows` helper.
 */

import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

jest.mock("./service", () => ({
	fetchFutureRiskCapability: jest.fn(() => Promise.resolve({})),
	fetchSimulatedFutureRisk: jest.fn(() => Promise.resolve({})),
}));

// eslint-disable-next-line import/first
import FutureRiskPanel, { buildHorizonRows } from "./future_risk_panel";
// eslint-disable-next-line import/first
import {
	generateSyntheticHistory,
	HISTORY_FEATURES,
	HISTORY_SAFETY_NOTE,
	isLongitudinal,
} from "./synthetic_history";

const capability = {
	clinical_future_risk_enabled: false,
	clinical_future_risk_blocker: "No real longitudinal cohort exists.",
	simulated_future_risk_enabled: true,
	simulated_future_risk_requires: "simulation_mode=true plus a simulation-only artefact",
	event_gate: { minimum_events: 50, minimum_non_events: 50 },
	disabled_outcomes: ["type1_diabetes", "cancer_site"],
	artifact: {
		selection: {
			"type2_diabetes:3y": { selected_model: "discrete_time_hazard" },
			"type2_diabetes:5y": { selected_model: "horizon_logistic" },
		},
	},
};

const scores = {
	simulation_only: true,
	horizons: {
		"1y": { selected_model: null, models: {} },
		"3y": {
			selected_model: "discrete_time_hazard",
			models: {
				discrete_time_hazard: {
					raw_cumulative_incidence: 0.0412,
					calibrated_cumulative_incidence: 0.0357,
				},
			},
		},
		"5y": {
			selected_model: "horizon_logistic",
			models: {
				horizon_logistic: {
					raw_cumulative_incidence: 0.1123,
					calibrated_cumulative_incidence: 0.0981,
				},
			},
		},
	},
};

describe("synthetic longitudinal history generator", () => {
	it("is deterministic for a fixed seed", () => {
		expect(generateSyntheticHistory({ seed: 42 }).visits).toEqual(
			generateSyntheticHistory({ seed: 42 }).visits,
		);
	});

	it("produces a different history for a different seed", () => {
		expect(generateSyntheticHistory({ seed: 1 }).visits).not.toEqual(
			generateSyntheticHistory({ seed: 2 }).visits,
		);
	});

	it("orders visits oldest first with strictly decreasing offsets ending at the index", () => {
		const offsets = generateSyntheticHistory({ seed: 7 }).visits.map(
			(visit) => visit.days_before_index,
		);
		expect(offsets[offsets.length - 1]).toBe(0);
		for (let index = 1; index < offsets.length; index += 1) {
			expect(offsets[index]).toBeLessThan(offsets[index - 1]);
		}
	});

	it("always measures age and fully populates the index visit", () => {
		const { visits } = generateSyntheticHistory({ seed: 11 });
		visits.forEach((visit) => expect(typeof visit.DEMO_RIDAGEYR).toBe("number"));
		const indexVisit = visits[visits.length - 1];
		HISTORY_FEATURES.forEach((feature) =>
			expect(indexVisit[feature]).not.toBeNull(),
		);
	});

	it("ages the person forward through the history", () => {
		const { visits } = generateSyntheticHistory({ seed: 5 });
		expect(visits[visits.length - 1].DEMO_RIDAGEYR).toBeGreaterThan(
			visits[0].DEMO_RIDAGEYR,
		);
	});

	it("emits explicit nulls for unmeasured labs instead of inventing values", () => {
		const { visits } = generateSyntheticHistory({
			seed: 3,
			visits: 6,
			measurementProbability: 0,
		});
		const missing = visits
			.slice(0, visits.length - 1)
			.some((visit) =>
				HISTORY_FEATURES.some(
					(feature) => feature !== "DEMO_RIDAGEYR" && visit[feature] === null,
				),
			);
		expect(missing).toBe(true);
	});

	it("keeps requested visit counts and marks itself simulation only", () => {
		const history = generateSyntheticHistory({ seed: 9, visits: 4 });
		expect(history.visits).toHaveLength(4);
		expect(history.simulationOnly).toBe(true);
	});

	it("rejects cross-sectional input", () => {
		expect(isLongitudinal([])).toBe(false);
		expect(isLongitudinal([{ days_before_index: 0 }])).toBe(false);
		expect(
			isLongitudinal([{ days_before_index: 0 }, { days_before_index: 0 }]),
		).toBe(false);
		expect(
			isLongitudinal([{ days_before_index: 400 }, { days_before_index: 0 }]),
		).toBe(true);
	});
});

describe("horizon row contract", () => {
	it("returns a calibrated simulated estimate for gated horizons", () => {
		const rows = buildHorizonRows({ scores, capability, outcome: "type2_diabetes" });
		const threeYear = rows.find((row) => row.key === "3y");
		expect(threeYear.status).toBe("simulated_estimate");
		expect(threeYear.calibrated).toBeCloseTo(0.0357);
		expect(threeYear.raw).toBeCloseTo(0.0412);
		expect(threeYear.model).toBe("discrete_time_hazard");
	});

	it("abstains for a horizon that did not pass the event gate", () => {
		const rows = buildHorizonRows({ scores, capability, outcome: "type2_diabetes" });
		const oneYear = rows.find((row) => row.key === "1y");
		expect(oneYear.status).toBe("abstained");
		expect(oneYear.calibrated).toBeNull();
		expect(oneYear.note).toMatch(/did not pass the 50-event gate/);
	});

	it("abstains for every horizon of an outcome with no gated model", () => {
		const rows = buildHorizonRows({ scores, capability, outcome: "pan_cancer" });
		expect(rows.every((row) => row.status === "abstained")).toBe(true);
	});

	it("abstains when there are no scores at all", () => {
		const rows = buildHorizonRows({
			scores: null,
			capability,
			outcome: "type2_diabetes",
		});
		expect(rows.every((row) => row.status === "abstained")).toBe(true);
	});

	it("abstains when capability reports no artifact", () => {
		const rows = buildHorizonRows({ scores, capability: {}, outcome: "type2_diabetes" });
		expect(rows.every((row) => row.status === "abstained")).toBe(true);
	});
});

describe("future risk panel markup", () => {
	const markup = renderToStaticMarkup(<FutureRiskPanel />);

	it("shows the simulation-only banner", () => {
		expect(markup).toContain("SIMULATION ONLY");
	});

	it("shows the synthetic research data indicator and safety note", () => {
		expect(markup).toContain("Synthetic research data");
		expect(markup).toContain(HISTORY_SAFETY_NOTE.slice(0, 40));
	});

	it("renders a synthetic visit history table with every schema feature", () => {
		HISTORY_FEATURES.forEach((feature) => expect(markup).toContain(feature));
		expect(markup).toContain("Synthetic visit history");
	});

	it("renders all three horizon rows, abstaining before any scoring", () => {
		expect(markup).toContain("1 year");
		expect(markup).toContain("3 years");
		expect(markup).toContain("5 years");
		expect(markup.match(/abstained/g).length).toBeGreaterThanOrEqual(3);
	});

	it("offers generate and score controls", () => {
		expect(markup).toContain("Generate synthetic history");
		expect(markup).toContain("Score simulated history");
	});

	it("never renders a clinical risk claim", () => {
		expect(markup).not.toMatch(/is a diagnosis/i);
		expect(markup).not.toMatch(/your risk/i);
		expect(markup).not.toMatch(/clinical risk estimate/i);
		// The only diagnosis-adjacent wording allowed is an explicit disclaimer.
		expect(markup).toContain("are not a diagnosis");
	});
});