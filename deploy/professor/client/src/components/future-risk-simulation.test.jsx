import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import FutureRiskSimulation, {
	buildSimulationRows,
} from "./FutureRiskSimulation.jsx";

const json = (body, status = 200) => ({
	ok: status >= 200 && status < 300,
	status,
	headers: { get: () => "application/json" },
	json: async () => body,
	text: async () => JSON.stringify(body),
});

const CAPABILITY = {
	simulation_only: true,
	clinical_use: "prohibited",
	portable_artifact_available: true,
	parity: {
		verdict: "parity",
		histories_compared: 40,
		max_abs_difference: 1.727e-8,
	},
	outcomes: [
		{
			id: "type2_diabetes",
			label: "Type 2 diabetes",
			enabled: true,
		},
		{
			id: "pan_cancer",
			label: "Cancer (pan-cancer composite)",
			enabled: true,
		},
	],
	supported_horizons: {
		"type2_diabetes:3y": { selected_model: "discrete_time_hazard" },
		"type2_diabetes:5y": {
			selected_model: "gradient_boosted_trees",
		},
	},
	abstained_horizons: {
		"type2_diabetes:1y": { reason: "Did not pass the event gate." },
	},
	evaluation_caveats: [
		"Synthetic data only: nothing here estimates real patient risk.",
	],
	competing_outcomes:
		"Competing death is handled by the cause-specific model.",
};

const SCORES = {
	simulation_only: true,
	clinical_use: "prohibited",
	inference_backend: "numpy_portable",
	history: {
		visits: 5,
		history_days: 1200,
		visit_density_per_year: 1.52,
		missingness_burden: 0.2,
	},
	outcomes: {
		type2_diabetes: {
			horizons: {
				"1y": {
					status: "abstained",
					selected_model: null,
					models: {},
					reason: "Gate failed.",
				},
				"3y": {
					status: "simulated_estimate",
					selected_model: "discrete_time_hazard",
					models: {
						discrete_time_hazard: {
							raw_cumulative_incidence: 0.0412,
							calibrated_cumulative_incidence: 0.0357,
							calibration_method:
								"isotonic",
						},
					},
				},
				"5y": {
					status: "simulated_estimate",
					selected_model:
						"gradient_boosted_trees",
					models: {
						gradient_boosted_trees: {
							raw_cumulative_incidence: 0.1123,
							calibrated_cumulative_incidence: 0.0981,
							calibration_method:
								"platt_logistic",
						},
					},
				},
			},
		},
	},
	interpretation: "Synthetic estimate only; not a patient's risk.",
	persistence: "none: history discarded.",
};

describe("simulation horizon rows", () => {
	it("distinguishes abstained, ready and simulated rows", () => {
		const ready = buildSimulationRows({
			capability: CAPABILITY,
			scores: null,
			outcome: "type2_diabetes",
		});
		expect(ready.map((row) => row.status)).toEqual([
			"abstained",
			"ready",
			"ready",
		]);

		const scored = buildSimulationRows({
			capability: CAPABILITY,
			scores: SCORES,
			outcome: "type2_diabetes",
		});
		expect(scored.map((row) => row.status)).toEqual([
			"abstained",
			"simulated_estimate",
			"simulated_estimate",
		]);
		expect(scored[1].calibrated).toBe(0.0357);
	});
});

describe("FutureRiskSimulation", () => {
	beforeEach(() => {
		global.fetch = vi.fn((url) => {
			if (
				String(url).endsWith(
					"/api/v1/simulation/capability",
				)
			) {
				return Promise.resolve(json(CAPABILITY));
			}
			if (String(url).endsWith("/api/v1/simulation/score")) {
				return Promise.resolve(json(SCORES));
			}
			return Promise.resolve(
				json({ error: "Unknown route" }, 404),
			);
		});
	});

	afterEach(() => vi.restoreAllMocks());

	it("shows synthetic-only boundaries and deterministic history controls", async () => {
		render(<FutureRiskSimulation onUnauthorised={vi.fn()} />);
		expect(
			await screen.findByText("NumPy portable"),
		).toBeInTheDocument();
		expect(screen.getByText("Prohibited")).toBeInTheDocument();
		expect(
			screen.getByText(
				/Synthetic longitudinal data for software verification/,
			),
		).toBeInTheDocument();
		expect(
			screen.getByTestId("simulation-history"),
		).toBeInTheDocument();
		expect(
			screen.getByTestId("simulation-horizon-1y"),
		).toHaveTextContent("abstained");
		expect(
			screen.getByTestId("simulation-horizon-3y"),
		).toHaveTextContent("ready");
	});

	it("scores the generated history and renders calibrated estimates", async () => {
		render(<FutureRiskSimulation onUnauthorised={vi.fn()} />);
		const user = userEvent.setup();
		await user.click(
			await screen.findByRole("button", {
				name: "Run simulation",
			}),
		);

		await waitFor(() =>
			expect(
				screen.getByTestId("simulation-horizon-3y"),
			).toHaveTextContent("3.57%"),
		);
		expect(
			screen.getByTestId("simulation-horizon-5y"),
		).toHaveTextContent("9.81%");
		expect(
			screen.getByText(/history discarded/),
		).toBeInTheDocument();

		const scoreCall = global.fetch.mock.calls.find(([url]) =>
			String(url).endsWith("/api/v1/simulation/score"),
		);
		const payload = JSON.parse(scoreCall[1].body);
		expect(payload.simulation_mode).toBe(true);
		expect(payload.visits).toHaveLength(5);
		expect(payload.visits[0]).not.toHaveProperty("name");
	});
});
