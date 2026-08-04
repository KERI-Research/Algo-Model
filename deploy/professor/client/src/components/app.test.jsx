import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App.jsx";
import { setSessionToken } from "../lib/api.js";

const json = (body, status = 200) => ({
	ok: status >= 200 && status < 300,
	status,
	headers: { get: () => "application/json" },
	json: async () => body,
	text: async () => JSON.stringify(body),
});

const MODEL = {
	model_name: "MetaboGuard-SSL",
	model_version: "metaboguard-ssl-v1",
	code_version: "metaboguard-ssl-v1.1",
	deployment_version: "professor-dashboard-1.0.0",
	inference_backend: "numpy",
	training_rows: 50000,
	reference_rows_scored: 63041,
	intended_use: "Research discovery.",
	architecture: { type: "Denoising autoencoder", latent_dimension: 16 },
	supported_outputs: ["metabolic_deviation_score"],
	prohibited_outputs: ["cancer type or site claim"],
	non_diagnostic_warning: "Research use only, non-diagnostic.",
};

const OVERVIEW = {
	posture: { headline: "Non-diagnostic discovery research", statement: "Discovery posture." },
	model: MODEL,
	status_cards: [
		{ id: "inference", label: "Inference path", value: "NumPy artifact", state: "ok", detail: "d" },
	],
	data_limitations: [
		{ id: "cross_sectional", title: "Cross-sectional data only", detail: "No horizons." },
	],
	integrity: { status: "ok" },
	non_diagnostic_warning: "Clinician review is required.",
};

const route = (url) => {
	if (url.endsWith("/api/v1/session")) {
		return json({ authenticated: false });
	}
	if (url.endsWith("/api/v1/status")) {
		return json({ auth_configured: true, persistence: "none" });
	}
	if (url.endsWith("/api/v1/auth/login")) {
		return json({ authenticated: true, session_token: "tok.sig" });
	}
	if (url.endsWith("/api/v1/overview")) {
		return json(OVERVIEW);
	}
	if (url.endsWith("/api/v1/auth/logout")) {
		return json({ authenticated: false });
	}
	return json({});
};

describe("App shell", () => {
	beforeEach(() => {
		setSessionToken(null);
		global.fetch = vi.fn((url) => Promise.resolve(route(String(url))));
	});

	afterEach(() => vi.restoreAllMocks());

	it("shows only the login screen until the server confirms a session", async () => {
		render(<App />);
		expect(await screen.findByTestId("input-access-key")).toBeInTheDocument();
		expect(screen.queryByTestId("nav-dataset")).not.toBeInTheDocument();
		expect(screen.queryByTestId("button-logout")).not.toBeInTheDocument();
	});

	it("renders the dashboard after a successful login", async () => {
		render(<App />);
		const user = userEvent.setup();
		await user.type(await screen.findByTestId("input-access-key"), "a-key");
		await user.click(screen.getByTestId("button-login"));
		await waitFor(() =>
			expect(screen.getByTestId("nav-overview")).toHaveAttribute("aria-current", "page"),
		);
		expect(await screen.findByText("Cross-sectional data only")).toBeInTheDocument();
		expect(screen.getByTestId("nav-probe")).toBeInTheDocument();
		expect(screen.getByTestId("nav-evidence")).toBeInTheDocument();
	});

	it("shows a generic message when the key is rejected", async () => {
		global.fetch = vi.fn((url) =>
			String(url).endsWith("/api/v1/auth/login")
				? Promise.resolve(json({ error: "Invalid access key." }, 401))
				: Promise.resolve(route(String(url))),
		);
		render(<App />);
		const user = userEvent.setup();
		await user.type(await screen.findByTestId("input-access-key"), "wrong");
		await user.click(screen.getByTestId("button-login"));
		expect(await screen.findByTestId("text-login-error")).toHaveTextContent(
			"Invalid access key.",
		);
		expect(screen.getByTestId("input-access-key")).toHaveValue("");
	});

	it("returns to the login screen when a protected call reports an expired session", async () => {
		global.fetch = vi.fn((url) =>
			String(url).endsWith("/api/v1/overview")
				? Promise.resolve(json({ error: "Authentication required." }, 401))
				: Promise.resolve(route(String(url))),
		);
		render(<App />);
		const user = userEvent.setup();
		await user.type(await screen.findByTestId("input-access-key"), "a-key");
		await user.click(screen.getByTestId("button-login"));
		expect(await screen.findByTestId("text-session-expired")).toBeInTheDocument();
		expect(screen.getByTestId("input-access-key")).toBeInTheDocument();
	});

	it("signs out and hides the dashboard", async () => {
		render(<App />);
		const user = userEvent.setup();
		await user.type(await screen.findByTestId("input-access-key"), "a-key");
		await user.click(screen.getByTestId("button-login"));
		await user.click(await screen.findByTestId("button-logout"));
		expect(await screen.findByTestId("input-access-key")).toBeInTheDocument();
		expect(screen.queryByTestId("nav-dataset")).not.toBeInTheDocument();
	});
});
