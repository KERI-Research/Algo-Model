import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	API_BASE,
	ApiError,
	apiGet,
	hasMemoryToken,
	login,
	logout,
	setSessionToken,
} from "./api.js";

const jsonResponse = (body, status = 200) => ({
	ok: status >= 200 && status < 300,
	status,
	headers: { get: () => "application/json" },
	json: async () => body,
	text: async () => JSON.stringify(body),
});

describe("api client", () => {
	beforeEach(() => {
		setSessionToken(null);
		global.fetch = vi.fn();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("falls back to same-origin requests when the deploy sentinel is not rewritten", () => {
		expect(API_BASE).toBe("");
	});

	it("sends credentials so the __Host- session cookie travels with the request", async () => {
		global.fetch.mockResolvedValue(jsonResponse({ ok: true }));
		await apiGet("/api/v1/model");
		const [url, options] = global.fetch.mock.calls[0];
		expect(url).toBe("/api/v1/model");
		expect(options.credentials).toBe("include");
		expect(options.headers.Authorization).toBeUndefined();
	});

	it("keeps the session token in memory only and sends it as a bearer header", async () => {
		global.fetch.mockResolvedValue(
			jsonResponse({ authenticated: true, session_token: "abc.def" }),
		);
		await login("a-key");
		expect(hasMemoryToken()).toBe(true);
		global.fetch.mockResolvedValue(jsonResponse({ ok: true }));
		await apiGet("/api/v1/model");
		const [, options] = global.fetch.mock.calls[1];
		expect(options.headers.Authorization).toBe("Bearer abc.def");
		// Nothing may be written to browser storage.
		expect(window.localStorage.length).toBe(0);
		expect(window.sessionStorage.length).toBe(0);
	});

	it("never places the access key in the request URL", async () => {
		global.fetch.mockResolvedValue(jsonResponse({ authenticated: true }));
		await login("super-secret-key");
		const [url, options] = global.fetch.mock.calls[0];
		expect(url).not.toContain("super-secret-key");
		expect(options.body).toContain("super-secret-key");
	});

	it("clears the in-memory token on logout", async () => {
		setSessionToken("abc.def");
		global.fetch.mockResolvedValue(jsonResponse({ authenticated: false }));
		await logout();
		expect(hasMemoryToken()).toBe(false);
	});

	it("raises ApiError with the status and server detail", async () => {
		global.fetch.mockResolvedValue(
			jsonResponse({ error: "Authentication required.", status_code: 401 }, 401),
		);
		await expect(apiGet("/api/v1/model")).rejects.toMatchObject({
			name: "ApiError",
			status: 401,
			message: "Authentication required.",
		});
		expect(new ApiError("x", 400).status).toBe(400);
	});

	it("unwraps structured error details", async () => {
		global.fetch.mockResolvedValue(
			jsonResponse(
				{
					error: { message: "File rejected.", identifier_columns: [] },
					status_code: 422,
				},
				422,
			),
		);
		await expect(apiGet("/api/v1/dataset/inspect")).rejects.toMatchObject({
			status: 422,
			message: "File rejected.",
		});
	});
});
