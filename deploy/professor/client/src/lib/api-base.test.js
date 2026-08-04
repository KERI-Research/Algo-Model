import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * Deployment-mode resolution. Each case re-imports the module so the
 * build-time env is read afresh.
 */
const loadModule = async (env) => {
	vi.resetModules();
	vi.stubGlobal("import.meta.env", env);
	for (const [key, value] of Object.entries(env)) {
		import.meta.env[key] = value;
	}
	return import("./api.js");
};

const clearEnv = () => {
	delete import.meta.env.VITE_DEPLOY_TARGET;
	delete import.meta.env.VITE_API_BASE;
};

describe("API base resolution", () => {
	afterEach(() => {
		clearEnv();
		vi.unstubAllGlobals();
		vi.resetModules();
	});

	it("uses same-origin /api/v1 on Vercel and ignores the pplx sentinel", async () => {
		const module = await loadModule({ VITE_DEPLOY_TARGET: "vercel" });
		expect(module.API_BASE).toBe("");
		expect(module.DEPLOY_TARGET).toBe("vercel");
		global.fetch = vi.fn(() =>
			Promise.resolve({
				ok: true,
				status: 200,
				headers: { get: () => "application/json" },
				json: async () => ({}),
				text: async () => "{}",
			}),
		);
		await module.apiGet("/api/v1/model");
		expect(global.fetch.mock.calls[0][0]).toBe("/api/v1/model");
		expect(global.fetch.mock.calls[0][1].credentials).toBe("include");
	});

	it("honours an explicit VITE_API_BASE override", async () => {
		const module = await loadModule({ VITE_API_BASE: "https://api.example.org" });
		expect(module.API_BASE).toBe("https://api.example.org");
	});

	it("falls back to same-origin when the pplx sentinel is unreplaced", async () => {
		clearEnv();
		const module = await loadModule({});
		expect(module.API_BASE).toBe("");
		expect(module.DEPLOY_TARGET).toBe("pplx");
	});
});
