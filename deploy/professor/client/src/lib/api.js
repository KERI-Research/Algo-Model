/**
 * API client.
 *
 * Deployment routing has three modes, resolved once at module load:
 *
 * 1. Vercel (`VITE_DEPLOY_TARGET=vercel`, set by `npm run build:vercel`):
 *    same-origin requests to `/api/v1/...`, which `vercel.json` rewrites to the
 *    Python function. The pplx proxy sentinel is not consulted at all.
 * 2. Explicit override (`VITE_API_BASE=https://host` or a path prefix): used as
 *    given. Handy for a split frontend/backend deployment.
 * 3. Perplexity sandbox (default): `__PORT_5000__` is a build-time sentinel that
 *    the deploy/publish step rewrites to the backend proxy path (`port/5000`).
 *    When it is still a sentinel - local development - requests are same-origin.
 *
 * Session transport: the server sets a `__Host-` prefixed, HttpOnly, Secure,
 * SameSite=Strict cookie, and `credentials: "include"` sends it. Some embedded
 * preview hosts block cookie storage entirely; for those the login response
 * also returns the signed session token, which is held in module memory only
 * (never localStorage, sessionStorage, IndexedDB or a URL) and sent as a bearer
 * header. No key, hash or secret is ever stored client side.
 */

const PPLX_PROXY_SENTINEL = "__PORT_5000__";

const resolveApiBase = () => {
	if (import.meta.env.VITE_DEPLOY_TARGET === "vercel") {
		return "";
	}
	const explicit = import.meta.env.VITE_API_BASE;
	if (typeof explicit === "string" && explicit.length > 0) {
		return explicit;
	}
	return PPLX_PROXY_SENTINEL.startsWith("__") ? "" : PPLX_PROXY_SENTINEL;
};

export const API_BASE = resolveApiBase();
export const DEPLOY_TARGET = import.meta.env.VITE_DEPLOY_TARGET || "pplx";

let sessionToken = null;

export const setSessionToken = (token) => {
	sessionToken = token || null;
};

export const hasMemoryToken = () => Boolean(sessionToken);

const url = (path) => `${API_BASE}${path}`;

const authHeaders = () =>
	sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {};

export class ApiError extends Error {
	constructor(message, status, detail) {
		super(message);
		this.name = "ApiError";
		this.status = status;
		this.detail = detail;
	}
}

const parse = async (response) => {
	const type = response.headers.get("content-type") || "";
	if (type.includes("application/json")) {
		return response.json();
	}
	return response.text();
};

const handle = async (response) => {
	const body = await parse(response);
	if (response.ok) {
		return body;
	}
	const detail = body && typeof body === "object" ? body.error : body;
	const message =
		typeof detail === "string"
			? detail
			: detail && detail.message
				? detail.message
				: `Request failed (${response.status}).`;
	throw new ApiError(message, response.status, detail);
};

export const apiGet = async (path) =>
	handle(
		await fetch(url(path), {
			method: "GET",
			credentials: "include",
			headers: { ...authHeaders() },
		}),
	);

export const apiPostJson = async (path, payload) =>
	handle(
		await fetch(url(path), {
			method: "POST",
			credentials: "include",
			headers: { "Content-Type": "application/json", ...authHeaders() },
			body: JSON.stringify(payload),
		}),
	);

export const apiPostForm = async (path, formData) =>
	handle(
		await fetch(url(path), {
			method: "POST",
			credentials: "include",
			headers: { ...authHeaders() },
			body: formData,
		}),
	);

export const apiPostForBlob = async (path, payload) => {
	const response = await fetch(url(path), {
		method: "POST",
		credentials: "include",
		headers: { "Content-Type": "application/json", ...authHeaders() },
		body: JSON.stringify(payload),
	});
	if (!response.ok) {
		return handle(response);
	}
	return response.blob();
};

export const login = async (accessKey) => {
	const body = await apiPostJson("/api/v1/auth/login", {
		access_key: accessKey,
	});
	// Held in memory only, for hosts where the cookie cannot be stored.
	setSessionToken(body.session_token);
	return body;
};

export const logout = async () => {
	try {
		await apiPostJson("/api/v1/auth/logout", {});
	} finally {
		setSessionToken(null);
	}
};

export const sessionStatus = () => apiGet("/api/v1/session");
export const deploymentStatus = () => apiGet("/api/v1/status");
