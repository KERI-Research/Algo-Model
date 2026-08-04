import { useState } from "react";
import { login } from "../lib/api.js";
import { Brand, Notice } from "./common.jsx";

/**
 * Login screen. The plaintext key is held in a controlled input for exactly as
 * long as the submit takes, then cleared. It is never stored, never logged and
 * never placed in a URL; the server hashes it with SHA-256 and compares the
 * digest in constant time.
 */
export default function Login({ onAuthenticated, status, expired = false }) {
	const [accessKey, setAccessKey] = useState("");
	const [pending, setPending] = useState(false);
	const [error, setError] = useState(null);

	const submit = async (event) => {
		event.preventDefault();
		if (!accessKey || pending) {
			return;
		}
		setPending(true);
		setError(null);
		try {
			const result = await login(accessKey);
			setAccessKey("");
			onAuthenticated(result);
		} catch (failure) {
			setAccessKey("");
			setError(
				failure.status === 429
					? "Too many attempts. Wait a few minutes before trying again."
					: failure.status === 503
						? "Access control is not configured on this deployment yet."
						: "Invalid access key.",
			);
		} finally {
			setPending(false);
		}
	};

	return (
		<div className="login-shell">
			<main className="login-card">
				<Brand subtitle="Research review dashboard" />
				<h1 style={{ marginBottom: "var(--space-2)" }}>Restricted access</h1>
				<p style={{ fontSize: "var(--text-sm)" }}>
					Enter the access key issued for this review session. This
					deployment holds no patient data and is not a clinical system.
				</p>

				{expired && !error ? (
					<Notice kind="caution" title="Session ended.">
						<span data-testid="text-session-expired">
							Your session expired or was invalidated. Sign in again to
							continue.
						</span>
					</Notice>
				) : null}

				{error ? (
					<Notice kind="blocked" title="Sign-in failed">
						<span data-testid="text-login-error">{error}</span>
					</Notice>
				) : null}

				{status && status.auth_configured === false ? (
					<Notice kind="caution" title="Not configured.">
						The server has no access key set, so sign-in is disabled.
					</Notice>
				) : null}

				<form onSubmit={submit} noValidate>
					<label htmlFor="access-key">Access key</label>
					<input
						id="access-key"
						name="access-key"
						type="password"
						autoComplete="off"
						autoCapitalize="off"
						spellCheck="false"
						value={accessKey}
						onChange={(event) => setAccessKey(event.target.value)}
						aria-describedby="access-key-hint"
						data-testid="input-access-key"
						required
					/>
					<p className="field-hint" id="access-key-hint">
						The key is hashed before it is checked. It is never stored
						in this browser.
					</p>
					<button
						type="submit"
						className="btn"
						style={{ width: "100%", marginTop: "var(--space-4)" }}
						disabled={pending || !accessKey}
						data-testid="button-login"
					>
						{pending ? (
							<>
								<span className="spinner" aria-hidden="true" />
								Checking
							</>
						) : (
							"Open dashboard"
						)}
					</button>
				</form>

				<div className="login-meta">
					MetaboGuard - non-diagnostic metabolic research, KERI department.
					Prototype hosting: do not upload identifiable or clinical patient
					data.
				</div>
			</main>
		</div>
	);
}
