import { useCallback, useEffect, useState } from "react";
import DatasetAnalysis from "./components/DatasetAnalysis.jsx";
import Evidence from "./components/Evidence.jsx";
import HowItWorks from "./components/HowItWorks.jsx";
import Login from "./components/Login.jsx";
import Overview from "./components/Overview.jsx";
import PatientProbe from "./components/PatientProbe.jsx";
import Reliability from "./components/Reliability.jsx";
import { Brand, Loading } from "./components/common.jsx";
import {
	deploymentStatus,
	logout as logoutRequest,
	sessionStatus,
	setSessionToken,
} from "./lib/api.js";

const SECTIONS = [
	{
		id: "overview",
		label: "Overview",
		title: "Overview",
		lede: "Posture, model version, capability and the limits the current data impose.",
		Component: Overview,
	},
	{
		id: "probe",
		label: "Patient Probe",
		title: "Patient Probe",
		lede: "Score one record on request and read the deviation output with its boundaries.",
		Component: PatientProbe,
	},
	{
		id: "dataset",
		label: "Dataset Analysis",
		title: "Dataset Analysis",
		lede: "Screen a de-identified CSV, then run the accepted rows through the deployed model.",
		Component: DatasetAnalysis,
	},
	{
		id: "reliability",
		label: "Reliability & Clusters",
		title: "Reliability and clusters",
		lede: "Fail-closed reliability audit, feature tiers, and the clustering abstention.",
		Component: Reliability,
	},
	{
		id: "evidence",
		label: "Evidence & Methods",
		title: "Evidence and methods",
		lede: "Source-linked biomarker catalogue, reporting standards, and claim boundaries.",
		Component: Evidence,
	},
];

/**
 * Reachable from Overview and Evidence & Methods, deliberately not in the
 * sidebar: five working sections stay uncluttered, and this explainer is one
 * click away from both places a reader would look for it.
 */
const AUX_SECTIONS = [
	{
		id: "how",
		title: "How the AI works",
		lede: "What the model learns, what it reports, and why longitudinal data is the blocker.",
		Component: HowItWorks,
		parent: "overview",
	},
];

export default function App() {
	const [checking, setChecking] = useState(true);
	const [authenticated, setAuthenticated] = useState(false);
	const [expiredNotice, setExpiredNotice] = useState(false);
	const [status, setStatus] = useState(null);
	const [active, setActive] = useState("overview");

	useEffect(() => {
		let cancelled = false;
		Promise.all([
			sessionStatus().catch(() => ({ authenticated: false })),
			deploymentStatus().catch(() => null),
		]).then(([session, deployment]) => {
			if (cancelled) {
				return;
			}
			setAuthenticated(Boolean(session && session.authenticated));
			setStatus(deployment);
			setChecking(false);
		});
		return () => {
			cancelled = true;
		};
	}, []);

	const handleUnauthorised = useCallback(() => {
		setSessionToken(null);
		setAuthenticated(false);
		setExpiredNotice(true);
		setActive("overview");
	}, []);

	const handleLogout = async () => {
		await logoutRequest().catch(() => undefined);
		setAuthenticated(false);
		setExpiredNotice(false);
		setActive("overview");
	};

	if (checking) {
		return (
			<div className="login-shell">
				<div style={{ width: "100%", maxWidth: 420 }}>
					<Loading label="Verifying session" rows={3} />
				</div>
			</div>
		);
	}

	// The frontend shows nothing but the login screen until the server confirms a session.
	if (!authenticated) {
		return (
			<Login
				status={status}
				expired={expiredNotice}
				onAuthenticated={() => {
					setExpiredNotice(false);
					setAuthenticated(true);
				}}
			/>
		);
	}

	const section =
		SECTIONS.find((item) => item.id === active) ||
		AUX_SECTIONS.find((item) => item.id === active) ||
		SECTIONS[0];
	const View = section.Component;

	return (
		<div className="app">
			<a className="skip-link" href="#main-content">
				Skip to main content
			</a>
			<header className="sidebar">
				<Brand />
				<nav className="nav" aria-label="Dashboard sections">
					{SECTIONS.map((item, index) => (
						<button
							type="button"
							key={item.id}
							className="nav-item"
							data-parent-of-active={
								item.id === section.parent ? "true" : undefined
							}
							aria-current={item.id === active ? "page" : undefined}
							onClick={() => setActive(item.id)}
							data-testid={`nav-${item.id}`}
						>
							<span className="nav-index" aria-hidden="true">
								{String(index + 1).padStart(2, "0")}
							</span>
							{item.label}
						</button>
					))}
				</nav>
				<button
					type="button"
					className="nav-aside"
					aria-current={active === "how" ? "page" : undefined}
					onClick={() => setActive("how")}
					data-testid="nav-how"
				>
					How the AI works
				</button>
				<div className="sidebar-foot">
					<span>
						Non-diagnostic research. Prototype hosting: no identifiable or clinical
						patient data.
					</span>
					<button
						type="button"
						className="btn btn-secondary"
						onClick={handleLogout}
						data-testid="button-logout"
					>
						Sign out
					</button>
				</div>
			</header>

			<main className="main" id="main-content">
				<div className="page-head">
					<h1>{section.title}</h1>
					<p>{section.lede}</p>
				</div>
				<View
					key={section.id}
					onUnauthorised={handleUnauthorised}
					onNavigate={setActive}
				/>
			</main>
		</div>
	);
}
