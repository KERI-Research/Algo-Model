import React from "react";
import { createRoot } from "react-dom/client";
import CausalDashboard from "./interface/dashboard";
import ResearchPanel from "./interface/research_panel";
import FutureRiskPanel from "./interface/future_risk_panel";
import "./styles.css";

const container = document.getElementById("root");
const root = createRoot(container);

root.render(
	<React.StrictMode>
		<CausalDashboard />
		<ResearchPanel />
		<FutureRiskPanel />
		<FutureRiskPanel />
	</React.StrictMode>,
);
