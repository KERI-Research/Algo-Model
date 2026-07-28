/**
 * Python Backend Integration Service
 * ====================================
 * This module handles all asynchronous communication between the React interface
 * and the Python DoWhy modeling backend. Adopting a skeptical approach, it includes
 * robust error handling to catch timeout issues or malformed data responses when
 * processing massive datasets. This centralizes our network logic, making it trivial
 * to swap out endpoints when we transition from local prototyping to a deployed
 * multi-agent architecture.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const buildApiError = async (response, fallbackMessage) => {
	let detailMessage = "";
	try {
		const payload = await response.json();
		const detail = payload?.detail;
		if (typeof detail === "string") {
			detailMessage = detail;
		} else if (detail?.message) {
			detailMessage = detail.message;
		}
	} catch {
		detailMessage = "";
	}

	const suffix = detailMessage ? ` | detail: ${detailMessage}` : "";
	return new Error(
		`${fallbackMessage} with status: ${response.status}${suffix}`,
	);
};

export const fetchDatasetCatalog = async () => {
	const response = await fetch(`${API_BASE_URL}/api/v1/datasets`);

	if (!response.ok) {
		throw await buildApiError(
			response,
			"Failed to load dataset catalog",
		);
	}

	return await response.json();
};

export const fetchDatasetPreview = async (datasetName) => {
	const response = await fetch(`${API_BASE_URL}/api/v1/dataset-preview`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({ dataset: datasetName }),
	});

	if (!response.ok) {
		throw await buildApiError(
			response,
			"Failed to load dataset preview",
		);
	}

	return await response.json();
};

export const fetchModelResults = async (datasetName, options = {}) => {
	const {
		treatment = "Diabetes",
		outcome = "Cancer",
		allowFallback = true,
	} = options;

	const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			dataset: datasetName,
			treatment,
			outcome,
			allow_fallback: allowFallback,
		}),
	});

	if (!response.ok) {
		throw await buildApiError(
			response,
			"Backend computation failed",
		);
	}

	return await response.json();
};

export const fetchPredictiveBaseline = async (datasetName) => {
	const response = await fetch(
		`${API_BASE_URL}/api/v1/predictive-baseline`,
		{
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ dataset: datasetName }),
		},
	);

	if (!response.ok) {
		throw await buildApiError(
			response,
			"Predictive baseline failed",
		);
	}

	return await response.json();
};

export const fetchBiomarkerDiscovery = async (datasetName, options = {}) => {
	const {
		patientRecord = null,
		topK = 8,
		forceRetrain = false,
	} = options;

	const response = await fetch(
		`${API_BASE_URL}/api/v1/biomarker-discovery`,
		{
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				dataset: datasetName,
				patient_record: patientRecord,
				top_k: topK,
				force_retrain: forceRetrain,
			}),
		},
	);

	if (!response.ok) {
		throw await buildApiError(
			response,
			"Biomarker discovery failed",
		);
	}

	return await response.json();
};

export const fetchPreventionCapabilities = async (datasetName) => {
	const response = await fetch(
		`${API_BASE_URL}/api/v1/prevention-capabilities`,
		{
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ dataset: datasetName }),
		},
	);
	if (!response.ok) {
		throw await buildApiError(
			response,
			"Failed to evaluate prevention capabilities",
		);
	}
	return await response.json();
};

export const fetchPreventionScore = async (
	patientRecord,
	artifact = "nhanes_multicycle_v2",
) => {
	const response = await fetch(`${API_BASE_URL}/api/v1/prevention-score`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			patient_record: patientRecord,
			artifact,
		}),
	});
	if (!response.ok) {
		throw await buildApiError(
			response,
			"Metabolic deviation scoring failed",
		);
	}
	return await response.json();
};
