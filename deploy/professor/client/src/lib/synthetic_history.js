/**
 * Deterministic synthetic longitudinal history builder for the simulation panel.
 *
 * This produces a plausible sequence of visits for a synthetic person: irregular gaps,
 * per-visit missingness, and trajectories that drift coherently rather than jumping at random.
 * It reuses the cross-sectional archetypes so a synthetic history stays internally consistent.
 *
 * SIMULATION ONLY. These are not patients, and nothing scored from them is a clinical risk.
 */

import {
	createRandomSource,
	generateSyntheticPatient,
	SYNTHETIC_ARCHETYPES,
} from "./synthetic_patient";

/** Feature codes the future-risk models accept, in schema order. */
export const HISTORY_FEATURES = [
	"DEMO_RIDAGEYR",
	"BMX_BMXBMI",
	"BMX_BMXWAIST",
	"BMX_BMXWT",
	"GHB_LBXGH",
	"GLU_LBXGLU",
	"INS_LBXIN",
	"TCHOL_LBXTC",
	"HDL_LBDHDD",
	"TRIGLY_LBXTR",
	"BPX_SYSTOLIC",
];

/** Per-year drift applied to each feature, scaled by the archetype's trend direction. */
const YEARLY_DRIFT = {
	DEMO_RIDAGEYR: 1,
	BMX_BMXBMI: 0.25,
	BMX_BMXWAIST: 0.6,
	BMX_BMXWT: 0.7,
	GHB_LBXGH: 0.06,
	GLU_LBXGLU: 1.4,
	INS_LBXIN: 0.3,
	TCHOL_LBXTC: 1.1,
	HDL_LBDHDD: -0.3,
	TRIGLY_LBXTR: 3.5,
	BPX_SYSTOLIC: 0.8,
};

/**
 * Features the cross-sectional probe schema already covers are taken from it directly, so a
 * synthetic history is consistent with the synthetic patient generator. The remaining
 * longitudinal features are drawn from declared ranges, scaled by how metabolically adverse the
 * archetype is, so a "worsening" profile does not get a pristine lipid panel.
 */
const SHARED_PROBE_FEATURES = [
	"DEMO_RIDAGEYR",
	"BMX_BMXBMI",
	"BMX_BMXWAIST",
	"GHB_LBXGH",
	"GLU_LBXGLU",
	"INS_LBXIN",
];

const DECLARED_RANGES = {
	// [healthy low, healthy high, adverse low, adverse high]
	BMX_BMXWT: [58, 82, 82, 118],
	TCHOL_LBXTC: [150, 195, 195, 265],
	HDL_LBDHDD: [50, 72, 30, 46],
	TRIGLY_LBXTR: [60, 120, 150, 340],
	BPX_SYSTOLIC: [105, 124, 128, 165],
};

const round = (value, digits) => {
	const factor = 10 ** digits;
	return Math.round(value * factor) / factor;
};

/**
 * Build a synthetic longitudinal history.
 *
 * @param {object} options
 * @param {number} options.seed deterministic seed
 * @param {number} options.visits number of visits (default 5)
 * @param {string} options.archetypeId optional archetype to force
 * @param {number} options.measurementProbability chance a given lab is measured at a visit
 * @returns {{visits: Array, archetype: string, seed: number, simulationOnly: boolean}}
 */
export const generateSyntheticHistory = ({
	seed = 1,
	visits = 5,
	archetypeId = undefined,
	measurementProbability = 0.75,
} = {}) => {
	// One seeded source drives both the anchor profile and the trajectory, so a given seed
	// always reproduces the same history. `generateSyntheticPatient` defaults to Math.random,
	// so the source must be passed explicitly.
	const random = createRandomSource(seed);
	const base = generateSyntheticPatient({ random, archetype: archetypeId });
	const archetype =
		SYNTHETIC_ARCHETYPES.find((entry) => entry.id === base.archetype.id) ||
		SYNTHETIC_ARCHETYPES[0];
	const adverse = archetype.id !== "reference_range";
	// Adverse archetypes drift upward over time; a reference-range profile drifts mildly down.
	const direction = adverse ? 1 : -0.4;

	// Index-visit anchor values for every feature, coherent with the archetype.
	const anchor = {};
	SHARED_PROBE_FEATURES.forEach((feature) => {
		const value = Number(base.values[feature]);
		anchor[feature] = Number.isFinite(value) ? value : null;
	});
	Object.entries(DECLARED_RANGES).forEach(([feature, bounds]) => {
		const [low, high] = adverse ? [bounds[2], bounds[3]] : [bounds[0], bounds[1]];
		anchor[feature] = round(low + random() * (high - low), 1);
	});

	// Irregular, strictly decreasing days-before-index, oldest visit first.
	const gaps = [];
	for (let index = 0; index < visits - 1; index += 1) {
		gaps.push(150 + Math.round(random() * 420));
	}
	const offsets = [0];
	for (let index = gaps.length - 1; index >= 0; index -= 1) {
		offsets.unshift(offsets[0] + gaps[index]);
	}

	const rows = offsets.map((daysBeforeIndex, visitIndex) => {
		const yearsBeforeIndex = daysBeforeIndex / 365.25;
		const visit = {
			visit_index: visitIndex,
			days_before_index: daysBeforeIndex,
			years_before_index: round(yearsBeforeIndex, 2),
		};
		HISTORY_FEATURES.forEach((feature) => {
			const endValue = Number(anchor[feature]);
			if (!Number.isFinite(endValue)) {
				visit[feature] = null;
				return;
			}
			const drift = (YEARLY_DRIFT[feature] || 0) * direction;
			// Age always moves with calendar time; labs may be missing at a visit.
			const measured =
				feature === "DEMO_RIDAGEYR" ||
				visitIndex === offsets.length - 1 ||
				random() < measurementProbability;
			if (!measured) {
				visit[feature] = null;
				return;
			}
			const noise = (random() - 0.5) * (Math.abs(endValue) * 0.02);
			const value =
				feature === "DEMO_RIDAGEYR"
					? endValue - yearsBeforeIndex
					: endValue - drift * yearsBeforeIndex + noise;
			visit[feature] = round(Math.max(value, 0), 2);
		});
		return visit;
	});

	return {
		visits: rows,
		archetype: archetype.id,
		anchor,
		archetypeLabel: archetype.label || archetype.id,
		seed,
		simulationOnly: true,
		featureOrder: HISTORY_FEATURES,
	};
};

/** Reject a cross-sectional payload: future risk needs at least two distinct visit times. */
export const isLongitudinal = (visitRows) => {
	if (!Array.isArray(visitRows) || visitRows.length < 2) {
		return false;
	}
	const distinct = new Set(visitRows.map((row) => row.days_before_index));
	return distinct.size >= 2;
};

export const HISTORY_SAFETY_NOTE =
	"Synthetic research data. Simulated longitudinal history, not a patient record. " +
	"Any estimate produced from it is simulation only and is not a clinical risk.";