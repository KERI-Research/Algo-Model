/**
 * Prevention-feature extension for the synthetic patient generator.
 *
 * `synthetic_patient.js` is vendored byte-for-byte from the KERI repository and
 * covers the original Biomarker Probe fields. The professor Patient Probe
 * exposes the full prevention allowlist (`PREVENTION_FEATURES` in
 * `api/self_supervised.py`), so this module fills the remaining visible inputs
 * with internally coherent, fabricated values.
 *
 * Safety scope is unchanged from the vendored module:
 *   - values are FABRICATED for interface testing, never real measurements;
 *   - no outcome label, no `tcga_*` column and no post-diagnosis field is ever
 *     produced;
 *   - ranges sit inside the plausibility windows in `api/data_reliability.py`
 *     (`PLAUSIBLE_RANGES`), so a generated profile never trips a range flag.
 */

import {
	generateSyntheticPatient,
	REQUIRED_PROBE_FIELDS,
	SYNTHETIC_FIELD_SCHEMA,
	SYNTHETIC_SAFETY_NOTE,
} from "./synthetic_patient.js";

/**
 * Re-exported from the vendored module so consumers have one import path for
 * the whole synthetic-patient surface. Both modules are tracked source files:
 * `synthetic_patient.js` is a byte-for-byte copy of
 * `frontend/src/interface/synthetic_patient.js`, refreshed by
 * `prepare_assets.py`, and this module extends it with the remaining
 * prevention-allowlist fields.
 */
export { REQUIRED_PROBE_FIELDS, SYNTHETIC_FIELD_SCHEMA, generateSyntheticPatient };

/** Extra prevention-allowlist fields, with the units the API expects. */
export const PREVENTION_FIELD_SCHEMA = {
	DEMO_RIDRETH3: {
		kind: "categorical",
		label: "Race/ethnicity code",
		options: ["1", "2", "3", "4", "6", "7"],
		optionLabels: {
			1: "Mexican American",
			2: "Other Hispanic",
			3: "Non-Hispanic White",
			4: "Non-Hispanic Black",
			6: "Non-Hispanic Asian",
			7: "Other / multi-racial",
		},
	},
	TRIGLY_LBXTR: { kind: "number", label: "Triglycerides", unit: "mg/dL", min: 20, max: 900, precision: 0 },
	TRIGLY_LBDLDL: { kind: "number", label: "LDL cholesterol", unit: "mg/dL", min: 20, max: 300, precision: 0 },
	HDL_LBDHDD: { kind: "number", label: "HDL cholesterol", unit: "mg/dL", min: 15, max: 120, precision: 0 },
	TCHOL_LBXTC: { kind: "number", label: "Total cholesterol", unit: "mg/dL", min: 80, max: 400, precision: 0 },
	HSCRP_LBXHSCRP: { kind: "number", label: "hs-CRP", unit: "mg/L", min: 0.1, max: 60, precision: 2 },
	CBC_LBXHGB: { kind: "number", label: "Haemoglobin", unit: "g/dL", min: 7, max: 19, precision: 1 },
	CBC_LBXPLTSI: { kind: "number", label: "Platelet count", unit: "1000 cells/uL", min: 60, max: 600, precision: 0 },
	BIOPRO_LBXSATSI: { kind: "number", label: "ALT", unit: "U/L", min: 5, max: 200, precision: 0 },
	BIOPRO_LBXSAPSI: { kind: "number", label: "Alkaline phosphatase", unit: "U/L", min: 20, max: 300, precision: 0 },
	BIOPRO_LBXSCR: { kind: "number", label: "Creatinine", unit: "mg/dL", min: 0.3, max: 8, precision: 2 },
	smoking_status: {
		kind: "categorical",
		label: "Smoking status",
		options: ["0", "1", "2"],
		optionLabels: { 0: "Never", 1: "Former", 2: "Current" },
	},
	alcohol_status: {
		kind: "categorical",
		label: "Alcohol status",
		options: ["0", "1", "2"],
		optionLabels: { 0: "None", 1: "Moderate", 2: "Heavy" },
	},
	average_drinks_per_day: { kind: "number", label: "Average drinks per day", unit: "drinks", min: 0, max: 20, precision: 1 },
	weight_loss_1yr_lb: { kind: "number", label: "Weight change, past year", unit: "lb lost", min: -60, max: 80, precision: 0 },
	weight_loss_10yr_lb: { kind: "number", label: "Weight change, past 10 years", unit: "lb lost", min: -80, max: 120, precision: 0 },
	homa_ir: { kind: "number", label: "HOMA-IR (derived)", unit: "index", min: 0, max: 60, precision: 2 },
};

const round = (value, precision) => {
	const factor = 10 ** precision;
	return Math.round(value * factor) / factor;
};

const clampField = (field, value) => {
	const schema = PREVENTION_FIELD_SCHEMA[field];
	const bounded = Math.min(schema.max, Math.max(schema.min, value));
	return round(bounded, schema.precision).toFixed(schema.precision);
};

const uniform = (random, low, high) => low + random() * (high - low);

/**
 * Generate a full prevention-allowlist profile.
 *
 * Coherence rules (descriptive only, never a clinical claim):
 *   - lipids track BMI and glycaemia in the usual monotone direction;
 *   - total cholesterol is reconstructed from HDL, LDL and triglycerides;
 *   - hs-CRP scales with adiposity;
 *   - HOMA-IR is derived from the generated insulin and glucose (insulin x
 *     glucose / 405), so it can never contradict them;
 *   - alcohol volume is zero unless a drinking category was generated;
 *   - the sparse archetype leaves the optional laboratory fields blank.
 */
export const generateFullSyntheticProfile = ({
	random = Math.random,
	archetype,
} = {}) => {
	const base = generateSyntheticPatient({ random, archetype });
	const values = { ...base.values };
	const sparse = base.omittedFields.includes("GHB_LBXGH");

	const bmi = Number(values.BMX_BMXBMI) || 26;
	const hba1c = Number(values.GHB_LBXGH) || 5.4;
	const glucose = Number(values.GLU_LBXGLU) || 95;
	const insulin = Number(values.INS_LBXIN) || 8;

	const ethnicityOptions = PREVENTION_FIELD_SCHEMA.DEMO_RIDRETH3.options;
	values.DEMO_RIDRETH3 =
		ethnicityOptions[Math.floor(random() * ethnicityOptions.length)];

	const smokingRoll = random();
	values.smoking_status = smokingRoll < 0.55 ? "0" : smokingRoll < 0.85 ? "1" : "2";
	const alcoholRoll = random();
	values.alcohol_status = alcoholRoll < 0.35 ? "0" : alcoholRoll < 0.85 ? "1" : "2";
	values.average_drinks_per_day =
		values.alcohol_status === "0"
			? clampField("average_drinks_per_day", 0)
			: clampField(
					"average_drinks_per_day",
					values.alcohol_status === "1"
						? uniform(random, 0.3, 2)
						: uniform(random, 3, 8),
				);

	if (sparse) {
		[
			"TRIGLY_LBXTR",
			"TRIGLY_LBDLDL",
			"HDL_LBDHDD",
			"TCHOL_LBXTC",
			"HSCRP_LBXHSCRP",
			"CBC_LBXHGB",
			"CBC_LBXPLTSI",
			"BIOPRO_LBXSATSI",
			"BIOPRO_LBXSAPSI",
			"BIOPRO_LBXSCR",
			"weight_loss_1yr_lb",
			"weight_loss_10yr_lb",
			"homa_ir",
		].forEach((field) => {
			values[field] = "";
		});
		return { ...base, values, sparse: true };
	}

	const triglycerides = 60 + (bmi - 20) * 6 + (hba1c - 5) * 22 + uniform(random, -20, 45);
	const hdl = 62 - (bmi - 20) * 0.8 + uniform(random, -8, 12);
	const ldl = 85 + (bmi - 22) * 1.6 + uniform(random, -22, 35);
	values.TRIGLY_LBXTR = clampField("TRIGLY_LBXTR", triglycerides);
	values.HDL_LBDHDD = clampField("HDL_LBDHDD", hdl);
	values.TRIGLY_LBDLDL = clampField("TRIGLY_LBDLDL", ldl);
	values.TCHOL_LBXTC = clampField(
		"TCHOL_LBXTC",
		Number(values.HDL_LBDHDD) +
			Number(values.TRIGLY_LBDLDL) +
			Number(values.TRIGLY_LBXTR) / 5,
	);
	values.HSCRP_LBXHSCRP = clampField(
		"HSCRP_LBXHSCRP",
		Math.max(0.2, (bmi - 19) * 0.22 + uniform(random, -0.4, 1.8)),
	);
	values.CBC_LBXHGB = clampField("CBC_LBXHGB", uniform(random, 12.2, 16.4));
	values.CBC_LBXPLTSI = clampField("CBC_LBXPLTSI", uniform(random, 170, 330));
	values.BIOPRO_LBXSATSI = clampField(
		"BIOPRO_LBXSATSI",
		Math.max(8, 14 + (bmi - 22) * 0.9 + uniform(random, -5, 14)),
	);
	values.BIOPRO_LBXSAPSI = clampField("BIOPRO_LBXSAPSI", uniform(random, 48, 108));
	values.BIOPRO_LBXSCR = clampField("BIOPRO_LBXSCR", uniform(random, 0.6, 1.25));
	values.weight_loss_1yr_lb = clampField(
		"weight_loss_1yr_lb",
		uniform(random, -12, 14),
	);
	values.weight_loss_10yr_lb = clampField(
		"weight_loss_10yr_lb",
		Number(values.weight_loss_1yr_lb) + uniform(random, -14, 26),
	);
	values.homa_ir = clampField("homa_ir", (insulin * glucose) / 405);

	return { ...base, values, sparse: false };
};

export const SYNTHETIC_NOTE = SYNTHETIC_SAFETY_NOTE;
