/**
 * Synthetic patient generator for the Biomarker Probe form.
 *
 * Purpose: fill every visible probe input with a plausible, internally coherent value so
 * the interface can be exercised without typing ten fields by hand, and without ever
 * using real patient data.
 *
 * Scope and safety:
 *   - Values are FABRICATED for interface testing. They are not real measurements, not a
 *     patient, and must never be used for clinical interpretation.
 *   - Only fields that already exist in the probe form are generated. No new fields are
 *     invented, and no outcome label (Cancer, PancreaticCancer, diabetes_subtype, any
 *     `tcga_*` column) is ever produced: those are denylisted model inputs.
 *   - `Diabetes` is generated because the existing form exposes it as a required *input*
 *     to the cross-sectional association model (its target is Cancer). It is a reported
 *     condition, not a prediction and not the model's outcome.
 *   - Archetypes describe metabolic patterns only. None of them is a disease label, a
 *     diagnosis or a risk stratum.
 *
 * Ranges, units, precision and categorical encodings are taken from the existing form
 * schema (`INITIAL_PATIENT_FORM`), the API field sets in `api/biomarker.py`
 * (`REQUIRED_FIELDS` / `OPTIONAL_HIGH_IMPACT_FIELDS`), `docs/COLUMN_DICTIONARY.md`, and the
 * plausibility windows in `api/data_reliability.py` (`PLAUSIBLE_RANGES`). Generation ranges
 * are deliberately narrower than those windows so generated values are always plausible.
 */

/** Fields the API needs before it will score a record (api/biomarker.py REQUIRED_FIELDS). */
export const REQUIRED_PROBE_FIELDS = [
	"Diabetes",
	"DEMO_RIDAGEYR",
	"DEMO_RIAGENDR",
	"BMX_BMXBMI",
];

/**
 * Field schema. `min`/`max`/`step` mirror what a valid numeric input accepts;
 * `precision` is the number of decimal places the generator emits.
 */
export const SYNTHETIC_FIELD_SCHEMA = {
	Diabetes: {
		kind: "categorical",
		options: ["0", "1"],
		optionLabels: { 0: "No", 1: "Yes" },
		description:
			"Reported diabetes status. Model input, not an outcome label.",
	},
	DEMO_RIDAGEYR: {
		kind: "number",
		unit: "years",
		min: 18,
		max: 85,
		step: 1,
		precision: 0,
	},
	DEMO_RIAGENDR: {
		kind: "categorical",
		options: ["1", "2"],
		optionLabels: { 1: "Male", 2: "Female" },
		description: "NHANES sex code.",
	},
	BMX_BMXBMI: {
		kind: "number",
		unit: "kg/m2",
		min: 16,
		max: 55,
		step: 0.1,
		precision: 1,
	},
	BMX_BMXWAIST: {
		kind: "number",
		unit: "cm",
		min: 60,
		max: 170,
		step: 0.1,
		precision: 1,
	},
	DIQ_DID040: {
		kind: "number",
		unit: "years (age at diabetes onset)",
		min: 1,
		max: 85,
		step: 1,
		precision: 0,
		appliesWhen: "Diabetes === '1'",
	},
	GHB_LBXGH: {
		kind: "number",
		unit: "% HbA1c",
		min: 3.5,
		max: 15,
		step: 0.1,
		precision: 1,
	},
	GLU_LBXGLU: {
		kind: "number",
		unit: "mg/dL glucose",
		min: 50,
		max: 400,
		step: 1,
		precision: 0,
	},
	INS_LBXIN: {
		kind: "number",
		unit: "uU/mL insulin",
		min: 1,
		max: 150,
		step: 0.1,
		precision: 1,
	},
	CPEP_LBXCPSI: {
		kind: "number",
		unit: "nmol/L C-peptide",
		min: 0,
		max: 20,
		step: 0.1,
		precision: 1,
	},
};

/**
 * Metabolic pattern archetypes. These are measurement patterns for interface testing.
 * They are NOT disease labels, diagnoses or risk categories.
 */
export const SYNTHETIC_ARCHETYPES = [
	{
		id: "reference_range",
		label: "Reference-range profile",
		description:
			"All exposed measurements inside commonly reported reference intervals.",
		weight: 3,
		reportedDiabetes: "0",
		bmi: [19, 27],
		hba1c: [4.8, 5.6],
		insulinFactor: [0.25, 0.5],
		omitOptional: false,
	},
	{
		id: "metabolic_deviation",
		label: "Metabolic-deviation profile",
		description:
			"Higher adiposity with dysglycaemic and insulin-resistant measurements. No diagnosis is implied.",
		weight: 3,
		reportedDiabetes: "0",
		bmi: [28, 42],
		hba1c: [5.7, 7.4],
		insulinFactor: [0.6, 1.4],
		omitOptional: false,
	},
	{
		id: "reported_diabetes_metabolic",
		label: "Reported-diabetes metabolic profile",
		description:
			"Reported diabetes with an onset age and correspondingly higher glycaemic measurements.",
		weight: 2,
		reportedDiabetes: "1",
		bmi: [26, 45],
		hba1c: [6.5, 11],
		insulinFactor: [0.4, 1.6],
		omitOptional: false,
	},
	{
		id: "sparse_but_valid",
		label: "Sparse-but-submittable profile",
		description:
			"Required fields only, optional laboratory fields left blank to exercise the missing-field path the API already supports.",
		weight: 1,
		reportedDiabetes: "0",
		bmi: [20, 38],
		hba1c: [5.0, 6.4],
		insulinFactor: [0.3, 0.9],
		omitOptional: true,
	},
];

/** Deterministic PRNG (mulberry32) so tests can pin a seed. */
export const createRandomSource = (seed = 1) => {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
};

const roundToPrecision = (value, precision) => {
	const factor = 10 ** precision;
	return Math.round(value * factor) / factor;
};

/** Clamp to the field's min/max, snap to its step, and format at its precision. */
export const formatForField = (field, value) => {
	const schema = SYNTHETIC_FIELD_SCHEMA[field];
	if (!schema || schema.kind !== "number") {
		return String(value);
	}
	const clamped = Math.min(schema.max, Math.max(schema.min, value));
	const snapped =
		schema.step > 0
			? schema.min +
				Math.round(
					(clamped - schema.min) / schema.step,
				) *
					schema.step
			: clamped;
	const bounded = Math.min(schema.max, Math.max(schema.min, snapped));
	return roundToPrecision(bounded, schema.precision).toFixed(
		schema.precision,
	);
};

const uniform = (random, low, high) => low + random() * (high - low);

const pickArchetype = (random, archetypeId) => {
	if (archetypeId) {
		const requested = SYNTHETIC_ARCHETYPES.find(
			(item) => item.id === archetypeId,
		);
		if (!requested) {
			throw new Error(
				`Unknown synthetic archetype: ${archetypeId}`,
			);
		}
		return requested;
	}
	const total = SYNTHETIC_ARCHETYPES.reduce(
		(sum, item) => sum + item.weight,
		0,
	);
	let threshold = random() * total;
	for (const candidate of SYNTHETIC_ARCHETYPES) {
		threshold -= candidate.weight;
		if (threshold <= 0) {
			return candidate;
		}
	}
	return SYNTHETIC_ARCHETYPES[0];
};

/**
 * Generate one coherent synthetic profile.
 *
 * Coherence rules (all one-directional, none of them a clinical claim):
 *   - waist circumference is derived from BMI and sex, then jittered;
 *   - fasting glucose is derived from HbA1c using the usual monotone relationship,
 *     then jittered, so the two never contradict each other;
 *   - insulin scales with BMI and the archetype's insulin factor;
 *   - age at diabetes onset is only produced when diabetes is reported, and is always
 *     at least one year below the generated age;
 *   - every value is clamped to its field min/max, snapped to its step and formatted at
 *     its precision.
 *
 * @param {object} options
 * @param {() => number} options.random  Injectable uniform [0,1) source.
 * @param {string} [options.archetype]   Force a specific archetype id.
 * @returns {{archetype: object, values: Record<string,string>, populatedFields: string[], omittedFields: string[]}}
 */
export const generateSyntheticPatient = ({
	random = Math.random,
	archetype: archetypeId,
} = {}) => {
	const archetype = pickArchetype(random, archetypeId);

	const reportedDiabetes = archetype.reportedDiabetes;
	const sexCode = random() < 0.5 ? "1" : "2";
	const minimumAge = reportedDiabetes === "1" ? 30 : 18;
	const ageValue = Math.round(uniform(random, minimumAge, 84));

	const bmiValue = uniform(random, archetype.bmi[0], archetype.bmi[1]);
	// Waist grows with BMI; the intercept differs by sex. Jitter keeps profiles varied.
	const waistValue =
		bmiValue * 2.1 +
		(sexCode === "1" ? 24 : 18) +
		uniform(random, -4, 4) +
		(ageValue - 50) * 0.06;

	const hba1cValue = uniform(
		random,
		archetype.hba1c[0],
		archetype.hba1c[1],
	);
	// Monotone HbA1c -> fasting glucose relationship with noise, never negative.
	const glucoseValue = Math.max(
		55,
		28.7 * hba1cValue - 46.7 + uniform(random, -8, 12),
	);
	const insulinValue = Math.max(
		1.2,
		(bmiValue - 12) *
			uniform(
				random,
				archetype.insulinFactor[0],
				archetype.insulinFactor[1],
			),
	);
	const cPeptideValue = Math.max(
		0.2,
		0.18 * insulinValue + uniform(random, 0.1, 1.6),
	);

	const values = {
		Diabetes: reportedDiabetes,
		DEMO_RIDAGEYR: formatForField("DEMO_RIDAGEYR", ageValue),
		DEMO_RIAGENDR: sexCode,
		BMX_BMXBMI: formatForField("BMX_BMXBMI", bmiValue),
		BMX_BMXWAIST: formatForField("BMX_BMXWAIST", waistValue),
		DIQ_DID040: "",
		GHB_LBXGH: formatForField("GHB_LBXGH", hba1cValue),
		GLU_LBXGLU: formatForField("GLU_LBXGLU", glucoseValue),
		INS_LBXIN: formatForField("INS_LBXIN", insulinValue),
		CPEP_LBXCPSI: formatForField("CPEP_LBXCPSI", cPeptideValue),
	};

	if (reportedDiabetes === "1") {
		const generatedAge = Number(values.DEMO_RIDAGEYR);
		const durationYears = Math.round(
			uniform(
				random,
				1,
				Math.max(2, Math.min(30, generatedAge - 20)),
			),
		);
		const onsetAge = Math.max(1, generatedAge - durationYears);
		values.DIQ_DID040 = formatForField("DIQ_DID040", onsetAge);
	}

	if (archetype.omitOptional) {
		// Blank only the optional fields. Required fields stay populated so the probe
		// remains submittable; this exercises the API's existing missing-field handling.
		[
			"BMX_BMXWAIST",
			"DIQ_DID040",
			"GHB_LBXGH",
			"GLU_LBXGLU",
			"INS_LBXIN",
			"CPEP_LBXCPSI",
		].forEach((field) => {
			values[field] = "";
		});
	}

	const populatedFields = Object.keys(values).filter(
		(field) => values[field] !== "",
	);
	return {
		archetype: {
			id: archetype.id,
			label: archetype.label,
			description: archetype.description,
		},
		values,
		populatedFields,
		omittedFields: Object.keys(values).filter(
			(field) => values[field] === "",
		),
	};
};

/** Validate a generated profile against the schema. Returns a list of problems. */
export const validateSyntheticProfile = (values) => {
	const problems = [];
	Object.entries(values).forEach(([field, value]) => {
		const schema = SYNTHETIC_FIELD_SCHEMA[field];
		if (!schema) {
			problems.push(`${field}: not part of the probe schema`);
			return;
		}
		if (value === "") {
			if (REQUIRED_PROBE_FIELDS.includes(field)) {
				problems.push(
					`${field}: required field is blank`,
				);
			}
			return;
		}
		if (schema.kind === "categorical") {
			if (!schema.options.includes(value)) {
				problems.push(
					`${field}: '${value}' is not an allowed category`,
				);
			}
			return;
		}
		const numeric = Number(value);
		if (Number.isNaN(numeric)) {
			problems.push(`${field}: '${value}' is not numeric`);
			return;
		}
		if (numeric < schema.min || numeric > schema.max) {
			problems.push(
				`${field}: ${numeric} outside [${schema.min}, ${schema.max}]`,
			);
		}
		if (numeric < 0) {
			problems.push(`${field}: negative value`);
		}
		const decimals = (String(value).split(".")[1] || "").length;
		if (decimals > schema.precision) {
			problems.push(
				`${field}: ${decimals} decimals exceeds precision ${schema.precision}`,
			);
		}
	});
	REQUIRED_PROBE_FIELDS.forEach((field) => {
		if (!(field in values)) {
			problems.push(
				`${field}: missing from the generated profile`,
			);
		}
	});
	return problems;
};

export const SYNTHETIC_SAFETY_NOTE =
	"Synthetic research data: these values are fabricated for interface testing only. " +
	"They are not a real patient, not a measurement, and must not be used for clinical " +
	"interpretation.";
