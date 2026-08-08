/**
 * Aggregate, privacy-preserving synthetic profiles for the prevention allowlist.
 *
 * The fitted asset contains only training-partition quantiles, category frequencies and
 * a regularized rank-correlation factor. Generation never selects or reproduces a source
 * row. Disease outcomes and post-diagnosis fields are not present in the asset or output.
 */

import SYNTHETIC_PROFILE_MODEL from "./synthetic_profile_model.json";
import {
	REQUIRED_PROBE_FIELDS,
	SYNTHETIC_FIELD_SCHEMA,
	SYNTHETIC_SAFETY_NOTE,
	formatForField,
} from "./synthetic_patient";

export { REQUIRED_PROBE_FIELDS, SYNTHETIC_FIELD_SCHEMA };

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
	TRIGLY_LBXTR: {
		kind: "number",
		label: "Triglycerides",
		unit: "mg/dL",
		min: 20,
		max: 900,
		precision: 0,
	},
	TRIGLY_LBDLDL: {
		kind: "number",
		label: "LDL cholesterol",
		unit: "mg/dL",
		min: 20,
		max: 300,
		precision: 0,
	},
	HDL_LBDHDD: {
		kind: "number",
		label: "HDL cholesterol",
		unit: "mg/dL",
		min: 15,
		max: 120,
		precision: 0,
	},
	TCHOL_LBXTC: {
		kind: "number",
		label: "Total cholesterol",
		unit: "mg/dL",
		min: 80,
		max: 400,
		precision: 0,
	},
	HSCRP_LBXHSCRP: {
		kind: "number",
		label: "hs-CRP",
		unit: "mg/L",
		min: 0.1,
		max: 60,
		precision: 2,
	},
	CBC_LBXHGB: {
		kind: "number",
		label: "Haemoglobin",
		unit: "g/dL",
		min: 7,
		max: 19,
		precision: 1,
	},
	CBC_LBXPLTSI: {
		kind: "number",
		label: "Platelet count",
		unit: "1000 cells/uL",
		min: 60,
		max: 600,
		precision: 0,
	},
	BIOPRO_LBXSATSI: {
		kind: "number",
		label: "ALT",
		unit: "U/L",
		min: 5,
		max: 200,
		precision: 0,
	},
	BIOPRO_LBXSAPSI: {
		kind: "number",
		label: "Alkaline phosphatase",
		unit: "U/L",
		min: 20,
		max: 300,
		precision: 0,
	},
	BIOPRO_LBXSCR: {
		kind: "number",
		label: "Creatinine",
		unit: "mg/dL",
		min: 0.3,
		max: 8,
		precision: 2,
	},
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
	average_drinks_per_day: {
		kind: "number",
		label: "Average drinks per day",
		unit: "drinks",
		min: 0,
		max: 20,
		precision: 1,
	},
	weight_loss_1yr_lb: {
		kind: "number",
		label: "Weight change, past year",
		unit: "lb lost",
		min: -60,
		max: 80,
		precision: 0,
	},
	weight_loss_10yr_lb: {
		kind: "number",
		label: "Weight change, past 10 years",
		unit: "lb lost",
		min: -80,
		max: 120,
		precision: 0,
	},
	homa_ir: {
		kind: "number",
		label: "HOMA-IR (derived)",
		unit: "index",
		min: 0,
		max: 60,
		precision: 2,
	},
};

const SPARSE_PROFILE = {
	id: "sparse_but_valid",
	label: "Sparse aggregate profile",
	description:
		"A fabricated reference-pattern adult with optional measurements omitted to exercise missing-data handling.",
	reportedDiabetes: "0",
	omitOptional: true,
};

export const SYNTHETIC_PROFILE_OPTIONS = [
	...Object.entries(SYNTHETIC_PROFILE_MODEL.profiles).map(
		([id, profile]) => ({
			id,
			label: profile.label,
			description: profile.description,
			reportedDiabetes: profile.reported_diabetes,
			omitOptional: false,
			trainingRows: profile.training_rows,
		}),
	),
	SPARSE_PROFILE,
];

const clampField = (field, value) => {
	if (SYNTHETIC_FIELD_SCHEMA[field]) {
		return formatForField(field, value);
	}
	const schema = PREVENTION_FIELD_SCHEMA[field];
	if (!schema || schema.kind !== "number") {
		return String(value);
	}
	const bounded = Math.min(schema.max, Math.max(schema.min, value));
	return bounded.toFixed(schema.precision);
};

const normalCdf = (value) => {
	const sign = value < 0 ? -1 : 1;
	const magnitude = Math.abs(value) / Math.sqrt(2);
	const t = 1 / (1 + 0.3275911 * magnitude);
	const polynomial =
		((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
			0.284496736) *
			t +
			0.254829592) *
		t;
	const erf = sign * (1 - polynomial * Math.exp(-(magnitude ** 2)));
	return Math.min(1 - 1e-9, Math.max(1e-9, 0.5 * (1 + erf)));
};

const standardNormals = (count, random) => {
	const values = [];
	while (values.length < count) {
		const radius = Math.sqrt(
			-2 * Math.log(Math.max(random(), 1e-12)),
		);
		const angle = 2 * Math.PI * random();
		values.push(radius * Math.cos(angle));
		if (values.length < count) {
			values.push(radius * Math.sin(angle));
		}
	}
	return values;
};

const correlatedUniforms = (factor, random) => {
	const independent = standardNormals(factor.length, random);
	return factor.map((row, rowIndex) => {
		let value = 0;
		for (let column = 0; column <= rowIndex; column += 1) {
			value += row[column] * independent[column];
		}
		return normalCdf(value);
	});
};

export const interpolateQuantile = (quantiles, probability) => {
	if (!Array.isArray(quantiles) || quantiles.length === 0) {
		throw new Error("Synthetic quantile table is empty.");
	}
	const grid = SYNTHETIC_PROFILE_MODEL.quantile_probabilities;
	if (probability <= grid[0]) return quantiles[0];
	if (probability >= grid[grid.length - 1])
		return quantiles[quantiles.length - 1];
	const scaled =
		((probability - grid[0]) / (grid[grid.length - 1] - grid[0])) *
		(quantiles.length - 1);
	const lower = Math.floor(scaled);
	const fraction = scaled - lower;
	return (
		quantiles[lower] +
		fraction * (quantiles[lower + 1] - quantiles[lower])
	);
};

const sampleCategory = (distribution, random) => {
	let threshold = random();
	for (let index = 0; index < distribution.values.length; index += 1) {
		threshold -= distribution.probabilities[index];
		if (threshold <= 0) return distribution.values[index];
	}
	return distribution.values[distribution.values.length - 1];
};

const chooseProfileId = (random, requested) => {
	if (requested === SPARSE_PROFILE.id) return requested;
	if (requested) {
		if (!SYNTHETIC_PROFILE_MODEL.profiles[requested]) {
			throw new Error(
				`Unknown synthetic profile: ${requested}`,
			);
		}
		return requested;
	}
	const profiles = Object.entries(SYNTHETIC_PROFILE_MODEL.profiles);
	const total = profiles.reduce(
		(sum, [, profile]) => sum + profile.sampling_weight,
		0,
	);
	let threshold = random() * total;
	for (const [id, profile] of profiles) {
		threshold -= profile.sampling_weight;
		if (threshold <= 0) return id;
	}
	return profiles[0][0];
};

const profileMetadata = (profileId) => {
	if (profileId === SPARSE_PROFILE.id) return SPARSE_PROFILE;
	const profile = SYNTHETIC_PROFILE_MODEL.profiles[profileId];
	return {
		id: profileId,
		label: profile.label,
		description: profile.description,
		reportedDiabetes: profile.reported_diabetes,
		omitOptional: false,
		trainingRows: profile.training_rows,
	};
};

const blankOptionalFields = (values) => {
	const required = new Set(REQUIRED_PROBE_FIELDS);
	Object.keys(values).forEach((field) => {
		if (!required.has(field)) values[field] = "";
	});
};

/** Generate a novel profile from aggregate NHANES training-partition statistics. */
export const generateFullSyntheticProfile = ({
	random = Math.random,
	archetype,
} = {}) => {
	const requestedId = chooseProfileId(random, archetype);
	const sampledId =
		requestedId === SPARSE_PROFILE.id
			? "reference_range"
			: requestedId;
	const profile = SYNTHETIC_PROFILE_MODEL.profiles[sampledId];
	const uniforms = correlatedUniforms(
		profile.correlation_cholesky,
		random,
	);
	const values = {};

	SYNTHETIC_PROFILE_MODEL.continuous_fields.forEach((field, index) => {
		values[field] = clampField(
			field,
			interpolateQuantile(
				profile.quantiles[field],
				uniforms[index],
			),
		);
	});
	Object.entries(profile.categories).forEach(([field, distribution]) => {
		values[field] = sampleCategory(distribution, random);
	});

	values.Diabetes = profile.reported_diabetes;
	values.DIQ_DID040 = "";
	if (values.Diabetes === "1") {
		const durations =
			profile.context_quantiles.diabetes_duration_years;
		const duration = Math.max(
			1,
			interpolateQuantile(durations, random()),
		);
		const age = Number(values.DEMO_RIDAGEYR);
		values.DIQ_DID040 = formatForField(
			"DIQ_DID040",
			Math.max(1, Math.min(age - 1, age - duration)),
		);
	}

	const insulin = Number(values.INS_LBXIN);
	const glucose = Number(values.GLU_LBXGLU);
	values.homa_ir = clampField("homa_ir", (insulin * glucose) / 405);
	values.TCHOL_LBXTC = clampField(
		"TCHOL_LBXTC",
		Number(values.HDL_LBDHDD) +
			Number(values.TRIGLY_LBDLDL) +
			Number(values.TRIGLY_LBXTR) / 5,
	);
	if (values.alcohol_status === "0") {
		values.average_drinks_per_day = clampField(
			"average_drinks_per_day",
			0,
		);
	} else if (values.alcohol_status === "1") {
		values.average_drinks_per_day = clampField(
			"average_drinks_per_day",
			Math.min(
				2,
				Math.max(
					0.1,
					Number(values.average_drinks_per_day),
				),
			),
		);
	} else {
		values.average_drinks_per_day = clampField(
			"average_drinks_per_day",
			Math.max(2.1, Number(values.average_drinks_per_day)),
		);
	}

	if (requestedId === SPARSE_PROFILE.id) blankOptionalFields(values);
	const populatedFields = Object.keys(values).filter(
		(field) => values[field] !== "",
	);
	const omittedFields = Object.keys(values).filter(
		(field) => values[field] === "",
	);
	return {
		archetype: profileMetadata(requestedId),
		values,
		populatedFields,
		omittedFields,
		sparse: requestedId === SPARSE_PROFILE.id,
		generation: {
			method: "aggregate_gaussian_copula",
			schemaVersion: SYNTHETIC_PROFILE_MODEL.schema_version,
			sourceDataset: SYNTHETIC_PROFILE_MODEL.source.dataset,
			sourcePartition:
				SYNTHETIC_PROFILE_MODEL.source.partition,
			trainingRows: profile.training_rows,
			containsSourceRows:
				SYNTHETIC_PROFILE_MODEL.privacy
					.contains_source_rows,
		},
	};
};

export const SYNTHETIC_NOTE =
	`${SYNTHETIC_SAFETY_NOTE} Generated from aggregate training-partition ` +
	"quantiles and rank correlations; no source row is selected or bundled.";
