import {
	PREVENTION_FIELD_SCHEMA,
	SYNTHETIC_FIELD_SCHEMA,
} from "./synthetic_prevention.js";

export const PROBE_FIELD_GROUPS = [
	{
		id: "demographics",
		legend: "Demographics and reported status",
		fields: [
			"DEMO_RIDAGEYR",
			"DEMO_RIAGENDR",
			"DEMO_RIDRETH3",
			"Diabetes",
			"DIQ_DID040",
		],
	},
	{
		id: "anthropometry",
		legend: "Anthropometry",
		fields: [
			"BMX_BMXBMI",
			"BMX_BMXWAIST",
			"weight_loss_1yr_lb",
			"weight_loss_10yr_lb",
		],
	},
	{
		id: "glycaemia",
		legend: "Glycaemia and insulin",
		fields: [
			"GHB_LBXGH",
			"GLU_LBXGLU",
			"INS_LBXIN",
			"CPEP_LBXCPSI",
			"homa_ir",
		],
	},
	{
		id: "lipids",
		legend: "Lipids and inflammation",
		fields: [
			"TRIGLY_LBXTR",
			"TRIGLY_LBDLDL",
			"HDL_LBDHDD",
			"TCHOL_LBXTC",
			"HSCRP_LBXHSCRP",
		],
	},
	{
		id: "haematology",
		legend: "Haematology and biochemistry",
		fields: [
			"CBC_LBXHGB",
			"CBC_LBXPLTSI",
			"BIOPRO_LBXSATSI",
			"BIOPRO_LBXSAPSI",
			"BIOPRO_LBXSCR",
		],
	},
	{
		id: "lifestyle",
		legend: "Lifestyle",
		fields: [
			"smoking_status",
			"alcohol_status",
			"average_drinks_per_day",
		],
	},
];

const LABELS = {
	DEMO_RIDAGEYR: "Age",
	DEMO_RIAGENDR: "Sex",
	Diabetes: "Reported diabetes",
	DIQ_DID040: "Age at diabetes onset",
	BMX_BMXBMI: "BMI",
	BMX_BMXWAIST: "Waist circumference",
	GHB_LBXGH: "HbA1c",
	GLU_LBXGLU: "Fasting glucose",
	INS_LBXIN: "Insulin",
	CPEP_LBXCPSI: "C-peptide",
};

export const PROBE_FIELDS = PROBE_FIELD_GROUPS.flatMap((group) => group.fields);
export const CONTEXT_FIELDS = ["Diabetes", "DIQ_DID040"];
export const MODEL_FIELDS = PROBE_FIELDS.filter(
	(field) => !CONTEXT_FIELDS.includes(field),
);

export const schemaFor = (field) =>
	PREVENTION_FIELD_SCHEMA[field] ||
	SYNTHETIC_FIELD_SCHEMA[field] || { kind: "number" };

export const labelFor = (field) => {
	const schema = schemaFor(field);
	return LABELS[field] || schema.label || field;
};

export const domainFor = (field) =>
	PROBE_FIELD_GROUPS.find((group) => group.fields.includes(field))
		?.legend || "Other measurement";

export const emptyProbeForm = () =>
	PROBE_FIELDS.reduce(
		(accumulator, field) => ({ ...accumulator, [field]: "" }),
		{},
	);

export const formatProbeValue = (field, value) => {
	if (
		value === null ||
		value === undefined ||
		String(value).trim() === ""
	) {
		return "Not supplied";
	}
	const schema = schemaFor(field);
	if (schema.kind === "categorical") {
		return schema.optionLabels?.[value] || String(value);
	}
	return schema.unit ? `${value} ${schema.unit}` : String(value);
};
