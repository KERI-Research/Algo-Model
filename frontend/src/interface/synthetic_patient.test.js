import {
	REQUIRED_PROBE_FIELDS,
	SYNTHETIC_ARCHETYPES,
	SYNTHETIC_FIELD_SCHEMA,
	SYNTHETIC_SAFETY_NOTE,
	createRandomSource,
	formatForField,
	generateSyntheticPatient,
	validateSyntheticProfile,
} from "./synthetic_patient";

const PROBE_FIELDS = [
	"Diabetes",
	"DEMO_RIDAGEYR",
	"DEMO_RIAGENDR",
	"BMX_BMXBMI",
	"BMX_BMXWAIST",
	"DIQ_DID040",
	"GHB_LBXGH",
	"GLU_LBXGLU",
	"INS_LBXIN",
];

const OUTCOME_LABELS = [
	"Cancer",
	"PancreaticCancer",
	"NODM_PancreaticCancer",
	"diabetes_subtype",
	"new_onset_diabetes",
	"tcga_stage_ordinal",
	"tcga_followup_days",
];

describe("synthetic patient generator", () => {
	test("is deterministic for a fixed seed", () => {
		const first = generateSyntheticPatient({ random: createRandomSource(42) });
		const second = generateSyntheticPatient({ random: createRandomSource(42) });
		expect(second.values).toEqual(first.values);
		expect(second.archetype.id).toBe(first.archetype.id);
	});

	test("produces a different profile on each successive draw", () => {
		const random = createRandomSource(7);
		const profiles = Array.from({ length: 8 }, () =>
			JSON.stringify(generateSyntheticPatient({ random }).values),
		);
		expect(new Set(profiles).size).toBeGreaterThan(1);
	});

	test("covers exactly the probe fields and no others", () => {
		const { values } = generateSyntheticPatient({
			random: createRandomSource(3),
		});
		expect(Object.keys(values).sort()).toEqual([...PROBE_FIELDS].sort());
	});

	test("never emits an outcome label or a post-diagnosis field", () => {
		const random = createRandomSource(11);
		for (let index = 0; index < 25; index += 1) {
			const { values } = generateSyntheticPatient({ random });
			OUTCOME_LABELS.forEach((label) => {
				expect(values).not.toHaveProperty(label);
			});
		}
	});

	test("every value satisfies its field constraints across many draws", () => {
		const random = createRandomSource(99);
		for (let index = 0; index < 200; index += 1) {
			const { values } = generateSyntheticPatient({ random });
			expect(validateSyntheticProfile(values)).toEqual([]);
		}
	});

	test("required fields are always populated so the probe stays submittable", () => {
		const random = createRandomSource(5);
		for (let index = 0; index < 60; index += 1) {
			const { values } = generateSyntheticPatient({ random });
			REQUIRED_PROBE_FIELDS.forEach((field) => {
				expect(values[field]).not.toBe("");
			});
		}
	});

	test("categorical values map to the options their inputs expose", () => {
		const random = createRandomSource(17);
		for (let index = 0; index < 50; index += 1) {
			const { values } = generateSyntheticPatient({ random });
			expect(SYNTHETIC_FIELD_SCHEMA.Diabetes.options).toContain(values.Diabetes);
			expect(SYNTHETIC_FIELD_SCHEMA.DEMO_RIAGENDR.options).toContain(
				values.DEMO_RIAGENDR,
			);
			// The form's select values are strings: "0"/"1" and "1"/"2".
			expect(typeof values.Diabetes).toBe("string");
			expect(typeof values.DEMO_RIAGENDR).toBe("string");
		}
	});

	test("numeric values respect step and precision", () => {
		const random = createRandomSource(23);
		for (let index = 0; index < 100; index += 1) {
			const { values } = generateSyntheticPatient({ random });
			Object.entries(values).forEach(([field, value]) => {
				const schema = SYNTHETIC_FIELD_SCHEMA[field];
				if (schema.kind !== "number" || value === "") {
					return;
				}
				const decimals = (value.split(".")[1] || "").length;
				expect(decimals).toBeLessThanOrEqual(schema.precision);
				const steps = (Number(value) - schema.min) / schema.step;
				expect(Math.abs(steps - Math.round(steps))).toBeLessThan(1e-6);
			});
		}
	});

	test("waist tracks BMI rather than being independent noise", () => {
		const random = createRandomSource(31);
		const pairs = [];
		while (pairs.length < 40) {
			const { values } = generateSyntheticPatient({ random });
			if (values.BMX_BMXWAIST !== "") {
				pairs.push([Number(values.BMX_BMXBMI), Number(values.BMX_BMXWAIST)]);
			}
		}
		const meanBmi = pairs.reduce((sum, [bmi]) => sum + bmi, 0) / pairs.length;
		const meanWaist =
			pairs.reduce((sum, [, waist]) => sum + waist, 0) / pairs.length;
		const covariance =
			pairs.reduce(
				(sum, [bmi, waist]) => sum + (bmi - meanBmi) * (waist - meanWaist),
				0,
			) / pairs.length;
		expect(covariance).toBeGreaterThan(0);
	});

	test("glucose tracks HbA1c rather than contradicting it", () => {
		const random = createRandomSource(37);
		const pairs = [];
		while (pairs.length < 40) {
			const { values } = generateSyntheticPatient({ random });
			if (values.GHB_LBXGH !== "" && values.GLU_LBXGLU !== "") {
				pairs.push([Number(values.GHB_LBXGH), Number(values.GLU_LBXGLU)]);
			}
		}
		const meanHba1c = pairs.reduce((sum, [a1c]) => sum + a1c, 0) / pairs.length;
		const meanGlucose =
			pairs.reduce((sum, [, glucose]) => sum + glucose, 0) / pairs.length;
		const covariance =
			pairs.reduce(
				(sum, [a1c, glucose]) =>
					sum + (a1c - meanHba1c) * (glucose - meanGlucose),
				0,
			) / pairs.length;
		expect(covariance).toBeGreaterThan(0);
	});

	test("diabetes onset age is only present with reported diabetes and stays below age", () => {
		const random = createRandomSource(53);
		for (let index = 0; index < 120; index += 1) {
			const { values, archetype } = generateSyntheticPatient({ random });
			if (values.Diabetes === "1" && !archetype.id.includes("sparse")) {
				expect(values.DIQ_DID040).not.toBe("");
				expect(Number(values.DIQ_DID040)).toBeLessThan(
					Number(values.DEMO_RIDAGEYR),
				);
				expect(Number(values.DIQ_DID040)).toBeGreaterThanOrEqual(1);
			}
			if (values.Diabetes === "0") {
				expect(values.DIQ_DID040).toBe("");
			}
		}
	});

	test("each archetype can be requested explicitly and behaves as documented", () => {
		SYNTHETIC_ARCHETYPES.forEach((archetype) => {
			const { values, archetype: used } = generateSyntheticPatient({
				random: createRandomSource(4),
				archetype: archetype.id,
			});
			expect(used.id).toBe(archetype.id);
			expect(validateSyntheticProfile(values)).toEqual([]);
			expect(values.Diabetes).toBe(archetype.reportedDiabetes);
			if (archetype.omitOptional) {
				expect(values.GHB_LBXGH).toBe("");
				expect(values.GLU_LBXGLU).toBe("");
				expect(values.INS_LBXIN).toBe("");
				REQUIRED_PROBE_FIELDS.forEach((field) => {
					expect(values[field]).not.toBe("");
				});
			} else {
				expect(values.GHB_LBXGH).not.toBe("");
			}
		});
	});

	test("reference-range and deviation archetypes differ in the expected direction", () => {
		const reference = generateSyntheticPatient({
			random: createRandomSource(64),
			archetype: "reference_range",
		}).values;
		const deviation = generateSyntheticPatient({
			random: createRandomSource(64),
			archetype: "metabolic_deviation",
		}).values;
		expect(Number(deviation.BMX_BMXBMI)).toBeGreaterThan(
			Number(reference.BMX_BMXBMI),
		);
		expect(Number(deviation.GHB_LBXGH)).toBeGreaterThan(
			Number(reference.GHB_LBXGH),
		);
	});

	test("archetype metadata avoids disease and risk wording", () => {
		SYNTHETIC_ARCHETYPES.forEach((archetype) => {
			const text = `${archetype.id} ${archetype.label}`.toLowerCase();
			["cancer", "tumour", "carcinoma", "risk", "diagnosis"].forEach((banned) => {
				expect(text).not.toContain(banned);
			});
		});
	});

	test("unknown archetypes fail loudly", () => {
		expect(() =>
			generateSyntheticPatient({
				random: createRandomSource(1),
				archetype: "cancer_positive",
			}),
		).toThrow(/Unknown synthetic archetype/);
	});

	test("formatForField clamps, snaps and formats", () => {
		expect(formatForField("DEMO_RIDAGEYR", 200)).toBe("85");
		expect(formatForField("DEMO_RIDAGEYR", -5)).toBe("18");
		expect(formatForField("BMX_BMXBMI", 27.349)).toBe("27.3");
		expect(formatForField("GLU_LBXGLU", 101.7)).toBe("102");
	});

	test("validateSyntheticProfile rejects out-of-range and unknown fields", () => {
		expect(validateSyntheticProfile({ DEMO_RIDAGEYR: "150" })).toEqual(
			expect.arrayContaining([expect.stringContaining("outside")]),
		);
		expect(validateSyntheticProfile({ Cancer: "1" })).toEqual(
			expect.arrayContaining([
				expect.stringContaining("not part of the probe schema"),
			]),
		);
		expect(validateSyntheticProfile({ Diabetes: "" })).toEqual(
			expect.arrayContaining([expect.stringContaining("required field is blank")]),
		);
		expect(validateSyntheticProfile({ DEMO_RIAGENDR: "3" })).toEqual(
			expect.arrayContaining([
				expect.stringContaining("not an allowed category"),
			]),
		);
	});

	test("safety note states the values are fabricated and non-clinical", () => {
		expect(SYNTHETIC_SAFETY_NOTE.toLowerCase()).toContain("fabricated");
		expect(SYNTHETIC_SAFETY_NOTE.toLowerCase()).toContain(
			"must not be used for clinical",
		);
	});
});