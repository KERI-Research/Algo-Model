import { describe, expect, it } from "vitest";
import { createRandomSource } from "./synthetic_patient.js";
import {
	generateFullSyntheticProfile,
	PREVENTION_FIELD_SCHEMA,
} from "./synthetic_prevention.js";

const PROHIBITED = [
	"Cancer",
	"PancreaticCancer",
	"NODM_PancreaticCancer",
	"diabetes_subtype",
	"new_onset_diabetes",
];

describe("generateFullSyntheticProfile", () => {
	it("is deterministic for a pinned seed", () => {
		const first = generateFullSyntheticProfile({ random: createRandomSource(7) });
		const second = generateFullSyntheticProfile({ random: createRandomSource(7) });
		expect(second.values).toEqual(first.values);
	});

	it("never emits an outcome label or a post-diagnosis field", () => {
		for (let seed = 1; seed <= 40; seed += 1) {
			const profile = generateFullSyntheticProfile({
				random: createRandomSource(seed),
			});
			Object.keys(profile.values).forEach((field) => {
				expect(PROHIBITED).not.toContain(field);
				expect(field.startsWith("tcga_")).toBe(false);
			});
		}
	});

	it("keeps every generated value inside its field range", () => {
		for (let seed = 1; seed <= 40; seed += 1) {
			const profile = generateFullSyntheticProfile({
				random: createRandomSource(seed),
			});
			Object.entries(PREVENTION_FIELD_SCHEMA).forEach(([field, schema]) => {
				const value = profile.values[field];
				if (value === "" || value === undefined) {
					return;
				}
				if (schema.kind === "categorical") {
					expect(schema.options).toContain(value);
					return;
				}
				const numeric = Number(value);
				expect(Number.isNaN(numeric)).toBe(false);
				expect(numeric).toBeGreaterThanOrEqual(schema.min);
				expect(numeric).toBeLessThanOrEqual(schema.max);
			});
		}
	});

	it("derives HOMA-IR from the generated insulin and glucose", () => {
		const profile = generateFullSyntheticProfile({
			random: createRandomSource(3),
			archetype: "metabolic_deviation",
		});
		const expected =
			(Number(profile.values.INS_LBXIN) * Number(profile.values.GLU_LBXGLU)) / 405;
		expect(Number(profile.values.homa_ir)).toBeCloseTo(
			Math.min(60, Math.max(0, Math.round(expected * 100) / 100)),
			2,
		);
	});

	it("reconstructs total cholesterol from its components", () => {
		const profile = generateFullSyntheticProfile({
			random: createRandomSource(11),
			archetype: "reference_range",
		});
		const expected =
			Number(profile.values.HDL_LBDHDD) +
			Number(profile.values.TRIGLY_LBDLDL) +
			Number(profile.values.TRIGLY_LBXTR) / 5;
		expect(Number(profile.values.TCHOL_LBXTC)).toBeCloseTo(
			Math.min(400, Math.max(80, expected)),
			0,
		);
	});

	it("blanks optional laboratory fields for the sparse archetype", () => {
		const profile = generateFullSyntheticProfile({
			random: createRandomSource(5),
			archetype: "sparse_but_valid",
		});
		expect(profile.sparse).toBe(true);
		expect(profile.values.HSCRP_LBXHSCRP).toBe("");
		expect(profile.values.homa_ir).toBe("");
		expect(profile.values.DEMO_RIDAGEYR).not.toBe("");
		expect(profile.values.BMX_BMXBMI).not.toBe("");
	});

	it("reports zero alcohol volume when no drinking category is generated", () => {
		for (let seed = 1; seed <= 30; seed += 1) {
			const profile = generateFullSyntheticProfile({
				random: createRandomSource(seed),
			});
			if (profile.values.alcohol_status === "0") {
				expect(Number(profile.values.average_drinks_per_day)).toBe(0);
			}
		}
	});
});
