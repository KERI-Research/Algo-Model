import { describe, expect, it } from "vitest";
import { createRandomSource } from "./synthetic_patient.js";
import SYNTHETIC_PROFILE_MODEL from "./synthetic_profile_model.json";
import {
	generateFullSyntheticProfile,
	PREVENTION_FIELD_SCHEMA,
	SYNTHETIC_PROFILE_OPTIONS,
} from "./synthetic_prevention.js";

const PROHIBITED = [
	"Cancer",
	"PancreaticCancer",
	"NODM_PancreaticCancer",
	"diabetes_subtype",
	"new_onset_diabetes",
];

const median = (values) => {
	const ordered = [...values].sort((left, right) => left - right);
	return ordered[Math.floor(ordered.length / 2)];
};

const correlation = (rows, leftField, rightField) => {
	const left = rows.map((row) => Number(row[leftField]));
	const right = rows.map((row) => Number(row[rightField]));
	const leftMean =
		left.reduce((sum, value) => sum + value, 0) / left.length;
	const rightMean =
		right.reduce((sum, value) => sum + value, 0) / right.length;
	const covariance = left.reduce(
		(sum, value, index) =>
			sum + (value - leftMean) * (right[index] - rightMean),
		0,
	);
	const leftSpread = Math.sqrt(
		left.reduce((sum, value) => sum + (value - leftMean) ** 2, 0),
	);
	const rightSpread = Math.sqrt(
		right.reduce((sum, value) => sum + (value - rightMean) ** 2, 0),
	);
	return covariance / (leftSpread * rightSpread);
};

describe("generateFullSyntheticProfile", () => {
	it("is deterministic for a pinned seed", () => {
		const first = generateFullSyntheticProfile({
			random: createRandomSource(7),
		});
		const second = generateFullSyntheticProfile({
			random: createRandomSource(7),
		});
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
			Object.entries(PREVENTION_FIELD_SCHEMA).forEach(
				([field, schema]) => {
					const value = profile.values[field];
					if (
						value === "" ||
						value === undefined
					) {
						return;
					}
					if (schema.kind === "categorical") {
						expect(
							schema.options,
						).toContain(value);
						return;
					}
					const numeric = Number(value);
					expect(Number.isNaN(numeric)).toBe(
						false,
					);
					expect(numeric).toBeGreaterThanOrEqual(
						schema.min,
					);
					expect(numeric).toBeLessThanOrEqual(
						schema.max,
					);
				},
			);
		}
	});

	it("derives HOMA-IR from the generated insulin and glucose", () => {
		const profile = generateFullSyntheticProfile({
			random: createRandomSource(3),
			archetype: "metabolic_deviation",
		});
		const expected =
			(Number(profile.values.INS_LBXIN) *
				Number(profile.values.GLU_LBXGLU)) /
			405;
		expect(Number(profile.values.homa_ir)).toBeCloseTo(
			Math.min(
				60,
				Math.max(0, Math.round(expected * 100) / 100),
			),
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
				expect(
					Number(
						profile.values
							.average_drinks_per_day,
					),
				).toBe(0);
			}
		}
	});

	it("reports aggregate-only provenance and no bundled source rows", () => {
		const profile = generateFullSyntheticProfile({
			random: createRandomSource(17),
		});
		expect(profile.generation).toMatchObject({
			method: "aggregate_gaussian_copula",
			containsSourceRows: false,
			sourcePartition: "participant-grouped training split",
		});
		expect(SYNTHETIC_PROFILE_MODEL.privacy).toMatchObject({
			contains_source_rows: false,
			contains_identifiers: false,
		});
		expect(JSON.stringify(SYNTHETIC_PROFILE_MODEL)).not.toMatch(
			/\/Volumes\/|\/Users\/|\/home\//,
		);
	});

	it("exposes every fitted profile plus an explicit sparse option", () => {
		expect(
			SYNTHETIC_PROFILE_OPTIONS.map((profile) => profile.id),
		).toEqual([
			"reference_range",
			"metabolic_deviation",
			"reported_diabetes_metabolic",
			"sparse_but_valid",
		]);
	});

	it("tracks fitted marginal medians for every complete profile", () => {
		const fields = [
			"DEMO_RIDAGEYR",
			"BMX_BMXBMI",
			"BMX_BMXWAIST",
			"GHB_LBXGH",
			"GLU_LBXGLU",
			"TRIGLY_LBXTR",
			"HDL_LBDHDD",
		];
		Object.entries(SYNTHETIC_PROFILE_MODEL.profiles).forEach(
			([profileId, fitted]) => {
				const random = createRandomSource(20260808);
				const generated = Array.from(
					{ length: 800 },
					() =>
						generateFullSyntheticProfile({
							random,
							archetype: profileId,
						}),
				).map((profile) => profile.values);
				fields.forEach((field) => {
					const quantiles =
						fitted.quantiles[field];
					const targetMedian = quantiles[49];
					const interquartileRange =
						quantiles[74] - quantiles[24];
					const generatedMedian = median(
						generated.map((row) =>
							Number(row[field]),
						),
					);
					expect(
						Math.abs(
							generatedMedian -
								targetMedian,
						),
						`${profileId}:${field}`,
					).toBeLessThanOrEqual(
						Math.max(
							1,
							interquartileRange *
								0.35,
						),
					);
				});
			},
		);
	});

	it("preserves key metabolic dependence directions", () => {
		const random = createRandomSource(20260808);
		const generated = Array.from({ length: 1200 }, () =>
			generateFullSyntheticProfile({
				random,
				archetype: "metabolic_deviation",
			}),
		).map((profile) => profile.values);
		expect(
			correlation(generated, "BMX_BMXBMI", "BMX_BMXWAIST"),
		).toBeGreaterThan(0.45);
		expect(
			correlation(generated, "GHB_LBXGH", "GLU_LBXGLU"),
		).toBeGreaterThan(0.2);
		expect(
			correlation(generated, "INS_LBXIN", "CPEP_LBXCPSI"),
		).toBeGreaterThan(0.2);
		expect(
			correlation(generated, "TRIGLY_LBXTR", "HDL_LBDHDD"),
		).toBeLessThan(0);
	});
});
