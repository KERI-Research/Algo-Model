import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import HowItWorks, {
	CAPABILITIES,
	COMPARISON_ROWS,
	LONGITUDINAL_LIMIT_STATEMENT,
	PIPELINE_STEPS,
	READING_OUTPUTS,
} from "./HowItWorks.jsx";

/** The exact sentence the deployment is required to show. */
const REQUIRED_STATEMENT =
	"The current model is not trained on longitudinal data. It cannot estimate a patient's probability of developing cancer over time, predict which cancer they will develop, or provide a diagnosis.";

/**
 * Wording that would turn a descriptive deviation score into a disease claim.
 * Matched case-insensitively against the whole rendered page.
 */
const PROHIBITED_PATTERNS = [
	/\bcancer risk score\b/i,
	/\brisk of (developing )?cancer is\b/i,
	/\bprobability of (developing )?cancer (is|of)\b/i,
	/\bpredicts? (that )?(the )?(patient|they) will develop\b/i,
	/\blikelihood of cancer\b/i,
	/\bdiagnoses\b/i,
	/\bdiagnostic tool\b/i,
	/\bearly detection of cancer in this deployment\b/i,
	/\bhigh(er)? chance of cancer\b/i,
	/\bcancer probability\b/i,
	/\bwill develop cancer\b/i,
	/\bscreening test\b/i,
	/\bcoming soon\b/i,
	/\bin a future release\b/i,
	/\broadmap item\b/i,
	/\bplanned for\b/i,
];

const renderPage = (onNavigate = vi.fn()) => {
	render(<HowItWorks onNavigate={onNavigate} />);
	return onNavigate;
};

describe("How the AI works", () => {
	it("states the longitudinal limitation verbatim, near the top", () => {
		renderPage();
		const statement = screen.getByTestId("text-longitudinal-limit");
		expect(statement).toHaveTextContent(REQUIRED_STATEMENT);
		expect(LONGITUDINAL_LIMIT_STATEMENT).toBe(REQUIRED_STATEMENT);
		// It must be the first notice on the page.
		const notices = document.querySelectorAll(".notice");
		expect(notices[0]).toContainElement(statement);
	});

	it("renders all eight pipeline steps in order", () => {
		renderPage();
		const flow = screen.getByTestId("pipeline-flow");
		const steps = within(flow).getAllByRole("listitem");
		expect(steps).toHaveLength(8);
		expect(PIPELINE_STEPS.map((step) => step.id)).toEqual([
			"inputs",
			"preprocessing",
			"training",
			"representation",
			"reference",
			"clustering",
			"evidence",
			"review",
		]);
		steps.forEach((step, index) => {
			expect(step).toHaveTextContent(String(index + 1));
			expect(step).toHaveTextContent(PIPELINE_STEPS[index].label);
		});
	});

	it("covers each required pipeline concept", () => {
		renderPage();
		const flow = screen.getByTestId("pipeline-flow").textContent.toLowerCase();
		for (const concept of [
			"allowlisted",
			"training split",
			"was missing",
			"noised and partly masked",
			"no cancer or diabetes label",
			"sixteen numbers",
			"reconstruction error",
			"percentile",
			"bootstrap resampling",
			"negative controls",
			"evidence grade",
			"clinician",
			"abstention",
		]) {
			expect(flow).toContain(concept);
		}
	});

	it("explains the deviation score as unusualness, not cancer probability", () => {
		renderPage();
		const deviation = screen.getByTestId("plain-deviation");
		expect(deviation).toHaveTextContent(/unusual compared with the reference cohort/i);
		expect(deviation).toHaveTextContent(/does not mean a high chance of cancer/i);
	});

	it("explains percentile, contributions and the clustering abstention in plain language", () => {
		renderPage();
		expect(screen.getByTestId("plain-percentile")).toHaveTextContent(
			/position in a distribution, not a probability/i,
		);
		expect(screen.getByTestId("plain-contributions")).toHaveTextContent(
			/not causes/i,
		);
		const abstention = screen.getByTestId("plain-abstention");
		expect(abstention).toHaveTextContent(/no_stable_clusters/i);
		expect(abstention).toHaveTextContent(/survey cycle/i);
		expect(READING_OUTPUTS).toHaveLength(4);
	});

	it("renders the current vs future longitudinal comparison", () => {
		renderPage();
		const table = screen.getByTestId("comparison-table");
		expect(within(table).getAllByRole("row")).toHaveLength(COMPARISON_ROWS.length + 1);
		const text = table.textContent.toLowerCase();
		for (const required of [
			"cross-sectional nhanes",
			"single visit",
			"representation and deviation score only",
			"post-hoc",
			"no cancer horizon",
			"repeated pre-diagnosis measurements",
			"dated incident cancer outcomes",
			"cancer site and stage",
			"patient-level temporal splits",
			"1, 3 and 5 year horizons",
			"calibration",
			"net-benefit",
			"independent cohorts",
		]) {
			expect(text).toContain(required);
		}
	});

	it("frames the future column as requirements, not a schedule", () => {
		renderPage();
		expect(screen.getByText(/is not a roadmap and nothing in it is scheduled/i)).toBeInTheDocument();
		expect(
			screen.getByText(/Nothing moves between them automatically/i),
		).toBeInTheDocument();
	});

	it("explains TCGA as post-diagnosis context only", () => {
		renderPage();
		const section = screen.getByLabelledBy
			? screen.getByLabelledBy("tcga-heading")
			: document.querySelector('[aria-labelledby="tcga-heading"]');
		const text = section.textContent.toLowerCase();
		expect(text).toContain("post-diagnosis biological context only");
		expect(text).toContain("never model inputs");
		expect(text).toContain("never used for prevention scoring");
		expect(text).toContain("cannot supply a risk horizon");
	});

	it("shows the three capability states with no fourth state", () => {
		renderPage();
		const table = screen.getByTestId("capability-table");
		const states = new Set(
			Array.from(table.querySelectorAll(".badge")).map((node) => node.textContent.trim()),
		);
		expect(states).toEqual(
			new Set(["Available now", "Research only", "Unavailable until longitudinal validation"]),
		);
		expect(within(table).getAllByRole("row")).toHaveLength(CAPABILITIES.length + 1);
	});

	it("marks every risk-over-time capability as unavailable", () => {
		renderPage();
		const table = screen.getByTestId("capability-table");
		const rows = within(table).getAllByRole("row").slice(1);
		const blocked = rows.filter((row) =>
			/probability of developing cancer|which cancer type|risk horizons|calibrated risk|diagnosis, screening or triage/i.test(
				row.textContent,
			),
		);
		expect(blocked).toHaveLength(5);
		blocked.forEach((row) => {
			expect(row).toHaveTextContent("Unavailable until longitudinal validation");
		});
	});

	it("contains no prohibited wording or disease-probability claim", () => {
		renderPage();
		const text = document.body.textContent;

		// Only affirmative sentences may make a claim, so negated sentences
		// ("It does not mean a high chance of cancer") are excluded before the
		// prohibited-wording scan. Each excluded sentence is still required to
		// carry an explicit negation.
		const sentences = text
			.split(/(?<=[.:])\s+|(?=[A-Z][a-z]+ [a-z])/)
			.filter(Boolean);
		const negation = /\b(not|never|cannot|can't|no|without|refus|prohibit|abstain|unavailable|out of scope)\b/i;
		const affirmative = sentences.filter((sentence) => !negation.test(sentence)).join(" ");

		for (const pattern of PROHIBITED_PATTERNS) {
			expect(affirmative).not.toMatch(pattern);
		}

		// Risk-claim phrases may appear only inside a negation. Capability-table
		// rows are excluded here because a row label is not a sentence: those rows
		// carry their status in the Status column, asserted in the test above.
		const capabilityText = screen.getByTestId("capability-table").textContent;
		const narrativeSentences = sentences.filter(
			(sentence) => !capabilityText.includes(sentence.trim()),
		);
		for (const phrase of [
			"probability of developing cancer",
			"chance of cancer",
			"which cancer they will develop",
			"provide a diagnosis",
		]) {
			const hits = narrativeSentences.filter((sentence) =>
				sentence.toLowerCase().includes(phrase.toLowerCase()),
			);
			expect(hits.length).toBeGreaterThan(0);
			hits.forEach((sentence) => expect(sentence).toMatch(negation));
		}

		// No scheduling or "coming soon" language anywhere, negated or not.
		for (const pattern of [/coming soon/i, /in a future release/i, /planned for/i, /roadmap item/i]) {
			expect(text).not.toMatch(pattern);
		}
	});

	it("offers navigation back to overview and on to the evidence catalogue", async () => {
		const onNavigate = renderPage();
		const user = userEvent.setup();
		await user.click(screen.getByTestId("button-back-overview"));
		expect(onNavigate).toHaveBeenCalledWith("overview");
		await user.click(screen.getByTestId("button-to-evidence"));
		expect(onNavigate).toHaveBeenCalledWith("evidence");
	});

	it("uses semantic headings and an ordered list for the flow", () => {
		renderPage();
		expect(screen.getByTestId("pipeline-flow").tagName).toBe("OL");
		const headings = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
		expect(headings).toEqual([
			"The pipeline, step by step",
			"Reading the outputs in plain language",
			"Current model vs a future longitudinal model",
			"Where TCGA fits, and where it does not",
			"Capability status",
		]);
	});
});
