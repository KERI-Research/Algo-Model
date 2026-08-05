# Future-risk evaluation guide (simulation only)

Implemented in `api/future_risk_models.py` and driven by `api/run_future_risk_pipeline.py`.

> **SIMULATION ONLY.** Every metric described here is currently computed on synthetic data.
> Synthetic metrics verify that the evaluation code, masks and gates behave correctly. They do
> not transfer to real patients, they establish no calibration, and they support no
> early-detection claim. Ordinary-incidence and declared-enrichment cohorts are reported
> separately and never pooled.

## What is evaluated, and on whom

All metrics are computed **per outcome and per horizon**, on **eligible patients only**: the
event occurred inside the horizon, or the patient was followed event-free and alive for the whole
horizon. Patients censored before the horizon are excluded by mask, never counted as negatives.
The same masks are used for the training loss, the calibration fit and every reported metric.

Splits are identical for every model: train (fit), validation (calibration), test (report), plus
a temporal holdout of index years ≥ 2016 reported alongside.

## Discrimination

- **AUROC** and **average precision (AP)**. AP is reported because the positive class is small
  at short horizons, where AUROC alone flatters a model.
- **Harrell C-index**, cause-specific: computed on `time_to_event` with the target cause as the
  event indicator, so competing deaths do not count as target events. Implemented in-repo
  (seeded pair subsampling) because `lifelines` is not installed; the seed is recorded.

## Calibration

- **Brier score** on eligible patients.
- **Calibration intercept** (mean observed − mean predicted, the probability-scale
  calibration-in-the-large) and **calibration slope** from a logistic fit on the predicted logit.
- **Calibration curve** as decile bins with `n`, mean predicted and observed rate per bin.
- Both **raw** and **calibrated** probabilities are evaluated and stored side by side. Calibration
  is fitted on the validation split only: isotonic when there are ≥50 rows and ≥10 events,
  otherwise Platt (logistic) — the fallback is recorded per model. Model selection is
  calibration-first, because external validations of real risk models routinely find substantial
  over-prediction on new populations
  ([BMJ 2012, DOI 10.1136/bmj.e5900](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445426/)).

## Clinical-usefulness scaffolding

- **Decision curve** at thresholds 0.02 / 0.05 / 0.10 / 0.20: flagged fraction, true positives,
  false positives, net benefit of the model and net benefit of "treat all".
- **False-alert burden**: false positives per 100 screened at each threshold — the number a
  clinician actually feels.

This is scaffolding, not a utility claim. On synthetic data, net benefit is a code-path check.

## Uncertainty

**Bootstrap 95 % confidence intervals** for AUROC and AP (200 resamples in a full run, 40 in
smoke mode), seeded, resamples that collapse to one class are skipped and the effective round
count is reported.

## Subgroups

AUROC and Brier by age band (<45, 45–59, 60–74, 75+), sex, missingness-burden tertile and
visit-density tertile. Any subgroup with fewer than 10 events or a single class is reported as
`suppressed` with its counts rather than given a misleading metric.

## Negative controls (mandatory)

1. **Shuffled outcome labels.** Labels are permuted on the test split with a fixed seed and the
   fitted model is re-scored. Expected AUROC ≈ 0.5; anything materially higher indicates leakage.
2. **Time-reversed sequences.** Visit order is reversed for the temporal model. Discrimination
   must **degrade**; identical performance means the model ignores temporal order.

The time-reversal control is enforced, not merely reported: a temporal model is admissible only
if reversing visit order costs at least 0.02 AUROC. Otherwise it is recorded as
`experimental_rejected_time_reversal_control` and excluded from selection, and a calibrated
baseline is chosen instead. The first GRU version — masked mean pooling over GRU outputs —
failed this control (reversed AUROC matched forward AUROC), which is why the architecture reads
the last valid hidden state instead. Temporal superiority is never manufactured.

## Abstention

A horizon that fails the 50-event / 50-non-event gate produces **no metric**. `results.json`
records the failing counts and the reason, the model card lists the horizon as `gate FAILED (not
trained)`, and the API returns an explicit abstention rather than a probability. Abstention is a
result, not a bug.

## What is persisted

Inside every artifact directory:

| File | Contents |
| --- | --- |
| `future_risk_models.joblib` | Fitted models, preprocessing pipelines, calibrators, hazard model, temporal weights and normaliser. |
| `predictions_simulation_only.csv` | Per patient / outcome / horizon / model / split: label, raw probability, calibrated probability. |
| `feature_columns.json` | Exact model input columns, in order. |
| `longitudinal_schema.json` | Machine-readable schema for the inputs. |
| `splits.json`, `split_manifest.json` | Split IDs, sizes, overlaps, per-split fingerprints. |
| `dataset_validation_report.json` | Gate counts and timeline checks for the cohort. |
| `results.json` | All metrics, subgroups, negative controls, admissibility and selection. |
| `metadata.json` | Capability state, `clinical_use: prohibited`, versions, fingerprints, selection. |
| `MODEL_CARD.md` | Intended use, prohibited use, data, gates, selection, controls, limitations. |
| `reload_parity_report.json` | Reload/scoring parity: max absolute difference vs stored predictions. |

## Reproduction

```bash
cd api
python run_future_risk_pipeline.py --prepared <prepared_dir> --output-dir <artifact_root>
python run_future_risk_pipeline.py --verify-artifact <artifact_root>/artifact --prepared <prepared_dir>
python -m unittest test_future_risk_pipeline -v
```

## Limitations that no amount of evaluation fixes

1. Synthetic data cannot establish real-world calibration, discrimination or clinical utility.
2. Enriched cohorts have deliberately inflated event rates; absolute probabilities from them are
   not population-calibrated.
3. There is no external validation, no clinician review and no prospective evaluation.
4. The clinical future-risk endpoint stays HTTP 409 regardless of any artifact trained here.