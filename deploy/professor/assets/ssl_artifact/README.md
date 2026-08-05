---
license: cc-by-4.0
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - tabular
  - self-supervised-learning
  - anomaly-detection
  - cancer-prevention-research
  - diabetes-prevention-research
  - nhanes
---

# MetaboGuard-SSL v1

## Model summary

MetaboGuard-SSL is a self-supervised denoising autoencoder trained on adult
NHANES metabolic, demographic, behavioural, CBC and biochemistry variables.
Disease labels were not used to train the representation.

The artifact produces:

- a 16-dimensional patient representation;
- a metabolic deviation score;
- a percentile relative to the NHANES training reference;
- the five features contributing most to reconstruction deviation.

## Intended use

Research into clinician-reviewed early-warning and prevention workflows.
Scores may help identify records whose metabolic profile merits additional
review or monitoring.

## Prohibited interpretation

This model:

- does not diagnose cancer or diabetes;
- does not estimate a validated future disease probability;
- does not recommend treatment;
- must not reassure patients that disease is absent;
- must not be used without clinician review;
- must not present the Type 1 proxy as an autoimmune diagnosis.

Current NHANES data are repeated cross-sectional observations, not
patient-level longitudinal trajectories. The supported output is therefore
**cross-sectional metabolic deviation only**.

## Training

- Dataset: corrected `nhanes_multicycle_v2.csv`
- Adult rows available: 89,472
- Unlabelled training sample: 50,000
- Input fields: 25
- Transformed dimensions: 55
- Latent dimensions: 16
- Objective: masked/noisy feature reconstruction
- Training epochs: 25
- Validation reconstruction loss: 0.0565

Diagnosis-dependent fields used to define diabetes subtype, including diagnosis
age and current insulin-use status, were excluded from the prevention encoder.

## Post-hoc association checks

Frozen embeddings were evaluated using cross-validated logistic heads. These
checks describe cross-sectional association, not future development.

| Check | Positives | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|
| Any-cancer prevalence | 5,584 | 0.699 | 0.169 | 0.223 |
| Type 2 diabetes proxy | 7,067 | 0.923 | 0.675 | 0.104 |
| Type 1 proxy, research only | 177 | 0.909 | 0.135 | 0.083 |

The Type 1 endpoint is an unvalidated proxy based on early diagnosis and
insulin use. It is not suitable for patient-facing output. A valid Type 1
warning model requires islet autoantibodies, C-peptide and approved
genetic/family-history inputs.

## Scoring

```bash
python score_prevention_record.py \
  --artifact ./nhanes_multicycle_v2 \
  --input sample_input.json
```

A higher percentile means the profile is more unusual relative to the training
reference. It does not mean a higher validated cancer or diabetes probability.

## Adaptive prediction horizons

Future risk heads may use 1-, 3- or 5-year horizons only when the supplied
longitudinal dataset contains enough events and non-events at that horizon.
The current artifact exposes no risk horizon.

## Limitations

- Cross-sectional self-reported disease labels
- No future diagnosis endpoint
- Missingness varies by NHANES cycle and laboratory subsample
- No CA19-9 or diabetes autoantibodies
- C-peptide available only in early cycles
- No genetics in the patient-facing model
- NHANES population may not transport to other countries or care settings

## Files

- `autoencoder_weights.npz`: NumPy-compatible network weights
- `preprocessor.joblib`: fitted tabular preprocessing
- `metadata.json`: configuration, score reference and association checks
- `score_prevention_record.py`: command-line scoring helper
- `self_supervised.py`: model and scoring implementation
- `sample_input.json`: example record
