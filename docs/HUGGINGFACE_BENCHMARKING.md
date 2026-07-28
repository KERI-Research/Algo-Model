# MetaboGuard Hugging Face Benchmarking Guide

## Artifact status

`MetaboGuard-SSL v1` may be shared for **self-supervised representation and
anomaly-scoring research**.

It must not be published as a cancer/diabetes diagnostic or future-risk model.
The former supervised DiaPan-XGB artifact remains invalidated.

## Artifact outputs

- 16-dimensional patient embedding
- metabolic-deviation score
- reference percentile
- five highest reconstruction-deviation features
- dataset capability declaration
- cross-sectional association report

## Appropriate benchmark tasks

| Task | Metric |
|---|---|
| Masked-feature reconstruction | Validation MSE |
| Representation stability | Embedding similarity across repeated corruption |
| Anomaly ranking | AUROC/AUPRC only against independently defined anomaly labels |
| Clustering | Silhouette/stability plus clinical interpretability |
| Linear probe | Cross-validated AUROC/AUPRC, explicitly cross-sectional |
| Missingness robustness | Score/embedding change after controlled masking |

## Inappropriate comparisons

Do not compare:

- MetaboGuard deviation percentiles against diagnostic probabilities;
- cross-sectional cancer association against future incident-cancer models;
- Type 1 proxy performance against autoantibody-confirmed Type 1 outcomes;
- TCGA prognosis performance against preventive early-warning performance.

## Candidate Hugging Face comparators

| Candidate | Use |
|---|---|
| AutoGluon TabPFN Mix | Frozen-representation or linear-probe comparator |
| Prior-Labs TabPFN | Generic tabular representation/classification comparator |
| Variational autoencoder | Probabilistic reconstruction baseline |
| Masked tabular transformer | Self-supervised representation comparator |
| Isolation Forest | Non-deep anomaly baseline |
| PCA | Linear representation baseline |

Every comparator must use identical folds, features and preprocessing
boundaries.

## Future disease-risk benchmark

A cancer/diabetes development benchmark becomes valid only when a dataset has:

- patient-level longitudinal observations;
- an index date before disease;
- incident outcome dates;
- adequate events at the chosen horizon;
- no post-diagnosis inputs.

Supported horizons are selected from 1, 3 and 5 years based on event
availability. Each horizon requires at least 50 events and 50 eligible
non-events for initial evaluation.

## Release checklist

- Non-diagnostic model card
- Dataset and cohort description
- Feature allowlist and denylist
- Type 1 research-only warning
- No genetics unless approved
- Capability declaration
- Training and scoring examples
- Reproducible preprocessing
- External benchmark contract
- CC BY 4.0 licence
