# MetaboGuard Cycle-Held-Out Validation: Invalidated Benchmark

## Status

The previously reported cycle-held-out pancreatic-risk benchmark is
**invalidated**.

The source dataset incorrectly mapped NHANES MCQ230 code 39 to pancreatic
cancer. Official NHANES documentation defines code 29 as pancreatic cancer and
code 39 as Other.

Therefore, all former AUROC, AUPRC, lift, Brier and feature-importance results
from that benchmark must not be quoted or used.

## Corrected-event audit

| Target | Positive cases |
|---|---:|
| Correct pancreatic cancer across pooled NHANES | 19 |
| Correct pancreatic cancer among diabetics | 7 |
| Usable diabetic positives after cleaning | 6 |
| Pancreatic cancer 0-3 years after diabetes | 2 |

Cycle-held-out model evaluation is not statistically meaningful with this event
count. Several test cycles contain no positive cases and the remaining
cycle-specific estimates would be dominated by one or two individuals.

## Safeguards added

- Corrected dataset version: `nhanes_multicycle_v2.csv`
- Deprecated v1 datasets removed from API discovery
- Minimum 20 positive and 20 negative usable cases before fitting
- SHA-256 dataset fingerprint stored in artifacts
- Automatic invalidation when dataset contents change
- Public artifact export blocked when event count is insufficient

## When to rerun

Rerun leave-one-cycle/site-out validation only after obtaining a cohort with
enough incident pancreatic-cancer cases. The primary target should be pancreatic
cancer diagnosed within three years of diabetes onset, with exact diagnosis
dates and no post-diagnosis features.
