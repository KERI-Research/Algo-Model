# DiaPan Cycle-Held-Out Temporal Validation

## Purpose

Each NHANES survey cycle is held out in turn, with imputation and model fitting performed only on the remaining cycles. This is an internal-external temporal validation design: it evaluates transportability across time and exposes heterogeneity that a random split can conceal. Clinical prediction guidance recommends evaluating performance in new settings and examining heterogeneity rather than treating one internal validation as definitive ([BMJ evaluation guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC10772854/); [Nieboer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5708595/)).

## Feature variants

- **clinical_only:** excludes `survey_cycle_index` and `hba1c_cycle_age_sex_z` to test transportability using patient-level clinical and metabolic features.
- **with_cycle_proxies:** includes the complete feature set to quantify whether cycle-derived context helps or signals temporal dependence.

## Pooled out-of-cycle results

| Variant | Model | Rows | Positives | AUROC (cycle-bootstrap 95% CI) | AUPRC (cycle-bootstrap 95% CI) | AUPRC lift | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clinical_only | logistic_regression_balanced | 6,473 | 41 | 0.601 (0.558-0.673) | 0.035 (0.009-0.069) | 5.49x | 0.1996 |
| clinical_only | random_forest_balanced | 6,473 | 41 | 0.629 (0.529-0.702) | 0.018 (0.007-0.074) | 2.91x | 0.0073 |
| clinical_only | diapan_hist_gradient_boosting_v1 | 6,473 | 41 | 0.635 (0.582-0.685) | 0.012 (0.009-0.025) | 1.88x | 0.0159 |
| clinical_only | diapan_xgboost_v1 | 6,473 | 41 | 0.643 (0.561-0.703) | 0.013 (0.008-0.029) | 2.03x | 0.0123 |
| with_cycle_proxies | logistic_regression_balanced | 6,473 | 41 | 0.593 (0.555-0.658) | 0.022 (0.009-0.064) | 3.46x | 0.1925 |
| with_cycle_proxies | random_forest_balanced | 6,473 | 41 | 0.585 (0.524-0.633) | 0.022 (0.007-0.080) | 3.41x | 0.0072 |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 6,473 | 41 | 0.609 (0.578-0.667) | 0.011 (0.009-0.021) | 1.81x | 0.0148 |
| with_cycle_proxies | diapan_xgboost_v1 | 6,473 | 41 | 0.630 (0.562-0.685) | 0.014 (0.008-0.039) | 2.17x | 0.0107 |

## Per-cycle results

| Variant | Model | Held-out cycle | Test rows | Positives | AUROC | AUPRC | Lift |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| clinical_only | logistic_regression_balanced | 1999-2000 | 409 | 2 | 0.581 | 0.041 | 8.48x |
| clinical_only | random_forest_balanced | 1999-2000 | 409 | 2 | 0.602 | 0.009 | 1.91x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 1999-2000 | 409 | 2 | 0.415 | 0.006 | 1.31x |
| clinical_only | diapan_xgboost_v1 | 1999-2000 | 409 | 2 | 0.475 | 0.007 | 1.41x |
| clinical_only | logistic_regression_balanced | 2001-2002 | 412 | 4 | 0.706 | 0.025 | 2.58x |
| clinical_only | random_forest_balanced | 2001-2002 | 412 | 4 | 0.613 | 0.016 | 1.62x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2001-2002 | 412 | 4 | 0.591 | 0.017 | 1.72x |
| clinical_only | diapan_xgboost_v1 | 2001-2002 | 412 | 4 | 0.562 | 0.013 | 1.35x |
| clinical_only | logistic_regression_balanced | 2003-2004 | 478 | 2 | 0.756 | 0.012 | 2.98x |
| clinical_only | random_forest_balanced | 2003-2004 | 478 | 2 | 0.502 | 0.011 | 2.70x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2003-2004 | 478 | 2 | 0.658 | 0.015 | 3.50x |
| clinical_only | diapan_xgboost_v1 | 2003-2004 | 478 | 2 | 0.589 | 0.013 | 3.13x |
| clinical_only | logistic_regression_balanced | 2005-2006 | 452 | 3 | 0.615 | 0.074 | 11.14x |
| clinical_only | random_forest_balanced | 2005-2006 | 452 | 3 | 0.478 | 0.009 | 1.41x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2005-2006 | 452 | 3 | 0.468 | 0.018 | 2.68x |
| clinical_only | diapan_xgboost_v1 | 2005-2006 | 452 | 3 | 0.510 | 0.014 | 2.06x |
| clinical_only | logistic_regression_balanced | 2007-2008 | 680 | 3 | 0.587 | 0.009 | 2.11x |
| clinical_only | random_forest_balanced | 2007-2008 | 680 | 3 | 0.698 | 0.044 | 9.92x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2007-2008 | 680 | 3 | 0.765 | 0.087 | 19.68x |
| clinical_only | diapan_xgboost_v1 | 2007-2008 | 680 | 3 | 0.753 | 0.050 | 11.33x |
| clinical_only | logistic_regression_balanced | 2009-2010 | 683 | 6 | 0.710 | 0.023 | 2.64x |
| clinical_only | random_forest_balanced | 2009-2010 | 683 | 6 | 0.684 | 0.025 | 2.79x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2009-2010 | 683 | 6 | 0.628 | 0.015 | 1.66x |
| clinical_only | diapan_xgboost_v1 | 2009-2010 | 683 | 6 | 0.656 | 0.019 | 2.15x |
| clinical_only | logistic_regression_balanced | 2011-2012 | 645 | 4 | 0.507 | 0.014 | 2.31x |
| clinical_only | random_forest_balanced | 2011-2012 | 645 | 4 | 0.446 | 0.010 | 1.61x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2011-2012 | 645 | 4 | 0.767 | 0.022 | 3.57x |
| clinical_only | diapan_xgboost_v1 | 2011-2012 | 645 | 4 | 0.762 | 0.019 | 3.01x |
| clinical_only | logistic_regression_balanced | 2013-2014 | 681 | 5 | 0.509 | 0.009 | 1.29x |
| clinical_only | random_forest_balanced | 2013-2014 | 681 | 5 | 0.660 | 0.212 | 28.86x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2013-2014 | 681 | 5 | 0.568 | 0.109 | 14.81x |
| clinical_only | diapan_xgboost_v1 | 2013-2014 | 681 | 5 | 0.477 | 0.073 | 9.95x |
| clinical_only | logistic_regression_balanced | 2015-2016 | 774 | 1 | 0.960 | 0.031 | 24.19x |
| clinical_only | random_forest_balanced | 2015-2016 | 774 | 1 | 0.662 | 0.004 | 2.95x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2015-2016 | 774 | 1 | 0.950 | 0.025 | 19.35x |
| clinical_only | diapan_xgboost_v1 | 2015-2016 | 774 | 1 | 0.834 | 0.008 | 6.00x |
| clinical_only | logistic_regression_balanced | 2017-March2020 | 1,259 | 11 | 0.572 | 0.105 | 12.04x |
| clinical_only | random_forest_balanced | 2017-March2020 | 1,259 | 11 | 0.768 | 0.024 | 2.73x |
| clinical_only | diapan_hist_gradient_boosting_v1 | 2017-March2020 | 1,259 | 11 | 0.682 | 0.019 | 2.22x |
| clinical_only | diapan_xgboost_v1 | 2017-March2020 | 1,259 | 11 | 0.736 | 0.028 | 3.19x |
| with_cycle_proxies | logistic_regression_balanced | 1999-2000 | 409 | 2 | 0.580 | 0.045 | 9.13x |
| with_cycle_proxies | random_forest_balanced | 1999-2000 | 409 | 2 | 0.478 | 0.008 | 1.70x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 1999-2000 | 409 | 2 | 0.550 | 0.009 | 1.86x |
| with_cycle_proxies | diapan_xgboost_v1 | 1999-2000 | 409 | 2 | 0.430 | 0.007 | 1.35x |
| with_cycle_proxies | logistic_regression_balanced | 2001-2002 | 412 | 4 | 0.705 | 0.023 | 2.40x |
| with_cycle_proxies | random_forest_balanced | 2001-2002 | 412 | 4 | 0.606 | 0.016 | 1.67x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2001-2002 | 412 | 4 | 0.568 | 0.017 | 1.79x |
| with_cycle_proxies | diapan_xgboost_v1 | 2001-2002 | 412 | 4 | 0.672 | 0.017 | 1.80x |
| with_cycle_proxies | logistic_regression_balanced | 2003-2004 | 478 | 2 | 0.776 | 0.014 | 3.26x |
| with_cycle_proxies | random_forest_balanced | 2003-2004 | 478 | 2 | 0.505 | 0.011 | 2.57x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2003-2004 | 478 | 2 | 0.690 | 0.024 | 5.85x |
| with_cycle_proxies | diapan_xgboost_v1 | 2003-2004 | 478 | 2 | 0.569 | 0.012 | 2.88x |
| with_cycle_proxies | logistic_regression_balanced | 2005-2006 | 452 | 3 | 0.634 | 0.075 | 11.30x |
| with_cycle_proxies | random_forest_balanced | 2005-2006 | 452 | 3 | 0.462 | 0.010 | 1.49x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2005-2006 | 452 | 3 | 0.522 | 0.018 | 2.65x |
| with_cycle_proxies | diapan_xgboost_v1 | 2005-2006 | 452 | 3 | 0.543 | 0.022 | 3.29x |
| with_cycle_proxies | logistic_regression_balanced | 2007-2008 | 680 | 3 | 0.564 | 0.009 | 2.07x |
| with_cycle_proxies | random_forest_balanced | 2007-2008 | 680 | 3 | 0.662 | 0.063 | 14.33x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2007-2008 | 680 | 3 | 0.756 | 0.094 | 21.41x |
| with_cycle_proxies | diapan_xgboost_v1 | 2007-2008 | 680 | 3 | 0.718 | 0.078 | 17.78x |
| with_cycle_proxies | logistic_regression_balanced | 2009-2010 | 683 | 6 | 0.718 | 0.021 | 2.42x |
| with_cycle_proxies | random_forest_balanced | 2009-2010 | 683 | 6 | 0.660 | 0.024 | 2.76x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2009-2010 | 683 | 6 | 0.615 | 0.015 | 1.68x |
| with_cycle_proxies | diapan_xgboost_v1 | 2009-2010 | 683 | 6 | 0.669 | 0.017 | 1.92x |
| with_cycle_proxies | logistic_regression_balanced | 2011-2012 | 645 | 4 | 0.519 | 0.012 | 2.00x |
| with_cycle_proxies | random_forest_balanced | 2011-2012 | 645 | 4 | 0.510 | 0.009 | 1.51x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2011-2012 | 645 | 4 | 0.718 | 0.020 | 3.16x |
| with_cycle_proxies | diapan_xgboost_v1 | 2011-2012 | 645 | 4 | 0.760 | 0.017 | 2.75x |
| with_cycle_proxies | logistic_regression_balanced | 2013-2014 | 681 | 5 | 0.517 | 0.010 | 1.34x |
| with_cycle_proxies | random_forest_balanced | 2013-2014 | 681 | 5 | 0.632 | 0.209 | 28.52x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2013-2014 | 681 | 5 | 0.542 | 0.043 | 5.80x |
| with_cycle_proxies | diapan_xgboost_v1 | 2013-2014 | 681 | 5 | 0.456 | 0.106 | 14.46x |
| with_cycle_proxies | logistic_regression_balanced | 2015-2016 | 774 | 1 | 0.956 | 0.029 | 22.11x |
| with_cycle_proxies | random_forest_balanced | 2015-2016 | 774 | 1 | 0.787 | 0.006 | 4.66x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2015-2016 | 774 | 1 | 0.946 | 0.023 | 18.00x |
| with_cycle_proxies | diapan_xgboost_v1 | 2015-2016 | 774 | 1 | 0.884 | 0.011 | 8.51x |
| with_cycle_proxies | logistic_regression_balanced | 2017-March2020 | 1,259 | 11 | 0.579 | 0.103 | 11.84x |
| with_cycle_proxies | random_forest_balanced | 2017-March2020 | 1,259 | 11 | 0.683 | 0.018 | 2.07x |
| with_cycle_proxies | diapan_hist_gradient_boosting_v1 | 2017-March2020 | 1,259 | 11 | 0.669 | 0.019 | 2.21x |
| with_cycle_proxies | diapan_xgboost_v1 | 2017-March2020 | 1,259 | 11 | 0.729 | 0.029 | 3.35x |

## Interpretation rules

- Prefer the clinical-only variant if performance is similar; it has less dependence on survey-era context.
- Large variation between held-out cycles indicates dataset shift and argues against deploying a frozen model without recalibration or refitting. Temporal studies show model stability can change even when average discrimination appears acceptable ([Lopes et al. 2023](https://doi.org/10.1016/j.heliyon.2023.e17139)).
- These results remain exploratory because pancreatic-cancer events are rare and the outcome is self-reported prevalent disease, not future incident cancer.

## Reproduction

```bash
cd api
python validate_cycle_holdout.py \
  --dataset ../data/nhanes_multicycle.csv \
  --target PancreaticCancer \
  --cohort-filter diabetics_only
```
