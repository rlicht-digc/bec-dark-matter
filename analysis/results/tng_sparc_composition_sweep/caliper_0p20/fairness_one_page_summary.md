# TNG vs SPARC Fairness Sweep Summary

## Inputs
- Source CSV: `analysis/results/tng_sparc_composition_sweep/caliper_0p20/fairness_gap_vs_threshold.csv`
- Rows evaluated: 60; valid mass-matched rows: 40
- Score: `mm_score = |Cliff's delta| * (-log10(MW p))`

## Best Per DM Threshold

| dm_threshold | min_pts | rmin_kpc | matched_pairs | median_tng | median_sparc | ratio_tng_over_sparc | cliffs_delta | mw_p | mm_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10.8 | 15 | 2.0 | 10 | 0.1014 | 0.0320 | 3.167 | 0.840 | 0.00171 | 2.325 |
| -10.7 | 15 | 2.0 | 14 | 0.1255 | 0.0304 | 4.128 | 0.867 | 0.000103 | 3.457 |
| -10.6 | 15 | 2.0 | 16 | 0.1059 | 0.0321 | 3.304 | 0.766 | 0.000238 | 2.774 |
| -10.5 | 15 | 2.0 | 18 | 0.0932 | 0.0344 | 2.712 | 0.636 | 0.00118 | 1.861 |
| -10.4 | 15 | 2.0 | 21 | 0.0880 | 0.0347 | 2.535 | 0.610 | 0.000749 | 1.906 |

## Top 12 Configurations (Overall)

| dm_threshold | min_pts | rmin_kpc | matched_pairs | dlogM_mean | ratio | cliffs_delta | mw_p | score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10.7 | 15 | 2.0 | 14 | 0.012 | 4.128 | 0.867 | 0.000103 | 3.457 |
| -10.6 | 15 | 2.0 | 16 | 0.011 | 3.304 | 0.766 | 0.000238 | 2.774 |
| -10.7 | 15 | 0.5 | 16 | 0.027 | 3.136 | 0.727 | 0.00049 | 2.405 |
| -10.7 | 15 | 1.0 | 15 | 0.017 | 3.437 | 0.733 | 0.000671 | 2.327 |
| -10.8 | 15 | 2.0 | 10 | 0.020 | 3.167 | 0.840 | 0.00171 | 2.325 |
| -10.7 | 15 | 0.0 | 17 | 0.026 | 2.800 | 0.696 | 0.000572 | 2.255 |
| -10.4 | 15 | 2.0 | 21 | 0.032 | 2.535 | 0.610 | 0.000749 | 1.906 |
| -10.5 | 15 | 2.0 | 18 | 0.022 | 2.712 | 0.636 | 0.00118 | 1.861 |
| -10.6 | 15 | 1.0 | 17 | 0.015 | 2.202 | 0.578 | 0.00425 | 1.370 |
| -10.6 | 15 | 0.5 | 18 | 0.025 | 2.143 | 0.562 | 0.00419 | 1.336 |
| -10.6 | 15 | 0.0 | 19 | 0.023 | 2.072 | 0.540 | 0.00463 | 1.261 |
| -10.8 | 15 | 1.0 | 11 | 0.026 | 2.039 | 0.620 | 0.0151 | 1.128 |

## Figures
- `analysis/results/tng_sparc_composition_sweep/caliper_0p20/fairness_best_ratio_vs_threshold.png`
- `analysis/results/tng_sparc_composition_sweep/caliper_0p20/fairness_best_effect_vs_threshold.png`
- `analysis/results/tng_sparc_composition_sweep/caliper_0p20/fairness_score_heatmaps_by_threshold.png`

