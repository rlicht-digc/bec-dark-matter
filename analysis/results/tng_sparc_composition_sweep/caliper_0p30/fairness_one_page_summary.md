# TNG vs SPARC Fairness Sweep Summary

## Inputs
- Source CSV: `analysis/results/tng_sparc_composition_sweep/caliper_0p30/fairness_gap_vs_threshold.csv`
- Rows evaluated: 60; valid mass-matched rows: 40
- Score: `mm_score = |Cliff's delta| * (-log10(MW p))`

## Best Per DM Threshold

| dm_threshold | min_pts | rmin_kpc | matched_pairs | median_tng | median_sparc | ratio_tng_over_sparc | cliffs_delta | mw_p | mm_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10.8 | 15 | 2.0 | 12 | 0.1014 | 0.0320 | 3.167 | 0.819 | 0.000731 | 2.570 |
| -10.7 | 15 | 2.0 | 17 | 0.1180 | 0.0315 | 3.744 | 0.869 | 1.67e-05 | 4.150 |
| -10.6 | 15 | 2.0 | 18 | 0.1059 | 0.0322 | 3.286 | 0.765 | 9.33e-05 | 3.085 |
| -10.5 | 15 | 2.0 | 21 | 0.0927 | 0.0347 | 2.670 | 0.655 | 0.000292 | 2.316 |
| -10.4 | 15 | 2.0 | 23 | 0.0880 | 0.0329 | 2.671 | 0.626 | 0.000289 | 2.214 |

## Top 12 Configurations (Overall)

| dm_threshold | min_pts | rmin_kpc | matched_pairs | dlogM_mean | ratio | cliffs_delta | mw_p | score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10.7 | 15 | 2.0 | 17 | 0.061 | 3.744 | 0.869 | 1.67e-05 | 4.150 |
| -10.7 | 15 | 1.0 | 18 | 0.062 | 3.592 | 0.765 | 9.33e-05 | 3.085 |
| -10.6 | 15 | 2.0 | 18 | 0.043 | 3.286 | 0.765 | 9.33e-05 | 3.085 |
| -10.7 | 15 | 0.5 | 19 | 0.068 | 3.169 | 0.751 | 8.1e-05 | 3.071 |
| -10.7 | 15 | 0.0 | 20 | 0.065 | 2.724 | 0.715 | 0.000116 | 2.814 |
| -10.8 | 15 | 2.0 | 12 | 0.066 | 3.167 | 0.819 | 0.000731 | 2.570 |
| -10.5 | 15 | 2.0 | 21 | 0.059 | 2.670 | 0.655 | 0.000292 | 2.316 |
| -10.4 | 15 | 2.0 | 23 | 0.056 | 2.671 | 0.626 | 0.000289 | 2.214 |
| -10.6 | 15 | 1.0 | 19 | 0.044 | 2.321 | 0.607 | 0.00146 | 1.720 |
| -10.6 | 15 | 0.5 | 20 | 0.052 | 2.121 | 0.580 | 0.00178 | 1.594 |
| -10.6 | 15 | 0.0 | 21 | 0.049 | 2.072 | 0.556 | 0.00215 | 1.482 |
| -10.8 | 15 | 1.0 | 13 | 0.067 | 2.039 | 0.633 | 0.00657 | 1.382 |

## Figures
- `analysis/results/tng_sparc_composition_sweep/caliper_0p30/fairness_best_ratio_vs_threshold.png`
- `analysis/results/tng_sparc_composition_sweep/caliper_0p30/fairness_best_effect_vs_threshold.png`
- `analysis/results/tng_sparc_composition_sweep/caliper_0p30/fairness_score_heatmaps_by_threshold.png`

