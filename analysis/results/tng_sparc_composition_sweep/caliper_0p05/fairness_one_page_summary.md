# TNG vs SPARC Fairness Sweep Summary

## Inputs
- Source CSV: `analysis/results/tng_sparc_composition_sweep/caliper_0p05/fairness_gap_vs_threshold.csv`
- Rows evaluated: 60; valid mass-matched rows: 40
- Score: `mm_score = |Cliff's delta| * (-log10(MW p))`

## Best Per DM Threshold

| dm_threshold | min_pts | rmin_kpc | matched_pairs | median_tng | median_sparc | ratio_tng_over_sparc | cliffs_delta | mw_p | mm_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10.8 | 15 | 2.0 | 9 | 0.1180 | 0.0329 | 3.583 | 0.877 | 0.002 | 2.366 |
| -10.7 | 15 | 2.0 | 13 | 0.1330 | 0.0329 | 4.037 | 0.858 | 0.000222 | 3.134 |
| -10.6 | 15 | 2.0 | 15 | 0.1180 | 0.0329 | 3.583 | 0.822 | 0.000136 | 3.180 |
| -10.5 | 15 | 2.0 | 16 | 0.0961 | 0.0370 | 2.600 | 0.648 | 0.00188 | 1.768 |
| -10.4 | 15 | 2.0 | 17 | 0.0938 | 0.0347 | 2.703 | 0.606 | 0.00273 | 1.552 |

## Top 12 Configurations (Overall)

| dm_threshold | min_pts | rmin_kpc | matched_pairs | dlogM_mean | ratio | cliffs_delta | mw_p | score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10.6 | 15 | 2.0 | 15 | 0.004 | 3.583 | 0.822 | 0.000136 | 3.180 |
| -10.7 | 15 | 2.0 | 13 | 0.005 | 4.037 | 0.858 | 0.000222 | 3.134 |
| -10.7 | 15 | 1.0 | 13 | 0.005 | 3.515 | 0.846 | 0.000272 | 3.018 |
| -10.7 | 15 | 0.5 | 13 | 0.005 | 3.515 | 0.846 | 0.000272 | 3.018 |
| -10.7 | 15 | 0.0 | 14 | 0.005 | 3.131 | 0.796 | 0.00037 | 2.732 |
| -10.6 | 15 | 0.5 | 15 | 0.004 | 2.375 | 0.751 | 0.000494 | 2.484 |
| -10.6 | 15 | 1.0 | 15 | 0.004 | 2.375 | 0.751 | 0.000494 | 2.484 |
| -10.8 | 15 | 2.0 | 9 | 0.010 | 3.583 | 0.877 | 0.002 | 2.366 |
| -10.6 | 15 | 0.0 | 16 | 0.004 | 2.323 | 0.703 | 0.000743 | 2.200 |
| -10.8 | 15 | 0.5 | 9 | 0.010 | 3.001 | 0.802 | 0.00472 | 1.867 |
| -10.8 | 15 | 1.0 | 9 | 0.010 | 3.010 | 0.802 | 0.00472 | 1.867 |
| -10.5 | 15 | 2.0 | 16 | 0.003 | 2.600 | 0.648 | 0.00188 | 1.768 |

## Figures
- `analysis/results/tng_sparc_composition_sweep/caliper_0p05/fairness_best_ratio_vs_threshold.png`
- `analysis/results/tng_sparc_composition_sweep/caliper_0p05/fairness_best_effect_vs_threshold.png`
- `analysis/results/tng_sparc_composition_sweep/caliper_0p05/fairness_score_heatmaps_by_threshold.png`

