# TNG vs SPARC Fairness Sweep Summary

## Inputs
- Source CSV: `analysis/results/tng_sparc_composition_sweep/caliper_0p10/fairness_gap_vs_threshold.csv`
- Rows evaluated: 60; valid mass-matched rows: 40
- Score: `mm_score = |Cliff's delta| * (-log10(MW p))`

## Best Per DM Threshold

| dm_threshold | min_pts | rmin_kpc | matched_pairs | median_tng | median_sparc | ratio_tng_over_sparc | cliffs_delta | mw_p | mm_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10.8 | 15 | 2.0 | 9 | 0.1180 | 0.0329 | 3.583 | 0.877 | 0.002 | 2.366 |
| -10.7 | 15 | 2.0 | 13 | 0.1330 | 0.0329 | 4.037 | 0.858 | 0.000222 | 3.134 |
| -10.6 | 15 | 2.0 | 15 | 0.1180 | 0.0329 | 3.583 | 0.822 | 0.000136 | 3.180 |
| -10.5 | 15 | 2.0 | 16 | 0.0961 | 0.0370 | 2.600 | 0.648 | 0.00188 | 1.768 |
| -10.4 | 15 | 2.0 | 18 | 0.0961 | 0.0338 | 2.842 | 0.636 | 0.00118 | 1.861 |

## Top 12 Configurations (Overall)

| dm_threshold | min_pts | rmin_kpc | matched_pairs | dlogM_mean | ratio | cliffs_delta | mw_p | score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -10.6 | 15 | 2.0 | 15 | 0.004 | 3.583 | 0.822 | 0.000136 | 3.180 |
| -10.7 | 15 | 2.0 | 13 | 0.005 | 4.037 | 0.858 | 0.000222 | 3.134 |
| -10.8 | 15 | 2.0 | 9 | 0.010 | 3.583 | 0.877 | 0.002 | 2.366 |
| -10.7 | 15 | 0.5 | 14 | 0.011 | 3.180 | 0.735 | 0.00102 | 2.198 |
| -10.7 | 15 | 1.0 | 14 | 0.011 | 3.312 | 0.724 | 0.0012 | 2.117 |
| -10.7 | 15 | 0.0 | 15 | 0.010 | 2.817 | 0.698 | 0.00122 | 2.034 |
| -10.4 | 15 | 2.0 | 18 | 0.010 | 2.842 | 0.636 | 0.00118 | 1.861 |
| -10.5 | 15 | 2.0 | 16 | 0.003 | 2.600 | 0.648 | 0.00188 | 1.768 |
| -10.6 | 15 | 0.5 | 16 | 0.009 | 2.143 | 0.625 | 0.00273 | 1.602 |
| -10.6 | 15 | 1.0 | 16 | 0.009 | 2.230 | 0.617 | 0.00309 | 1.549 |
| -10.6 | 15 | 0.0 | 17 | 0.008 | 2.072 | 0.592 | 0.00341 | 1.459 |
| -10.8 | 15 | 0.5 | 10 | 0.017 | 2.498 | 0.620 | 0.0211 | 1.039 |

## Figures
- `analysis/results/tng_sparc_composition_sweep/caliper_0p10/fairness_best_ratio_vs_threshold.png`
- `analysis/results/tng_sparc_composition_sweep/caliper_0p10/fairness_best_effect_vs_threshold.png`
- `analysis/results/tng_sparc_composition_sweep/caliper_0p10/fairness_score_heatmaps_by_threshold.png`

