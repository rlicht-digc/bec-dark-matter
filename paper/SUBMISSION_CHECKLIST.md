# Submission Checklist — g† Hunt Paper

Mapping claims in the manuscript to source fields in the latest refereeproof summary.

**Run folder:** `outputs/gdagger_hunt/20260224_152455_refereeproof`

## Baseline SPARC (Table 1)

| Metric | summary.json key | Value |
|--------|------------------|-------|
| N data | `baseline.n_data` | 2740 |
| Best kernel | `baseline.best_kernel_name` | BE_RAR |
| Best log10(s) | `baseline.best_kernel.log_scale_best` | -9.9347 |
| Δ from log10(g†) | computed from key above | 0.0139 dex |
| ΔAIC/N at ±0.1 dex | `baseline.scale_scan.delta_aic_pm_0p1_per_dof.mean_0p1` | 0.0521 |
| Peak sharpness | `baseline.scale_scan.peak_sharpness` | 28859 |

## Null/Control Dual-Window Stats (Table 2)

| Test | Type | Hits/N (±0.10) | Hits/N (±0.05) | 95% CP UB (±0.10 / ±0.05, only when k=0) |
|------|------|----------------|----------------|-------------------------------------------|
| A1 Global shuffle | Null | 0/500 (0.0000) | 0/500 (0.0000) | 0.0060 / 0.0060 |
| A2b Block-permute bins | Null | 79/500 (0.1580) | 39/500 (0.0780) | --- |
| A3 Galaxy circular shift | Null | 0/500 (0.0000) | 0/500 (0.0000) | 0.0060 / 0.0060 |
| A2 Within-bin shuffle | Control | 500/500 (1.0000) | 499/500 (0.9980) | --- |

## Validation Stability

| Metric | summary.json key | Value |
|--------|------------------|-------|
| GroupKFold mean log10(s) | `suite_b.best_log_scale_mean` | -9.9195 |
| GroupKFold std | `suite_b.best_log_scale_std` | 0.0437 |
| Fold-wise best-kernel split | `suite_b.fold_details[*].best_kernel` | BE_RAR: 4/5; coth: 1/5 |
| Cut sensitivity max Δ | `suite_c.max_delta_log_scale` | 0.0698 dex |
| Grid sensitivity max Δ | `suite_d.max_delta_log_scale` | 0.0000 dex |

## Cluster Pilot

| Metric | summary.json key | Value |
|--------|------------------|-------|
| Best log10(s) | `suite_g.best_log_scale` | -8.0000 |
| Δ from g† | `suite_g.delta_from_gdagger` | 1.9208 dex |

## Figure Asset Paths

| Figure | Source in run folder | Paper destination |
|--------|----------------------|-------------------|
| Fig 1 | `figures/fig1_three_panel.png` | `paper/figures/fig1_three_panel.png` |
| Fig 2 | `figures/fig2_validation_stability.png` | `paper/figures/fig2_validation_stability.png` |
| Fig 3 | `figures/fig3_negative_controls.png` | `paper/figures/fig3_negative_controls.png` |

## Reproducibility

| Item | Location |
|------|----------|
| Git commit | `summary.json -> reproducibility.git_commit` |
| RNG seed | `summary.json -> seed` |
| SHA256 hashes | `summary.json -> sha256_hashes` |
| Pip freeze | `<run_folder>/pip_freeze.txt` |
| Asset command | `python3 analysis/tools/make_paper_assets.py` (omit arg to use newest refereeproof run) |

*Last updated: 2026-02-24*
