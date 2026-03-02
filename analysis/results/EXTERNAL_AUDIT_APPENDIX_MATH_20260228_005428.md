# External Audit Appendix: Math Definitions (Branch C/D)

## 1. RAR/BEC Prediction and Residuals
- Baseline predictor (as implemented):
  - `g_pred = g_bar / (1 - exp(-sqrt(g_bar / g_dagger)))`
- Point residual in dex:
  - `log_resid = log10(g_obs) - log10(g_pred)`
- Branch C uses per-galaxy `rho_score` and top/bottom rank split over eligible galaxies.

## 2. Branch C Statistics
- C1 pooled median shift:
  - `shift = median(resid_top) - median(resid_bottom)`
- C1 galaxy-weighted median uses per-point weights `w=1/N_points(galaxy)` within each group.
- C1 galaxy-block permutation p-value:
  - Permute top/bottom galaxy labels preserving group sizes; recompute shift.
  - Two-sided permutation p: `(1 + #(|null| >= |obs|)) / (1 + N_perm)`
- C2 within-bin shift:
  - Bin by `log_gbar` (default 15 bins), require min counts per group per bin.
  - Bin statistic: median(top_bin) - median(bottom_bin).
  - Uncertainty: galaxy-block bootstrap CI (2.5%, 97.5%).
- C2 aggregate statistics:
  - Equal-bin aggregate: arithmetic mean across valid bins.
  - Matched-bin aggregate: weighted mean with `w_bin ~ min(n_top_bin, n_bottom_bin)`.
- C3 source-stratified replication: rerun C1/C2 on source subsets with top_n shrink if eligibility is insufficient.

## 3. Branch D Statistics (Refereeproof)
- Scale scan / kernel matching uses AIC-based comparison across candidate kernels and scales.
- Peak sharpness is reported from second-derivative behavior of AIC around optimum.
- Null/control suites include:
  - A1 global shuffle null (destructive).
  - A2 within-bin shuffle control (structure-preserving; explicitly *not* a destructive null).
  - A2b block-permute-bin null (destructive, bin-aware).
  - A3 within-galaxy circular-shift null (destructive).
- Dual-window proximity rates:
  - `p_within_0p05_dex = hits(|log_scale - log_gdagger| < 0.05)/N`
  - `p_within_0p10_dex = hits(|log_scale - log_gdagger| < 0.10)/N`
- For zero-hit cases, Clopper-Pearson one-sided upper 95% bound is reported (`p_upper_95*`).
- Nearby-scale comparisons (Suite F):
  - F1 eta-fixed direct substitution AIC.
  - F2 eta-free matched-DoF comparison with fitted amplitude.

## 4. Provenance/Guardrail Math-Adjacent Definitions
- Dataset immutability guard: SHA256 hash equality check for required CSV.
- Run stamp provenance tuple: `(repo_root, git_head, dataset_sha256, output_dir, timestamp_utc)`.

## 5. Code Pointers
- Branch C core: `/Users/russelllicht/bec-dark-matter/analysis/paper3/paper3_branchC_experiments.py`
- Branch C baseline bridge: `/Users/russelllicht/bec-dark-matter/analysis/paper3/paper3_bridge_pack.py`
- Branch D refereeproof suite: `/Users/russelllicht/bec-dark-matter/analysis/tests/test_gdagger_hunt_refereeproof.py`
- Branch D kernel/scan engine: `/Users/russelllicht/bec-dark-matter/analysis/gdagger_hunt.py`
