---
id: M0052
author: claude
timestamp: 2026-03-09T03:30:00Z
type: implementation-request
tags: [xcop-inversion, lensing-crosscheck, clash-nfw, smacs0723, response-function]
in_reply_to: [M0050, M0051]
priority: P1
repo_state:
  git_sha: 5bf2ca3efded6cbbacbd4026ff84c1a823d87d6e
status: pending
---

# X-COP-Only Response Function Inversion + Lensing Cross-Check

## WHAT CHANGED SINCE M0050/M0051

Three new lensing-calibrated cluster datasets are now available in the repo.
These remove the hydrostatic bias question entirely for the clusters they cover:

| Dataset | Location | Systems | Bins | Mass type |
|---------|----------|---------|------|-----------|
| CLASH NFW (Umetsu+2016) | `raw_data/observational/cluster_rar/clash_nfw_profiles.csv` | 20 | 100/cluster | Weak+strong lensing |
| SMACS J0723 (Liu+2023) | `raw_data/observational/cluster_rar/smacs0723_liu2023_profiles.csv` | 1 | 100 | X-ray HSE (lensing-confirmed) |
| Frontier Fields kappa | `raw_data/observational/cluster_rar/frontier_fields_kappa_profiles.csv` | 6 | 60/cluster | Strong lensing (core only) |
| CLASH summary | `raw_data/observational/cluster_rar/clash_nfw_summary.csv` | 20 | 1/cluster | NFW params + deficit at R500 |

Key finding from CLASH lensing: **median deficit at R500 = 1.80×** using lensing masses.
This is LARGER than the ~1.6× from hydrostatic masses. Hydrostatic bias makes the
problem worse, not better — lensing masses are higher, so the BEC underpredicts more.

SMACS J0723 (the first JWST deep field cluster): BEC underpredicts by **3.0×** across
30–2500 kpc. This is a massive cluster (M500 ~ 10^15 M_sun) where Liu+2023 showed
that hydrostatic and strong-lensing masses agree, explicitly confirming the RAR failure.

**IMPORTANT**: X-COP and CLASH are completely different cluster samples with zero
overlap. This means the CLASH lensing data cannot replace X-COP hydrostatic masses
for the same systems. Instead, CLASH provides an independent parallel test: if
X-COP (hydrostatic) and CLASH (lensing) give the same deficit, the deficit is real
and not an artifact of hydrostatic bias.

---

## CENTRAL QUESTION

This prompt has the original X-COP inversion as the primary analysis, PLUS a
parallel CLASH lensing inversion that answers the hydrostatic bias question
definitively.

The combined question: Does the cleanest current cluster sample require a modification
beyond the standard BEC response, and does this conclusion survive when you use
lensing masses that have no hydrostatic bias?

---

## STRICT SCOPE

- **Primary analysis**: X-COP only (12 clusters, hydrostatic masses)
- **Parallel lensing test**: CLASH NFW only (20 clusters, lensing masses)
- **Single-system validation**: SMACS J0723 (1 cluster, lensing-confirmed HSE)
- **Galaxy control**: SPARC
- Do NOT pool X-COP, CLASH, and ACCEPT in the main analysis
- Do NOT pool Frontier Fields with CLASH (core-only coverage, different systematics)
- Keep all source labels explicit

Output to a NEW timestamped directory under:

    /Users/russelllicht/Documents/New project/outputs/final_synthesis/xcop_inversion/

Work inside:

    /Users/russelllicht/Documents/New project/

---

## INDEPENDENCE AND WEIGHTING

For X-COP: 12 clusters are the primary independent units.
For CLASH: 20 clusters are the primary independent units.
Radial bins within a cluster are correlated in both samples.

Report BOTH pooled pointwise metrics AND cluster-level metrics for each sample.

Required cluster-level diagnostics (for both X-COP and CLASH):
- leave-one-cluster-out cross-validation
- cluster bootstrap uncertainty estimates
- per-cluster median residuals
- equal-cluster-weight fits in addition to point-weighted fits

---

## THE INVERSION (same for both samples)

At each radial bin:

    g_bar(r) = G * M_bar(<r) / r^2
    g_obs(r) = G * M_obs(<r) / r^2
    epsilon(r) = sqrt(g_bar(r) / g_dagger)

    nbar_required(r) = g_obs(r) / g_bar(r) - 1
    nbar_BEC(r) = 1 / (exp(epsilon) - 1)
    ratio(r) = nbar_required(r) / nbar_BEC(r)
    residual(r) = log10(nbar_required / nbar_BEC)

For X-COP: g_obs uses hydrostatic mass M_HE. M_bar from X-ray gas + stellar estimate.
For CLASH: g_obs uses lensing mass M_NFW. M_bar from the baryonic mass column in
`clash_nfw_profiles.csv` (gas fraction profile + stellar fraction).
For SMACS J0723: g_obs uses M_HSE. M_bar = M_gas + M_star from the profile CSV.

---

## DATA SOURCES

### X-COP (primary — hydrostatic)
Use existing X-COP loaders from prior packages:
- `analysis/cluster_junction/loaders.py`
- `analysis/finite_size_bec/loaders.py`
- Or the parsed profiles in `data/cluster_xray/parsed/`
- 12 clusters, ~583 total radial bins

### CLASH lensing (parallel — no hydrostatic bias)
New data in bec-dark-matter repo (copy to working directory or read directly):
- Profiles: `/Users/russelllicht/bec-dark-matter/raw_data/observational/cluster_rar/clash_nfw_profiles.csv`
  - Columns: cluster, z, r_kpc, M_NFW_Msun, M_bar_Msun, g_bar_m_s2, g_tot_m_s2,
    g_BEC_m_s2, log_g_bar, log_g_tot, log_g_BEC, r_over_r500, r_over_r200
  - 20 clusters × 100 radii = 2000 rows
- Summary: `/Users/russelllicht/bec-dark-matter/raw_data/observational/cluster_rar/clash_nfw_summary.csv`
  - Per-cluster: M_200c, c_200c, r_200c, r_500, M_500, deficit_at_r500

### SMACS J0723 (single-system validation)
- `/Users/russelllicht/bec-dark-matter/raw_data/observational/cluster_rar/smacs0723_liu2023_profiles.csv`
  - Columns: r_kpc, n_e_cm3, T_keV, rho_gas_kg_m3, M_gas_Msun, M_HSE_Msun,
    M_star_Msun, M_bar_Msun, g_bar_m_s2, g_tot_m_s2, g_BEC_m_s2, log_g_bar,
    log_g_tot, log_g_BEC
  - 1 cluster, 100 radii (10–3000 kpc), z = 0.388

### Frontier Fields (supplementary core-only)
- `/Users/russelllicht/bec-dark-matter/raw_data/observational/cluster_rar/frontier_fields_kappa_profiles.csv`
  - 6 clusters × 60 radii (core only, 10–500 kpc)
  - Use ONLY for core-region cross-check, not for full inversion

### SPARC (galaxy control)
- Existing SPARC data from prior loaders
- `/Users/russelllicht/bec-dark-matter/raw_data/observational/sparc/`
  or unified results in bec_rar_identity artifacts

---

## PHASE 1 — PER-CLUSTER INVERSION (X-COP)

Identical to original prompt. For each of the 12 X-COP clusters, compute at each
usable radial bin: epsilon, nbar_required, nbar_BEC, ratio, residual.

Handle edge cases explicitly:
- g_bar <= 0: skip and log
- nbar_required < 0: keep, flag, report
- nbar_required <= 0: do not silently include in log-space fits

Per-cluster report: cluster name, M500, T_keV, N_bins, median ratio,
median |residual|, RMSE(residual), fraction in [0.8,1.2], fraction in [0.9,1.1],
fraction with nbar_required < 0.

Output: `table_xcop_per_cluster_inversion.csv`

---

## PHASE 1L — PER-CLUSTER INVERSION (CLASH LENSING)

**NEW PHASE.** Run the identical inversion on CLASH lensing data.

For each of the 20 CLASH clusters, read from `clash_nfw_profiles.csv`:
- g_bar from the `g_bar_m_s2` column
- g_obs from the `g_tot_m_s2` column (this is lensing, not hydrostatic)
- Compute epsilon, nbar_required, nbar_BEC, ratio, residual at each radial bin

Same per-cluster report format as Phase 1.

Also run the inversion for SMACS J0723 (1 cluster) from its profile CSV.

Outputs:
- `table_clash_per_cluster_inversion.csv`
- `table_smacs0723_inversion.csv`

---

## PHASE 2 — MONEY PLOTS

### X-COP money plot (same as original)
- `fig_xcop_money_plot.png`: nbar_required vs epsilon, colored by cluster, BEC curve in black
- `fig_xcop_money_plot_with_errorbars.png`: with propagated uncertainties
- `fig_xcop_sparc_overlay.png`: X-COP + SPARC gray background

### CLASH lensing money plot (NEW)
- `fig_clash_money_plot.png`: same format, CLASH lensing data, colored by cluster
- `fig_clash_sparc_overlay.png`: CLASH + SPARC gray background

### Combined comparison (NEW)
- `fig_xcop_vs_clash_money_plot.png`: X-COP (circles) and CLASH (squares) on the same
  plot with BEC curve. This is the key figure — if both samples trace the same curve
  (or same systematic offset), the deficit is real regardless of mass measurement method.

### SMACS J0723 overlay (NEW)
- `fig_smacs0723_overlay.png`: SMACS J0723 points overlaid on the X-COP + CLASH plot

---

## PHASE 3 — COLLAPSE TESTS (both samples)

### X-COP collapse (same as original)
Bin X-COP data in epsilon bins. Compute median, 16/84 percentiles, MAD.
Stratify by cluster, r/R500, eta.

### CLASH collapse (NEW)
Same binning and stratification for CLASH lensing data.

### Cross-sample collapse (NEW)
At matched epsilon bins, compare:
- X-COP median nbar_required vs CLASH median nbar_required
- Are they consistent within scatter?
- Does the offset (if any) have the sign expected from hydrostatic bias?
  (CLASH lensing should give HIGHER nbar_required if hydrostatic bias suppresses M_obs)

Outputs:
- `table_xcop_collapse.csv`
- `table_clash_collapse.csv`
- `table_xcop_vs_clash_collapse.csv`
- `fig_xcop_collapse_by_cluster.png`
- `fig_clash_collapse_by_cluster.png`
- `fig_xcop_vs_clash_collapse.png`
- `fig_xcop_money_plot_colored_by_eta.png`
- `fig_xcop_money_plot_colored_by_r_over_r500.png`

---

## PHASE 3.5 — NOISE VS REAL SCATTER

Same as original for X-COP. If CLASH uncertainties are available (they are
approximate from NFW parameter errors), do the same for CLASH.

Output: `table_xcop_noise_vs_scatter.csv`, `table_clash_noise_vs_scatter.csv`

---

## PHASE 4 — CANDIDATE FITTING (X-COP primary, CLASH validation)

Fit candidates using ONLY X-COP data (primary analysis).
Then APPLY the X-COP-fitted candidates to CLASH to check cross-sample consistency.

Candidates:
- C0: nbar = 1 / (exp(epsilon) - 1)
- C1: nbar = A * epsilon^alpha / (exp(epsilon) - 1)
- C2: nbar = 1 / (exp(epsilon^p) - 1)
- C3: nbar = exp(beta * epsilon) / (exp(epsilon) - 1)
- C4: nbar = 1 / ([1 + (q-1)*epsilon]^(1/(q-1)) - 1)

For each candidate, report both point-weighted and equal-cluster-weight fits,
leave-one-cluster-out stability, cluster bootstrap uncertainty.

**NEW**: After fitting on X-COP, evaluate each candidate on CLASH lensing data.
Report RMSE, median ratio, and residual structure for CLASH using the X-COP-fitted
parameters. If a candidate fits X-COP but fails CLASH, it's overfitting to
hydrostatic systematics.

Outputs:
- `table_xcop_candidate_fits.csv`
- `table_clash_candidate_validation.csv` (NEW)
- `fig_xcop_candidate_comparison.png`
- `fig_clash_candidate_validation.png` (NEW)

---

## PHASE 4.5 — PER-CLUSTER CANDIDATE CONSISTENCY

Same as original for X-COP. Also report per-cluster parameters for CLASH.

Outputs:
- `table_xcop_per_cluster_params.csv`
- `table_clash_per_cluster_params.csv` (NEW)
- `fig_xcop_per_cluster_params.png`

---

## PHASE 5 — RADIAL PROFILE TEST

For each X-COP cluster AND each CLASH cluster, plot observed vs predicted g_obs(r)
for C0 and best candidate. Separate panels or figures for each sample.

Outputs:
- `fig_xcop_profile_overlays.png`
- `fig_clash_profile_overlays.png` (NEW)
- `table_xcop_radial_residuals.csv`
- `table_clash_radial_residuals.csv` (NEW)

---

## PHASE 6 — HYDROSTATIC BIAS SENSITIVITY (X-COP only)

Same as original. For b = 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, recompute
corrected nbar_required and all metrics.

**NEW comparison**: At each bias level, compare X-COP corrected deficit to
the CLASH lensing deficit (which needs no bias correction). The bias value
where X-COP matches CLASH is the empirically implied hydrostatic bias.

Outputs:
- `table_xcop_bias_sensitivity.csv`
- `table_xcop_bias_vs_clash.csv` (NEW: at each b, X-COP deficit vs CLASH deficit)
- `fig_xcop_bias_sensitivity.png`
- `fig_xcop_bias_vs_clash.png` (NEW)

---

## PHASE 7 — SPARC GALAXY IMPACT TEST

Same as original. Apply X-COP-fitted candidates to SPARC. Hard rejection
if SPARC is materially degraded.

Outputs:
- `table_xcop_galaxy_impact.csv`
- `fig_xcop_galaxy_null.png`

---

## PHASE 8 — CROSS-SAMPLE SYNTHESIS (REPLACES original comparison phase)

The original Phase 8 compared X-COP to pooled results. This version compares
X-COP (hydrostatic) to CLASH (lensing) to SMACS J0723 (lensing-confirmed HSE).

Report:
- X-COP median ratio vs CLASH median ratio vs SMACS J0723 median ratio
- X-COP RMSE vs CLASH RMSE
- X-COP best candidate vs CLASH cross-validation performance
- Whether the deficit is consistent across hydrostatic and lensing mass methods
- The implied hydrostatic bias from Phase 6 comparison

Key question this phase answers: **Is the cluster deficit a hydrostatic bias
artifact, or does it survive in lensing data?**

If CLASH lensing shows the same or larger deficit as X-COP hydrostatic,
hydrostatic bias is ruled out as the explanation.

Outputs:
- `table_cross_sample_synthesis.csv`
- `fig_cross_sample_comparison.png`

---

## REQUIRED OUTPUTS SUMMARY

### Tables (21 total)
| File | Source |
|------|--------|
| table_xcop_per_cluster_inversion.csv | Phase 1 |
| table_clash_per_cluster_inversion.csv | Phase 1L |
| table_smacs0723_inversion.csv | Phase 1L |
| table_xcop_collapse.csv | Phase 3 |
| table_clash_collapse.csv | Phase 3 |
| table_xcop_vs_clash_collapse.csv | Phase 3 |
| table_xcop_noise_vs_scatter.csv | Phase 3.5 |
| table_clash_noise_vs_scatter.csv | Phase 3.5 |
| table_xcop_candidate_fits.csv | Phase 4 |
| table_clash_candidate_validation.csv | Phase 4 |
| table_xcop_per_cluster_params.csv | Phase 4.5 |
| table_clash_per_cluster_params.csv | Phase 4.5 |
| table_xcop_radial_residuals.csv | Phase 5 |
| table_clash_radial_residuals.csv | Phase 5 |
| table_xcop_bias_sensitivity.csv | Phase 6 |
| table_xcop_bias_vs_clash.csv | Phase 6 |
| table_xcop_galaxy_impact.csv | Phase 7 |
| table_cross_sample_synthesis.csv | Phase 8 |

### Figures (20 total)
| File | Source |
|------|--------|
| fig_xcop_money_plot.png | Phase 2 |
| fig_xcop_money_plot_with_errorbars.png | Phase 2 |
| fig_xcop_sparc_overlay.png | Phase 2 |
| fig_clash_money_plot.png | Phase 2 |
| fig_clash_sparc_overlay.png | Phase 2 |
| fig_xcop_vs_clash_money_plot.png | Phase 2 |
| fig_smacs0723_overlay.png | Phase 2 |
| fig_xcop_collapse_by_cluster.png | Phase 3 |
| fig_clash_collapse_by_cluster.png | Phase 3 |
| fig_xcop_vs_clash_collapse.png | Phase 3 |
| fig_xcop_money_plot_colored_by_eta.png | Phase 3 |
| fig_xcop_money_plot_colored_by_r_over_r500.png | Phase 3 |
| fig_xcop_candidate_comparison.png | Phase 4 |
| fig_clash_candidate_validation.png | Phase 4 |
| fig_xcop_per_cluster_params.png | Phase 4.5 |
| fig_xcop_profile_overlays.png | Phase 5 |
| fig_clash_profile_overlays.png | Phase 5 |
| fig_xcop_bias_sensitivity.png | Phase 6 |
| fig_xcop_bias_vs_clash.png | Phase 6 |
| fig_xcop_galaxy_null.png | Phase 7 |
| fig_cross_sample_comparison.png | Phase 8 |

### Report
- report_xcop_inversion.md
- summary_xcop_inversion.json

---

## HONEST VERDICTS (PICK EXACTLY ONE)

1. **"X-COP does not require a modification, and CLASH lensing confirms this"**
   - Both samples are consistent with standard BEC within current precision
   - This substantially weakens the case for a universal cluster-scale failure

2. **"X-COP shows a small modification that CLASH lensing also supports"**
   - Both hydrostatic and lensing samples show the same modest deviation
   - A specific candidate improves on C0 robustly and preserves SPARC
   - The modification is real, not a hydrostatic artifact

3. **"X-COP and CLASH both confirm the cluster deficit survives"**
   - Standard BEC is robustly disfavored in BOTH samples
   - Lensing masses show the same or larger deficit as hydrostatic
   - Hydrostatic bias is ruled out as the explanation
   - The deficit is real and survives in the cleanest current data

4. **"X-COP and CLASH disagree, implicating hydrostatic bias"**
   - X-COP shows a deficit but CLASH does not (or vice versa)
   - The discrepancy is consistent with hydrostatic bias at b ~ [value]
   - The cluster RAR question cannot be resolved without mass-method agreement

5. **"The deficit has the wrong radial shape for any 1D response in both samples"**
   - No 1D candidate fixes coherent radial residual structure
   - This holds in both hydrostatic and lensing data
   - The issue is structural, not a global normalization problem

6. **"Both samples show the deficit but disagree on its magnitude or structure"**
   - X-COP deficit ≠ CLASH deficit in detail
   - Qualitative agreement (both underpredict) but quantitative disagreement
   - Cross-calibration issues remain unresolved

No hedging. Pick one.

---

## PHYSICS GUARDRAILS

- Do NOT pool X-COP with CLASH in a joint fit — they are independent cross-checks
- Do NOT treat CLASH NFW-reconstructed M_bar as high-precision (it uses approximate
  gas/stellar fraction profiles, not measured X-ray profiles)
- Do NOT overclaim from correlated radial bins — cluster-level metrics are mandatory
- CLASH baryonic masses are approximate (f_gas profile + f_star estimate).
  This is a systematic that works against the deficit (underestimating M_bar
  increases nbar_required). Note it explicitly.
- Hydrostatic bias is a sensitivity screen in X-COP, not an internally determined truth
- SPARC preservation is a hard requirement for all candidates
- Frontier Fields data covers only the strong-lensing core (<500 kpc). Do NOT use it
  for full inversions or verdicts. It is supplementary only.
- SMACS J0723 is a single system. It validates or contradicts but does not determine
  the verdict by itself.

---

## WHY THE LENSING DATA MATTERS

The M0044/M0045 cluster RAR tests and the M0048 xi-partition audit all used
hydrostatic masses. The persistent question has been: "Is the 1.6× deficit real,
or is it hydrostatic bias?"

The CLASH lensing data answers this:
- If CLASH lensing shows deficit ≈ X-COP hydrostatic deficit: bias is not the explanation
- If CLASH lensing shows deficit < X-COP: some of the deficit is bias, some is real
- If CLASH lensing shows no deficit: the entire cluster problem was hydrostatic bias

The preliminary answer from the NFW reconstruction: **CLASH lensing deficit is 1.80×
at R500, which is LARGER than the X-COP ~1.6×.** This strongly suggests the deficit
is real and lensing makes it worse.

This run will determine whether that preliminary finding holds up under the full
inversion analysis.

---

## COMPUTE REQUIREMENTS

All local. X-COP inversion (~583 bins) + CLASH inversion (~2000 bins) + SMACS J0723
(~100 bins) + SPARC control (~2800 bins) + candidate fitting. Under 15 minutes total.
