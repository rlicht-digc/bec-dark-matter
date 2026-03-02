# External Audit Report: BEC-dark-matter Branch C/D Program

## Run Stamp
- RUN_ID: `20260228_011307`
- git HEAD: `f8f914933f3b0e3d9b2d1935c6377cd1f5db9a73`
- repo root: `/Users/russelllicht/bec-dark-matter`
- dataset sha256: `11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c`
- start_time_utc: `2026-02-28T01:13:07.440145Z`
- end_time_utc: `2026-02-28T01:13:07.878691Z`
- file counts: tests=86, runs=49, manifest_files=70, warnings=0

## Executive Summary
- Audit generation UTC: `2026-02-28T01:13:07.878691Z`
- Repo root: `/Users/russelllicht/bec-dark-matter`
- Git HEAD: `f8f914933f3b0e3d9b2d1935c6377cd1f5db9a73`
- Unified dataset SHA256: `11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c` (`/Users/russelllicht/bec-dark-matter/analysis/results/rar_points_unified.csv`)
- Core interpretation from artifact evidence:
  - Branch D refereeproof suite shows persistent scale recovery near g† with destructive null controls rejecting easy coincidences.
  - Branch C baseline bridge pooled-point shift is positive in restored runs, but per-galaxy shift and trend significance are weak.
  - Branch C decisive C1-C3 run (`BRANCHC_C1C2C3_20260228_004320`) concludes **"sampling artifact"** under galaxy-aware controls.
- Conservative conclusion: current artifacts do **not** establish a robust physical pooled residual-shift effect for Branch C; Branch D identifiability controls are stronger and internally coherent.

## Project Map
- Branch D (RAR control / g† identifiability): `analysis/tests/test_gdagger_hunt_refereeproof.py`, `analysis/gdagger_hunt.py`, `analysis/pipeline/run_referee_required_tests.py`
- Branch C (Paper3 density bridge): `analysis/paper3/paper3_bridge_pack.py`, `analysis/paper3/paper3_branchC_experiments.py`
- Entrypoints / guardrails: `analysis/run_branchD_rar_control.py`, `analysis/run_branchC_paper3_experiments.py`, `analysis/tools/repo_provenance_scan.py`

## Test Catalog (Designed vs Executed)
- Designed scripts cataloged: **86**
  - Branch C-tagged: 4
  - Branch D-tagged: 5
  - Adjacent/other: 77
- Executed run folders found: **49**
  - Branch C-tagged runs: 19
  - Branch D-tagged runs: 30
  - Unclassified/adjacent runs: 0
- Full machine-readable catalogs are in the INDEX JSON (tests + runs arrays).

## Branch D Results (RAR=BEC g† Identifiability)
### D1. Refereeproof Run Folders Detected
- `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_145514_refereeproof`
- `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_145527_refereeproof`
- `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_151048_refereeproof`
- `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_151241_refereeproof`
- `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_151453_refereeproof`
- `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof`

### D2. Key Metrics Table (from summary.json)
| run | best_kernel | best_log_scale | A1 p±0.10 | A2 control p±0.10 | A2b null p±0.10 | A2b null p±0.05 | A3 p±0.10 | maxΔlogscale(cuts) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20260224_145514_refereeproof | BE_RAR | -9.934740124750645 | 0.0 | 1.0 | None | None | 0.0 | 0.06982542067017583 |
| 20260224_145527_refereeproof | BE_RAR | -9.934740124750645 | 0.0 | 1.0 | None | None | 0.0 | 0.06982542067017583 |
| 20260224_151048_refereeproof | BE_RAR | -9.934740124750645 | 0.0 | 1.0 | None | None | 0.0 | 0.06982542067017583 |
| 20260224_151241_refereeproof | BE_RAR | -9.934740124750645 | 0.0 | 1.0 | 0.158 | None | 0.0 | 0.06982542067017583 |
| 20260224_151453_refereeproof | BE_RAR | -9.934740124750645 | 0.0 | 1.0 | 0.19 | None | 0.0 | 0.06982542067017583 |
| 20260224_152455_refereeproof | BE_RAR | -9.934740124750645 | 0.0 | 1.0 | 0.158 | 0.078 | 0.0 | 0.06982542067017583 |

### D3. Headline Evidence (exact keys)
- `baseline.best_kernel_name = BE_RAR`
- `baseline.best_log_scale = -9.934740124750645`
- `suite_a.A2b_block_permute_bins.p_within_0p10_dex = 0.158`
- `suite_a.A2b_block_permute_bins.p_within_0p05_dex = 0.078`
- `suite_f.F1_eta_fixed.a_Lambda.aic = 15994.65019980624`
- `suite_f.F1_eta_fixed.g_dagger.aic = 10840.180208741956`

### D3b. Coverage of Required Branch D Controls (exact values)
- Kernel matcher / scale scan:
  - `baseline.best_kernel_name = BE_RAR`
  - `baseline.best_log_scale = -9.934740124750645`
  - `baseline.scale_scan.peak_sharpness = 28858.973779990905`
  - `baseline.scale_scan.delta_aic_pm_0p1_dex.mean_0p1 = 142.7425071197058`
- AIC/BIC availability:
  - AIC arrays and ΔAIC diagnostics are present in `baseline.scale_scan` and `suite_f`.
  - BIC is **not explicitly stored** in `summary.json` (`Not evidenced in artifacts`).
- CV (grouped by galaxy):
  - `suite_b.best_log_scale_mean = -9.919461088139231`
  - `suite_b.best_log_scale_std = 0.043658509342881124`
  - `suite_b.best_kernel_frequency = {'coth': 1, 'BE_RAR': 4}`
- Null tests and controls (Suite A):
  - `A1 global null p±0.10 = 0.0`
  - `A2 within-bin control p±0.10 = 1.0`
  - `A2b block-permute null p±0.10 = 0.158`
  - `A3 within-galaxy null p±0.10 = 0.0`
  - `A1 CP upper bound (0 hits, ±0.10) = 0.005973551516349596`
- Cut/sample sensitivity (Suite C):
  - `suite_c.max_delta_log_scale = 0.06982542067017583`
  - `suite_c.all_within_0p05_dex = False`
- Grid invariance (Suite D):
  - `suite_d.max_delta_log_scale = 0.0`
  - `suite_d.within_0p02_dex = True`
- Negative controls (Suite E):
  - `E1_noise_sigma_0.3.delta_from_gdagger = 0.18542303756363232`
  - `E1_noise_sigma_0.3.sharpness_ratio = 0.6406049026316474`
  - `E3_galaxy_swap.delta_from_gdagger = 0.269563298097772`
  - `E3_galaxy_swap.sharpness_ratio = 0.23067772374796283`
- Nearby-scale comparisons (Suite F):
  - `F1_eta_fixed: AIC(a_Lambda)=15994.65019980624, AIC(g_dagger)=10840.180208741956`
  - `F2_eta_free: AIC(a_Lambda)=10889.775236809575, AIC(g_dagger)=10839.657704913801`
- Non-RAR pilot / explicit null (Suite G):
  - `suite_g.best_log_scale = -8.000004154309972`
  - `suite_g.delta_from_gdagger = 1.9208145996424033`

### D4. What Branch D Proves / Does Not Prove
- Proves (artifact-backed): for SPARC-control refereeproof runs, destructive nulls do not reproduce tight scale concentration around g†, while structure-preserving within-bin control does (expected).
- Does not prove: universal physical origin across all datasets by itself; this suite is identifiability-focused and model-comparison-focused.
- Figures referenced:
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/figures/fig1_three_panel.png`
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/figures/fig2_validation_stability.png`
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/figures/fig3_negative_controls.png`

## Branch C Results (Paper3 Density Bridge)
### C0. Early High-Density Prototype Runs (paper3_density_window_report)
| run | pooled_shift_dex | perm_p | theilsen_slope | bootstrap_p |
|---|---:|---:|---:|---:|
| 20260227_143339 | 0.184776 | 9.999e-05 | 0.043813 | 0.0476 |
| 20260227_143408 | 0.184776 | 9.999e-05 | 0.043813 | 0.0476 |
| 20260227_174839 | -0.010387 | 0.70223 | -0.061178 | 0.3204 |
| 20260227_174849 | 0.156497 | 9.999e-05 | -0.020133 | 0.7984 |
| 20260227_174900 | 0.033429 | 0.0746925 | 0.001740 | 0.9748 |

### C1. Baseline Bridge-Pack Runs (Executed)
| run | actual_csv_sha | require_csv_sha | include_ss20 | ss20_excluded | pooled_shift_dex | pergal_shift_and_p | theilsen_slope |
|---|---|---|---|---:|---:|---|---:|
| 20260227_162156 | None | None | None | None | 0.184776 | 0.020178 dex; permutation p-value: 0.129887 | 0.043813 |
| 20260227_162238 | None | None | None | None | 0.184776 | 0.020178 dex; permutation p-value: 0.129887 | 0.043813 |
| 20260227_175543 | None | None | None | None | 0.156497 | 0.103712 dex; permutation p-value: 0.428557 | -0.020133 |
| 20260227_175557 | None | None | None | None | 0.156497 | 0.103712 dex; permutation p-value: 0.428557 | -0.020133 |
| 20260227_181036 | None | None | None | None | 0.156497 | 0.103712 dex; permutation p-value: 0.428557 | -0.020133 |
| 20260227_181106 | None | None | None | None | 0.156497 | 0.103712 dex; permutation p-value: 0.428557 | -0.020133 |
| 20260227_181129 | None | None | None | None | 0.156497 | 0.103712 dex; permutation p-value: 0.428557 | -0.020133 |
| 20260227_193632 | None | None | False | 40 | 0.027282 | -0.005223 dex; permutation p-value: 0.693731 | 0.006696 |
| 20260227_193650 | None | None | True | 0 | 0.027282 | -0.005223 dex; permutation p-value: 0.693731 | 0.006696 |
| 20260227_194611 | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | False | 40 | 0.027282 | -0.005223 dex; permutation p-value: 0.693731 | 0.006696 |

### C2. Decisive C1-C3 Runs (Executed)
| run | final_decision | input_csv_sha256 | C1 weighted shift | C1 weighted p_block | C2 matched | C2 valid bins |
|---|---|---|---:|---:|---:|---:|
| BRANCHC_C1C2C3_20260228_001114 | sampling artifact | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | 0.050821975424623744 | 0.0805919408059194 | 0.23049695215247568 | 4 |
| BRANCHC_C1C2C3_20260228_001605 | sampling artifact | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | 0.050821975424623744 | 0.0845771144278607 | 0.22429160504767454 | 3 |
| BRANCHC_C1C2C3_20260228_002303 | sampling artifact | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | 0.050821975424623744 | 0.0845771144278607 | 0.22429160504767454 | 3 |
| BRANCHC_C1C2C3_20260228_004320 | sampling artifact | 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c | 0.050821975424623744 | 0.0852914708529147 | 0.23049695215247568 | 4 |

### C3. Source-Stratified Replication (latest C1-C3 run)
| subset | status | n_points | n_galaxies | c1_shift_weighted | c1_p_weighted | verdict |
|---|---|---:|---:|---:|---:|---|
| SPARC-only | ok | 2784 | 131 | 0.024974681762989448 | 0.0170982901709829 | consistent direction |
| GHASP-only | ok | 3877 | 69 | 0.016325857110713216 | 0.8984101589841016 | inconclusive (N too small) |
| deBlok2002-only | skipped | 365 | 18 | None | None | inconclusive (N too small) |
| WALLABY-only | skipped | 1220 | 165 | None | None | inconclusive (N too small) |
| WALLABY+WALLABY_DR2 | skipped | 1414 | 204 | None | None | inconclusive (N too small) |

### C4. Branch C Interpretation (Conservative)
- Restored hash-locked baseline run shows pooled positive shift but weak/negative per-galaxy signal and non-significant trend slope CI crossing zero.
- C1-C3 latest run reports `final_decision = sampling artifact` with weighted block-permutation p-value > 0.01 and limited valid within-bin support.
- Therefore, this artifact set does **not** support a robust physical pooled-shift claim at present.
- Figures referenced:
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/rho_vs_residual_scatter.png`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/residual_hist_high_vs_low.png`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/residual_vs_gbar_high_density.png`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/figures/C1_block_nulls.png`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/figures/C2_binwise_shift.png`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/figures/C3_source_replication.png`

## Corrections & Incident Timeline
### CORR-01: AASTeX acknowledgments syntax compliance
- Status: `partially_evidenced_current_state_only`
- What was wrong/risk: Deprecated AASTeX acknowledgment command can break manuscript checks.
- What changed: Manuscript currently uses environment form `begin/end{acknowledgments}`.
- Why this matters scientifically: Formatting compliance risk reduced; no direct Branch C/D metric impact.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/paper/main.tex` | `text:line=277` | `\begin{acknowledgments}`

### CORR-02: CP bound terminology and explicit storage
- Status: `evidenced`
- What was wrong/risk: Zero-hit null experiments need explicit binomial upper bounds.
- What changed: Refereeproof suite computes Clopper-Pearson upper bounds and stores `p_upper_95*` fields.
- Why this matters scientifically: Null proximity claims are bounded conservatively when hits=0.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/analysis/tests/test_gdagger_hunt_refereeproof.py` | `text:line=213` | `def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:`
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json` | `json:suite_a.A1_global.p_upper_95_0p10` | `0.005973551516349596`

### CORR-03: Null vs control relabeling
- Status: `evidenced`
- What was wrong/risk: Within-bin shuffle can be misinterpreted as a destructive null.
- What changed: Suite A2 is explicitly labeled `structure_preserving_control` and annotated as not a null test.
- Why this matters scientifically: Interpretation avoids false evidence inflation from a control expected to give p≈1.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json` | `json:suite_a.A2_within_bin.type` | `structure_preserving_control`
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json` | `json:suite_a.A2_within_bin.note` | `Within-bin shuffle preserves the conditional distribution p(y|x) by construction. It is NOT a null test; p≈1 is the expected outcome. This control confirms that the scale preference resides in the global x-y correlation, not in bin-local structure.`

### CORR-04: Added destructive bin-aware null (A2b)
- Status: `evidenced`
- What was wrong/risk: Global and galaxy-shift nulls alone do not isolate cross-bin composition effects.
- What changed: Suite A2b block-permute-bin destructive null added with dedicated output fields.
- Why this matters scientifically: Shows intermediate proximity rates (`p_within_0p10_dex=0.158`), refining null interpretation.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json` | `json:suite_a.A2b_block_permute_bins.type` | `destructive_bin_aware_null`
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json` | `json:suite_a.A2b_block_permute_bins.p_within_0p10_dex` | `0.158`
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json` | `json:suite_a.A2b_block_permute_bins.p_within_0p05_dex` | `0.078`

### CORR-05: Dual-window proximity (±0.05/±0.10 dex)
- Status: `evidenced`
- What was wrong/risk: Single-window proximity can hide scale-local sensitivity.
- What changed: Dual-window proximity function emits both ±0.05 and ±0.10 dex hit fractions and p-values.
- Why this matters scientifically: Improves discrimination between tight and broad scale concentration under null.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/analysis/tests/test_gdagger_hunt_refereeproof.py` | `text:line=224` | `def compute_dual_window_proximity(`
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json` | `json:suite_a.A2b_block_permute_bins.p_within_0p05_dex` | `0.078`
  - `/Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json` | `json:suite_a.A2b_block_permute_bins.p_within_0p10_dex` | `0.158`

### CORR-06: SPARC path resolver + dataset rebuild hash change
- Status: `evidenced`
- What was wrong/risk: Legacy-path drift could drop SPARC coverage and shift pooled residual statistics.
- What changed: Unified pipeline adds path resolution/guardrails and emits stamped metadata; dataset hash changed from backup `430a75f2...` to restored `11742ae3...`.
- Why this matters scientifically: Branch C headline changed materially: pooled shift from ~0.184776 dex (older run) to ~0.027282 dex (restored hash-locked run).
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/analysis/pipeline/09_unified_rar_pipeline.py` | `text:line=769` | `def resolve_sparc_paths():`
  - `/Users/russelllicht/bec-dark-matter/analysis/pipeline/09_unified_rar_pipeline.py` | `text:line=3997` | `"SPARC guardrail triggered: insufficient SPARC coverage. "`
  - `/Users/russelllicht/bec-dark-matter/analysis/results/rar_points_unified__430a75f2__20260227.csv` | `sha256` | `430a75f28edcf322c0dc88f768e89003d9970dc1b70f548ff273fa3caea4283d`
  - `/Users/russelllicht/bec-dark-matter/analysis/results/rar_points_unified.csv` | `sha256` | `11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_162238/paper3_density_bridge_report.txt` | `text:- pooled-point median shift` | `0.184776`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/paper3_density_bridge_report.txt` | `text:- pooled-point median shift` | `0.027282`

### CORR-07: galaxy_key canonicalization and default SS20 exclusion
- Status: `evidenced`
- What was wrong/risk: Mixed galaxy naming and SS20 single-point stubs can distort pooled-point analyses.
- What changed: Bridge pack canonicalizes `galaxy_key` and excludes `SS20_*` by default unless opt-in.
- Why this matters scientifically: Improves grouping integrity; report explicitly shows excluded SS20 points.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/analysis/paper3/paper3_bridge_pack.py` | `text:line=241` | `galaxy_key = raw_df[mapping["galaxy_key"]].astype(str).map(canonicalize_galaxy_name)`
  - `/Users/russelllicht/bec-dark-matter/analysis/paper3/paper3_bridge_pack.py` | `text:line=1004` | `if not args.include_ss20:`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/paper3_density_bridge_report.txt` | `text:include_ss20` | `False`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/paper3_density_bridge_report.txt` | `text:ss20_excluded_points` | `40`

### CORR-08: Meta stamping + enforce-hash execution guardrails
- Status: `evidenced`
- What was wrong/risk: Unstamped runs allow silent dataset drift.
- What changed: Pipeline writes `rar_points_unified.meta.json` with output SHA/git head; bridge and Branch C scripts enforce `require_csv_sha` checks.
- Why this matters scientifically: Run provenance is auditable and hash-locked.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/analysis/results/rar_points_unified.meta.json` | `json:output_csv_sha256` | `11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c`
  - `/Users/russelllicht/bec-dark-matter/analysis/results/rar_points_unified.meta.json` | `json:git_head` | `f8f914933f3b0e3d9b2d1935c6377cd1f5db9a73`
  - `/Users/russelllicht/bec-dark-matter/analysis/paper3/paper3_bridge_pack.py` | `text:line=976` | `if args.require_csv_sha is not None:`
  - `/Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/paper3_density_bridge_report.txt` | `text:require_csv_sha` | `11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c`

### CORR-09: Regression tests for SPARC presence floors
- Status: `evidenced`
- What was wrong/risk: SPARC depletion can silently pass unless explicitly tested.
- What changed: Regression test enforces minimum SPARC points and galaxy-key counts in loader and unified CSV.
- Why this matters scientifically: Protects Branch C/D against recurrence of SPARC-missing drift.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/analysis/tests/test_unified_pipeline_regression.py` | `text:line=68` | `assert sparc_points >= 2500, (`
  - `/Users/russelllicht/bec-dark-matter/analysis/tests/test_unified_pipeline_regression.py` | `text:line=73` | `assert sparc_galaxies >= 120, (`

### CORR-10: Branch entrypoints + provenance scan guardrails
- Status: `evidenced`
- What was wrong/risk: Cross-repo leakage and wrong-CWD execution can invalidate provenance.
- What changed: Dedicated Branch C/D entrypoints enforce repo-root checks and run-stamp sidecars; provenance scanner reports BH references.
- Why this matters scientifically: Improves reproducibility and auditability of execution context.
- Evidence:
  - `/Users/russelllicht/bec-dark-matter/analysis/run_branchC_paper3_experiments.py` | `text:line=45` | `"Wrong repo execution context. "`
  - `/Users/russelllicht/bec-dark-matter/analysis/run_branchD_rar_control.py` | `text:line=45` | `"Wrong repo execution context. "`
  - `/Users/russelllicht/bec-dark-matter/analysis/tools/repo_provenance_scan.py` | `docstring` | `Scan bec-dark-matter for potential provenance leakage from bh-singularity.`
  - `/Users/russelllicht/bec-dark-matter/analysis/results/provenance_scan_20260228_001541.md` | `text:Vendoring recommendation` | `- Vendoring recommendation: **no vendoring required** (no identical-hash matches found).`

## Evidence Map (Major Claims)
| claim | file | locator | value |
|---|---|---|---|
| Refereeproof baseline best kernel | /Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json | json:baseline.best_kernel_name | BE_RAR |
| Refereeproof baseline best log scale | /Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json | json:baseline.best_log_scale | -9.934740124750645 |
| Destructive block-permute null p(±0.10 dex) | /Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json | json:suite_a.A2b_block_permute_bins.p_within_0p10_dex | 0.158 |
| Destructive block-permute null p(±0.05 dex) | /Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json | json:suite_a.A2b_block_permute_bins.p_within_0p05_dex | 0.078 |
| Nearby-scale F1 eta-fixed AIC(a_Lambda) | /Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json | json:suite_f.F1_eta_fixed.a_Lambda.aic | 15994.65019980624 |
| Nearby-scale F1 eta-fixed AIC(g_dagger) | /Users/russelllicht/bec-dark-matter/outputs/gdagger_hunt/20260224_152455_refereeproof/summary.json | json:suite_f.F1_eta_fixed.g_dagger.aic | 10840.180208741956 |
| Branch C baseline pooled shift (restored hash) | /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/paper3_density_bridge_report.txt | text:line=90 | - pooled-point median shift (top-bottom): 0.027282 dex |
| Branch C baseline per-galaxy shift | /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/paper3_density_bridge_report.txt | text:line=92 | - per-galaxy median shift (top-bottom): -0.005223 dex; permutation p-value: 0.693731 |
| Branch C baseline Theil-Sen slope | /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/20260227_194611/paper3_density_bridge_report.txt | text:line=85 | - Theil-Sen slope: 0.006696 |
| Branch C C1 weighted shift | /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/summary_C1C2C3.json | json:C1.observed.shift_weighted | 0.050821975424623744 |
| Branch C C1 weighted block-permutation p | /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/summary_C1C2C3.json | json:C1.p_weighted_block | 0.0852914708529147 |
| Branch C C2 matched aggregate shift | /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/summary_C1C2C3.json | json:C2.aggregate_matched | 0.23049695215247568 |
| Branch C final decision | /Users/russelllicht/bec-dark-matter/outputs/paper3_high_density/BRANCHC_C1C2C3_20260228_004320/summary_C1C2C3.json | json:final_decision | sampling artifact |

## Limitations / Not Claimed
- No claim is made beyond available local artifacts; no web or external DB checks were used.
- Several correction items are evidenced in current code/state but lack granular pre-fix commits due coarse commit history (`main` currently squashed at `f8f9149...`).
- Branch D wrapper runs under `outputs/branchD_rar_control/20260228_002202` and `.../20260228_002303` contain run-stamps only (smoke/dry execution), not full science outputs.
- If an artifact/key was absent, it is treated as `Not evidenced in artifacts` rather than inferred.

## Reproducibility Instructions
Run from repo root `/Users/russelllicht/bec-dark-matter`:
```bash
python3 analysis/run_branchD_rar_control.py --dataset analysis/results/rar_points_unified.csv --seed 42 --n_shuffles 1000
python3 analysis/run_branchC_paper3_experiments.py --dataset analysis/results/rar_points_unified.csv --seed 42
python3 analysis/paper3/paper3_branchC_experiments.py --rar_points_file analysis/results/rar_points_unified.csv --require_csv_sha 11742ae37d3cfdab57014e40ee982de07d7955d3ad1d02cce94587530a148b2c --n_perm 10000 --n_bins 15 --seed 42
```
- Confirm run stamps include repo root, git head, dataset sha, and output folder before interpreting results.

## Audit Artifacts
- Index JSON: `/Users/russelllicht/bec-dark-matter/analysis/results/EXTERNAL_AUDIT_INDEX_20260228_011307.json`
- File manifest CSV: `/Users/russelllicht/bec-dark-matter/analysis/results/EXTERNAL_AUDIT_FILE_MANIFEST_20260228_011307.csv`
- Math appendix: `/Users/russelllicht/bec-dark-matter/analysis/results/EXTERNAL_AUDIT_APPENDIX_MATH_20260228_011307.md`

