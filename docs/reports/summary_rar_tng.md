# WHERE WE ARE NOW (RAR + TNG Rerun)

## Run scope
- Output folder: `/Users/russelllicht/Documents/New project/rerun_outputs/20260223_203407`
- Core batteries completed: Test 1, Test 2, Test 3, Test 4, Test 5, and consistency check
- Determinism check: Test 1 rerun with identical seeds reproduced identical headline stats (`abs_diff = 0` for all tracked metrics)

## Dataset sizes at every stage

### Global inputs
| Stage | Points | Galaxies | Notes |
|---|---:|---:|---|
| `rar_points_unified.csv` loaded | 15,745 | 2,892 | 21 sources |
| `galaxy_results_unified.csv` loaded | — | 2,892 rows | 1 row/galaxy result |
| SPARC subset from unified points | 2,784 | 131 | Used by Tests 1, 2, 3, 4 |
| Chosen verified-clean TNG per-point set | 150,000 | 3,000 | `rar_points_CLEAN.parquet`; all galaxies have `>=5` points |

## Exact cut flow (removed and why)

### Test 1: Phase peak null distribution (SPARC)
| Step | Points | Galaxies | Removed | Why |
|---|---:|---:|---:|---|
| SPARC input | 2,784 | 131 | 0 | Initial SPARC-only subset |
| Residual consistency check | 2,784 | 131 | 0 | `fraction(|res_check-log_res|>0.01)=0.0`; kept `log_res` |
| Restrict to fixed bin window `[-13.5, -8.0)` | 2,779 | 131 | 5 | Outside spec window (`1` below, `4` above) |
| Drop bins with `N<10` | 2,759 | 131 | 20 | Spec minimum occupancy; dropped bins centered at `-13.375, -13.125, -12.875, -12.625, -12.375, -8.125` |
| Final fit sample | 2,759 | 131 | — | 16 bins used (of 22 total centers) |

### Test 2: Mass-matched SPARC vs TNG phase
| Step | SPARC (points/gal) | TNG (points/gal) | Removed | Why |
|---|---:|---:|---:|---|
| Inputs to Test 2 | 2,784 / 131 | 150,000 / 3,000 | 0 | From validated SPARC + verified TNG |
| `compute_galaxy_mass_table(min_points=5)` | 2,784 / 131 | 150,000 / 3,000 | 0 | No galaxies failed min-points/finite-mass checks |
| Caliper match (`±0.3 dex`, unique nearest TNG per SPARC) | 2,217 / 89 | 4,450 / 89 | SPARC: `567` pts / `42` gal; TNG: `145,550` pts / `2,911` gal | Unmatched galaxies outside caliper or already used by closer SPARC match |
| Final matched fit samples | 2,217 / 89 | 4,450 / 89 | — | Effective bins used: SPARC `15`, TNG `10` |

### Test 3: Xi organizing
| Step | SPARC (points/gal) | TNG (points/gal) | Removed | Why |
|---|---:|---:|---:|---|
| Inputs to Test 3 | 2,784 / 131 | 150,000 / 3,000 | 0 | Starting per-point sets |
| Require `>=8` points per galaxy for xi payload | 2,666 / 113 | 150,000 / 3,000 | SPARC: `118` pts / `18` gal; TNG: `0` | Xi payload gate in `per_galaxy_xi_payload` |
| Finite-value checks (`R_kpc`, `log_res`, `log_gobs`) | 2,666 / 113 | 150,000 / 3,000 | 0 | No failures |
| Valid `M_dyn` and `xi` checks | 2,666 / 113 | 150,000 / 3,000 | 0 | No failures |
| X-peak Wilcoxon subset (`n_points>=10`) | — / 98 | — / 3,000 | SPARC: `15` gal; TNG: `0` | Peak-location test requires higher per-galaxy support |

### Test 4: α* convergence
| Condition filter | SPARC galaxies kept | TNG galaxies kept | Removed and why |
|---|---:|---:|---|
| Full | 131 | 3,000 | Baseline α* tables |
| Mass-matched overlap (`m in [9.258656, 11.551354]`) | 96 | 2,896 | SPARC `-35`, TNG `-104`: outside overlap mass window |
| Resolution-matched (`n_points>=10`) | 98 | 3,000 | SPARC `-33`, TNG `-0`: below point-count threshold |
| Mass+Resolution | 81 | 2,896 | SPARC `-50`, TNG `-104`: combined filters |

### Test 5: Dataset lineage
| Stage | Points | Galaxies | Removed | Why |
|---|---:|---:|---:|---|
| Unified lineage audit coverage | 15,745 | 2,892 | 0 | Descriptive audit over all discovered sources |

## Test 1–5 metric table

| Test | Primary sample size | Key metrics |
|---|---|---|
| Test 1 (phase null) | SPARC: 2,759 points in 16 bins | `mu_peak=-9.1323855`, `Δ=0.7886145` from `log_gdagger=-9.921`, `ΔAIC=-147.7959`; shuffle A: `p(Δ)=0.631`, `p(ΔAIC)=0.057`; shuffle B: `p(Δ)=0.672`, `p(ΔAIC)=0.15` |
| Test 2 (mass-matched phase) | Matched 89 SPARC–TNG galaxy pairs | Mass KS `p=0.3953`; SPARC matched: `mu_peak=-9.8878536`, `ΔAIC=-112.7375`, `n_bins=15`; TNG matched: `mu_peak=-11.8761987`, `ΔAIC=1.6941`, `n_bins=10` |
| Test 3 (xi organizing) | SPARC: 113 gal; TNG: 3,000 gal | SPARC: `C=0.6874`, `p_perm=0.565`, `median(log10 X_peak)=-0.46875`, Wilcoxon `p=2.5356e-17`; TNG: `C=1.5489`, `p_perm=0.0`, `median(log10 X_peak)=-0.03125`, Wilcoxon `p=0.0` |
| Test 4 (α* convergence) | SPARC/TNG by condition (Full, Mass, Resolution, Mass+Resolution) | Full: SPARC `0.1175`, TNG `-0.0840` (`Δα_max=0.2015`); Mass overlap: SPARC `0.0956`, TNG `0.0385` (`Δα_max=0.0571`); Resolution: SPARC `0.1149`, TNG `-0.0840` (`Δα_max=0.1988`); Mass+Resolution: SPARC `0.0907`, TNG `0.0385` (`Δα_max=0.0522`) |
| Test 5 (lineage audit) | All unified sources | SPARC lineage summary: `n=2784 pts / 131 gal`, `sigma_res median=0.072374`, `IQR=0.0840575`, `logMh range=[0.0,13.2]`; contamination note status `present` with 30 historical ratio values (`min=0.896529`, `max=1.4519`) |

## Key plots and diagnosis target
- `fig_phase_null.png`: Diagnoses whether observed phase-peak proximity (`Δ`) and model preference (`ΔAIC`) are unusual versus shuffle nulls A/B.
- `fig_mass_matched_phase.png`: Diagnoses phase-profile behavior after strict mass matching and verifies matched mass distributions are comparable.
- `fig_xi_organizing.png`: Diagnoses whether variance concentrates near `X=R/xi ~ 1` and whether concentration `C` exceeds permutation null.
- `fig_alpha_star.png`: Diagnoses α* convergence/divergence between SPARC and TNG under mass and/or resolution matching.

## Robust vs sensitive (explicit)

| Choice | Evidence from this run | Assessment |
|---|---|---|
| Bin count / effective bin support | Full SPARC Test 1 uses 16 bins and gives `mu_peak=-9.132`; matched SPARC uses 15 bins with `mu_peak=-9.888`; matched TNG uses 10 bins with very wide CI and `ΔAIC` crossing non-preference | **Sensitive** (effective bin support changes are associated with large parameter shifts; effect is not isolated from sample-composition changes) |
| N-cut thresholds | Test 3 `>=8` removes 18/131 SPARC galaxies (none in TNG). Test 4 `n_points>=10` removes 33/131 SPARC galaxies (0/3000 TNG) and shifts SPARC α* from `0.117` (Full) to `0.091` (Mass+Resolution) | **Mixed**: **sensitive for SPARC**, **robust for TNG** in this dataset |
| Mass matching | Test 2 matching reduces SPARC to 89 galaxies and shifts phase metrics strongly (`mu_peak -9.132 -> -9.888`; `ΔAIC -147.8 -> -112.7`); TNG matched phase differs strongly (`mu_peak=-11.876`, `ΔAIC=+1.694`) | **Sensitive** |

### Additional robust findings
- Determinism is robust: rerunning Test 1 with identical seeds reproduced all tracked summary metrics exactly.
- Null conclusion for `Δ` is robust across shuffle types in Test 1 (both `p(Δ) > 0.6`).
