# BEC Dark Matter — Master Progress Map
## 37 Tests Across 6 Datasets, ~33,000 Galaxies

**Author:** Russell Licht
**Date:** February 2026
**Status:** Active research
**Core claim:** The Radial Acceleration Relation (RAR) is algebraically identical to the Bose-Einstein occupation number: `g_DM/g_bar = 1/(exp(sqrt(g_bar/g†)) - 1)`, implying dark matter is a quantum condensate with universal scale g† = 1.2×10⁻¹⁰ m/s².

---

## Quick Scorecard

| Category | Tests | BEC Support | Against BEC | Inconclusive |
|----------|-------|-------------|-------------|--------------|
| Environmental scatter | 7 | 4 (universal σ) | 0 | 3 |
| Inversion stability | 5 | 5 (rock-solid) | 0 | 0 |
| Quantum signatures | 5 | 1 | 3 | 1 |
| Simulation cross-val | 8 | — | — | see below |
| Healing length / ACF | 7 | 2 | 1 | 4 |
| Multi-scale / external | 5 | 1 | 1 | 3 |
| **TOTAL** | **37** | **13** | **5** | **11** |

**+ 8 simulation tests that rewrite the narrative** (see Section F)

---

## CRITICAL UPDATE: The Simulation Bombshell

The most important finding of the entire project came last (Tests #33-35):

> **When EAGLE galaxies are analyzed with galaxy-adapted radii (R_half, 2×R_half) instead of fixed apertures (1-100 kpc), the "missing" inversion at g† appears at 95-99.7% bootstrap confidence — matching SPARC and TNG.**

This means:
- The scatter inversion at g† is **universal** — it appears in observations (SPARC), SPH simulations (EAGLE), and moving-mesh simulations (TNG)
- The original "EAGLE has no inversion" (0.9%) was a **method artifact** of fixed aperture radii
- The inversion is a **structural feature** of galaxy mass profiles in ΛCDM, not evidence for new physics
- EAGLE's fitted g† with adapted radii = **1.27×10⁻¹⁰** m/s² — matches the observed 1.2×10⁻¹⁰ within 0.023 dex

This does NOT kill the BEC theory, but it removes the scatter inversion as a discriminant. The mathematical identity between the RAR and Bose-Einstein statistics remains unexplained.

---

## The Theory in 60 Seconds

The RAR (McGaugh+2016) relates observed gravitational acceleration to baryonic acceleration:
```
g_obs = g_bar / (1 - exp(-sqrt(g_bar/g†)))
```

Rearranging to isolate the dark matter contribution:
```
g_DM/g_bar = 1 / (exp(sqrt(g_bar/g†)) - 1)
```

This is **exactly** the Bose-Einstein occupation number `n̄(ε) = 1/(exp(ε) - 1)` with energy parameter `ε = sqrt(g_bar/g†)`. If this identity is physical (not coincidental):
- g† is the condensation temperature
- Below g†: DM condenses (quantum regime, flat rotation curves)
- Above g†: DM thermally suppressed (classical regime, Newtonian)
- Variance follows quantum bunching: σ² ∝ n̄(n̄+1), not Poisson σ² ∝ n̄
- Healing length ξ = sqrt(GM/g†) sets coherence scale
- Phase transition at g† should produce observable scatter minimum

---

## Datasets Used

| Dataset | Galaxies | Points | Source |
|---------|----------|--------|--------|
| SPARC | 175 (131 quality) | 2,777 | Lelli+2016, 3.6μm rotation curves |
| PROBES | 3,163 (1,506 non-SPARC) | 45,176 | Stone+2021, heterogeneous RCs |
| ALFALFA × Yang DR7 | 25,255 (2,273 matched) | N/A | BTFR scatter test |
| EAGLE RefL0100N1504 | 29,716 (7,119 matched) | 71,189 | Schaye+2015, SPH simulation |
| TNG100-1 | 21,052 | 59,291 | Pillepich+2018, AREPO simulation |
| PHANGS | 56 | 1,547 | CO rotation curves, inner radii |
| Brouwer KiDS-1000 | 259,886 lenses | 16 bins | Weak lensing RAR |
| Tian+2020 clusters | 20 | 84 | Galaxy cluster RAR |

---

## A. Environmental Scatter Tests (7 tests)

The BEC theory predicts universal scatter (no environmental dependence) because g† is a fundamental constant. CDM predicts environment-dependent scatter from halo assembly bias.

### A1. Definitive 7-Test Battery
- **File:** `test_env_scatter_definitive.py` → `summary_env_definitive.json`
- **Sample:** 131 SPARC galaxies (48 dense, 83 field), SPARC distances, Haubner+2025 errors
- **Tests:** Levene, Brown-Forsythe, bootstrap, permutation, F-test, acceleration-binned, block
- **Result:** All 4 primaries p > 0.05 (Levene p = 0.64, 0.76, 0.45, 0.74)
- **Verdict: BEC-CONSISTENT** (universal scatter, no environmental difference)
- **Key discovery:** Prior CF4 result (p < 0.001) was artifact of UGC06786 incorrect distance (9.88 vs 29.3 Mpc)

### A2. Acceleration-Binned Environmental Scatter
- **File:** `test_env_cf4_accel_binned.py` → `summary_env_cf4_accel_binned.json`
- **Result:** σ_dense = 0.165, σ_field = 0.171, p = 0.64 across all regimes
- **Verdict: BEC-CONSISTENT**

### A3. Confound Control (Distance, Morphology, Inclination)
- **File:** `test_env_confound_control.py` → `summary_env_confound_control.json`
- **Result:** Distance imbalanced (field 30 Mpc vs dense 12 Mpc); after control, no signal remains
- **Verdict: BEC-CONSISTENT** (environmental "signal" was distance artifact)

### A4. Propensity-Score Matched Pairs
- **File:** `test_propensity_matched_env.py` → `summary_propensity_matched_env.json`
- **Result:** 29 matched pairs, Δσ = +0.008 (p = 0.12), Δσ = -0.007 (p = 0.17)
- **Verdict: BEC-CONSISTENT** (no difference in matched sample)

### A5. Monte Carlo Distance Error Injection
- **File:** `test_mc_distance_and_inversion.py` → `summary_mc_distance_and_inversion.json`
- **Result:** Environmental delta preserved across 5-30% error injections; inversion at -9.856
- **Verdict: BEC-CONSISTENT** (result robust to distance uncertainties)

### A6. Void-Field-Dense Gradient
- **File:** `test_void_gradient.py` → `summary_void_gradient.json`
- **Result:** Only 10 void galaxies; no monotonic gradient; void σ inconsistent between metrics
- **Verdict: INCONCLUSIVE** (too few void galaxies)

### A7. ALFALFA × Yang DR7 BTFR Scatter
- **File:** `test_alfalfa_yang_btfr.py` → `summary_alfalfa_yang_btfr.json`
- **Sample:** 2,273 galaxies (1,917 field, 356 dense)
- **Result:** Dense has HIGHER scatter at low W50 (p < 0.05), no difference at high W50
- **Verdict: INCONCLUSIVE** (complex pattern, doesn't match simple BEC prediction)

---

## B. Inversion Point Stability Tests (5 tests)

The scatter derivative of the RAR shows a minimum (inversion) near g†. These tests ask: is this robust?

### B1. Binning Robustness
- **File:** `test_binning_robustness.py` → `summary_binning_robustness.json`
- **Result:** 25/25 configurations find inversion within ±0.20 dex of g†; mean = -9.89, std = 0.04
- **Verdict: STRONGLY STABLE**

### B2. Leave-One-Out Jackknife
- **File:** `test_jackknife_robustness.py` → `summary_jackknife_robustness.json`
- **Result:** 131 iterations; delta flips sign 4/131 times; max shift 0.065 dex
- **Verdict: ROBUST** (no single galaxy drives the result)

### B3. Split-Half Replication (1,000×)
- **File:** `test_split_half_replication.py` → `summary_split_half_replication.json`
- **Result:** 95.65% of random splits find inversion within ±0.20 dex in BOTH halves
- **Verdict: STRONGLY REPLICATED**

### B4. Four-Method Nonparametric Test
- **File:** `test_nonparametric_inversion.py` → `summary_nonparametric_inversion.json`
- **Methods:** RAR parametric, LOESS, cubic spline, isotonic regression × 9 binning configs
- **Result:** All 36 configurations within ±0.20 dex of g†; crossing = -9.971 ± 0.001
- **Verdict: MODEL-INDEPENDENT** (inversion is real, not an artifact of fitting method)

### B5. PROBES External Replication
- **File:** `test_probes_inversion_replication.py` → `summary_probes_inversion_replication.json`
- **Sample:** 1,506 non-SPARC galaxies, 45,176 points
- **Result:** Inversion at -11.12 (1.20 dex from g†); 0/8 configs within ±0.20 dex
- **Verdict: NOT CONFIRMED** (PROBES uses stars-only g_bar — no gas correction available)

---

## C. Quantum Signature Tests (5 tests)

Direct tests of whether DM behaves quantum-mechanically at scales predicted by BEC theory.

### C1. Forward-Model Soliton vs NFW Per-Galaxy Fits
- **File:** `test_forward_model_bunching.py` → `summary_forward_model_bunching.json`
- **Sample:** 113 SPARC galaxies, differential_evolution optimizer
- **Result:** NFW preferred 53% (ΔAIC < -2), BEC preferred 34%; mean ΔAIC = -20.3; no mass trend (ρ = -0.05, p = 0.59)
- **Verdict: FAILS BEC** (NFW wins; no mass-dependent preference for soliton cores)
- **Caveat:** Soliton-only model lacks NFW-like envelope present in real BEC halos

### C2. PHANGS Inner Quantum Regime (X < 1)
- **File:** `test_phangs_inner_quantum.py` → `summary_phangs_inner_quantum.json`
- **Sample:** 56 PHANGS galaxies, 1,547 points at X = R/ξ < 1
- **Result:** Inner deficit of -0.181 dex (p < 10⁻⁶) — **opposite** of BEC prediction
- **Verdict: FAILS BEC** (expected inner excess from soliton core)
- **Caveat:** Exponential disk approximation poor in bulge-dominated inner regions; needs resolved photometry

### C3. Within-Galaxy Radial Variance Decomposition
- **File:** `test_radial_variance_profile.py` → `summary_radial_variance_profile.json`
- **Result:** Inner (R < ξ) ΔAIC = -27 (classical preferred); outer ΔAIC = +3
- **Verdict: FAILS BEC** (opposite trend to prediction)

### C4. BEC Exponent α vs Total Mass
- **File:** `test_alpha_with_mtotal.py` → `summary_alpha_mtotal.json`
- **Result:** Global α ≈ 1.06; weak mass trend
- **Verdict: MARGINAL** (α close to BEC prediction of 1.0 but not definitive)

### C5. Phase Diagram Model
- **File:** `test_phase_diagram_model.py` → `summary_phase_diagram_model.json`
- **Result:** SPARC data follows predicted phase boundary
- **Verdict: BEC-CONSISTENT**

---

## D. Healing Length / Coherence Tests (7 tests)

Tests whether the predicted BEC healing length ξ = sqrt(GM/g†) leaves observable imprints in rotation curve residuals.

### D1. Radial Autocorrelation of RAR Residuals
- **File:** `test_interface_oscillation.py` → `summary_interface_oscillation.json`
- **Sample:** 67 SPARC galaxies (≥15 points each)
- **Result:** Mean lag-1 ACF = 0.700 (SE 0.023), 100% positive, p = 2.3×10⁻⁴¹
- **Verdict: STRONG POSITIVE AUTOCORRELATION** — residuals have coherent radial structure

### D2. Autocorrelation Controls (Demeaning + Detrending)
- **File:** `test_interface_oscillation_controls.py` → `summary_interface_oscillation_controls.json`
- **Result:** Survives per-galaxy demeaning, first-differencing, spline detrending, permutation null
- **Verdict: GENUINE STRUCTURE** (not artifact of trends or galaxy-level offsets)

### D3. Lomb-Scargle Spectral Test
- **File:** `test_interface_spectral.py` → `summary_interface_spectral_test.json`
- **Result:** 25/67 (37.3%) galaxies show significant periodicity vs 3.4 expected; median wavelength 6.23 kpc
- **Verdict: EXCESS PERIODICITY** (p = 6.3×10⁻¹⁶ vs null)

### D4. Baryon Wiggle vs Coherence Control
- **File:** `test_baryon_wiggle_vs_coherence.py` → `summary_baryon_wiggle_vs_coherence.json`
- **Result:** Baryonic wiggle metrics do NOT predict ACF (all p > 0.15); spectral AR1 vs ξ survives controls
- **Verdict: INTERFACE PHYSICS** (ACF not explained by baryonic structure alone)

### D5. Healing Length Lc Scaling
- **File:** `test_healing_length_scaling.py` → `summary_healing_length_scaling.json`
- **Result:** ρ(Lc, ξ) = 0.551 (p = 10⁻⁶) BUT ρ(Lc, R_ext) = 0.591; normalized: ρ = 0.147 (p = 0.23)
- **Verdict: CONFOUNDED BY SIZE** (cannot isolate ξ scaling from galaxy extent)

### D6. Healing Length k* (Lag-Index Coherence)
- **File:** `test_healing_length_kstar.py` → `summary_healing_length_kstar.json`
- **Result:** ρ(k*, ξ) = 0.278 (p = 0.023) but k* confounded with N_pts (ρ = 0.487)
- **Verdict: CONFOUNDED** (marginal at best)

### D7. Healing Length ACF + Distance Diagnostics
- **Files:** `test_healing_length_acf.py`, `test_healing_length_distance_diag.py`
- **Result:** r1 vs ξ/R partial ρ = 0.226 (p = 0.066, not BH-FDR significant); signal vanishes in clean subsample (p = 0.64)
- **Verdict: EDD ARTIFACT** (ξ/R→ACF correlation partially driven by distance quality)

---

## E. Multi-Scale and External Tests (3 tests)

### E1. Brouwer+2021 KiDS-1000 Weak Lensing RAR
- **File:** `test_brouwer_lensing_rar.py` → `summary_brouwer_lensing_rar.json`
- **Sample:** 259,886 lenses, 16 subsamples, R > 35 kpc
- **Result:** Environment matters MORE at high mass (Δσ = 0.06-0.08); soliton core ξ = 1.7-9.6 kpc unresolved
- **Verdict: BEC-CONSISTENT** (environment signal at high mass, core unresolved)

### E2. Tian+2020 Galaxy Cluster RAR
- **File:** `test_cluster_rar_tian2020.py` → `summary_cluster_rar_tian2020.json`
- **Sample:** 20 clusters, 84 points
- **Result:** Best-fit a₀ = 1.73×10⁻⁹ = 14.4× g†; galaxy g† rejected (χ² = 3,569)
- **Verdict: CONSTRAINS THEORY** (clusters have distinctly higher acceleration scale)

### E3. Literature Cross-Reference
- **File:** `literature_crossref_tests.py` → `summary_literature_crossref.json`
- **Result:** Consistent with McGaugh+2016, Lelli+2017, Desmond+2017
- **Verdict: CONSISTENT** with established results

### E4. Cluster σ Scaling — Does g‡ ∝ σ²?
- **File:** `test_cluster_sigma_scaling.py` → `summary_cluster_sigma_scaling.json`
- **Sample:** 14 CLASH clusters with both per-cluster g‡ and σ (8 spectroscopic, 6 virial from M200/r200)
- **BEC prediction:** If g† is a condensation temperature with T ∝ σ², then log(g‡) = A + B × log(σ²) with B ≈ 1
- **Result:** B = 0.014 ± 0.229 (full), B = 0.16 ± 0.39 (clean, N=11). Pearson r = 0.018, p = 0.95. Bootstrap 95% CI for B: [-0.33, 0.60]. B=1 NOT in CI. g‡/[g† × (σ/σ_gal)²] = 0.31 ± 0.17
- **Verdict: AGAINST simplest BEC σ²-scaling** — g‡ is ~constant across clusters (consistent with Tian+2020 universality finding), not proportional to σ². The acceleration scale at cluster scale appears to be set by something other than velocity dispersion alone.

### E5. Cluster Core Acceleration & Multivariate Analysis — What Controls g‡?
- **File:** `test_cluster_gcore_scaling.py` → `summary_cluster_gcore_A3.json`
- **Diagnostics (from test_cluster_sigma_diagnostics.py):**
  - g‡ intrinsic variance = 93.8% of observed (σ_int = 0.157 dex) — REAL variation, not noise
  - **BCG aperture total mass Mtot** is the strongest univariate predictor: r = +0.578, p = 0.008, N=20
  - BCG stellar mass M* marginal: r = +0.424, p = 0.062
  - σ² is dead last: r = +0.018, p = 0.95
- **g_core = G M_tot / Rad²** (direct core acceleration):
  - Univariate: B = 0.14 ± 0.17, r = +0.19, p = 0.43 (not significant — variable aperture Rad adds noise)
  - Using fixed 14 kpc: equivalent to log Mtot predictor → r = +0.58, p = 0.008
  - g‡ ≈ 0.50× g_core (median), range 0.15–1.6×
- **Multivariate (N=9 clusters with g_core + c_200 + g_200):**
  - **R² = 0.749** (adjusted R² = 0.598) — explains 75% of g‡ variance!
  - **g_core: p = 0.026**, g_200: p = 0.033, c_200: not significant (p = 0.14)
  - Best bivariate: **g_core + g_200** (R² = 0.59, best AIC)
  - g‡ is jointly determined by core mass AND virial-scale gravitational field
- **σ anti-correlation:**
  - Clean spectroscopic N=5: r = -0.91, p = 0.034 — but fragile (driven by aperture size variation in RXJ2248)
  - In full N=14 sample: partial r(g‡, σ² | g_core) = +0.10, p = 0.73 — vanishes completely
- **Verdict: ILLUMINATING** — g‡ is NOT controlled by velocity dispersion but IS jointly controlled by the core mass profile (g_core) and the virial potential (g_200). The 14× enhancement of g‡ over g† reflects where galaxy clusters sit on the mass-acceleration plane — their cores are denser and their potentials are deeper. The multivariate R²=0.75 leaves only 25% unexplained variance.

---

## F. Simulation Cross-Validation Tests (8 tests)

**This is where the narrative changes fundamentally.**

### F1. EAGLE RAR Inversion v1 (Unit Bug)
- **File:** `test_eagle_rar_inversion.py` → `summary_eagle_rar_inversion.json`
- **Sample:** 29,716 galaxies, 294,694 points, 10 fixed apertures
- **Result:** Inversion at -4.79 (5.13 dex off) — UNIT BUG in mass conversion
- **Verdict: SUPERSEDED by F2**

### F2. EAGLE RAR Inversion v2 (Corrected)
- **File:** `test_eagle_rar_inversion_v2.py` → `summary_eagle_rar_inversion_v2.json`
- **Sample:** 12,294 EAGLE galaxies, 56,105 points, 10 fixed apertures
- **Result:** Bootstrap 0.9% near g† (SPARC: 99.9%); galaxy-level inversion at -11.38
- **Verdict: DISCRIMINATING** — EAGLE ≠ SPARC (this was the key prior result)

### F3. EAGLE 6-Test Pipeline (A, B, P1-P4)
- **File:** `eagle_bec_pipeline.py` → `pipeline_summary.json`
- **Tests:**
  - A (Pooled scatter inversion): PASS — EAGLE no inversion near g†
  - B (Galaxy-level inversion): PASS — EAGLE 0.9% vs SPARC 99.9%
  - P1 (α variance vs disturbance): CONFIRMED — disturbed → more α variance
  - P2 (α-depth slope): REFUTED — steeper for disturbed (opposite prediction)
  - P3 (g† vs disturbance): CONFIRMED — g† shifts with disturbance
  - P4 (BEC form minimizes residuals): CONFIRMED — BEC_α best BIC, α = 1.028
- **Verdict: 5/6 PASS** (P2 refuted)

### F4. TNG100-1 Phase 1 (Stellar-Only, Fixed Apertures)
- **File:** `tng_rar_analysis.py` → `tng_results.json`
- **Sample:** 21,052 galaxies, 84,208 points, 4 fixed apertures (5/10/30/100 kpc)
- **Result:** g† = 3.16×10⁻¹¹ (offset -0.58 dex); bootstrap 27.2% near g†; α = 1.20
- **Verdict: INTERMEDIATE** — TNG sits between SPARC (99.9%) and EAGLE (0.9%)

### F5. TNG Phase 1B: Stars+Gas with Group Catalog Radii ★
- **File:** `tng_phase1b_gas.py` → `phase1b_results.json`
- **Sample:** 20,951 galaxies, 59,291 points, adapted radii (R_half, 2×R_half, R_Vmax)
- **Star-only results:**
  - g† = 2.92×10⁻¹⁰ (log = -9.535); scatter = 0.072 dex
  - Bootstrap: 99.9% (bw=0.15), 89.9% (bw=0.30); median = -9.943
  - α (fitted g†) = 1.012
- **Stars+gas results:**
  - g† = 6.10×10⁻¹¹ (log = -10.214); scatter = 0.075 dex
  - Bootstrap: **100%** at ALL bin widths; median = -9.956; 95% CI [-10.04, -9.79]
  - α (observed g†) = 1.094
- **Verdict: INVERSION SURVIVES AND STRENGTHENS** with complete baryonic mass

### F6. EAGLE Adapted Radii: Fixed vs Galaxy-Adapted ★★★
- **File:** `eagle_adapted_radii.py` → `eagle_adapted_results.json`
- **Sample:** 7,119 galaxies matched to R_half, same galaxies tested both ways
- **Fixed aperture (10 radii, 1-100 kpc):**
  - g† = 6.97×10⁻¹¹; scatter = 0.115; bootstrap 6.2% near g† (bw=0.30)
- **Adapted radii (R_half, 2×R_half):**
  - g† = **1.27×10⁻¹⁰** (offset +0.023 dex from observed — essentially perfect)
  - Scatter = **0.071 dex**
  - Bootstrap: **99.7%** (bw=0.15), **99.3%** (bw=0.20), **95.0%** (bw=0.30)
  - Median inversion = **-9.887** (SPARC: -9.89)
- **Verdict: THE ORIGINAL 0.9% WAS A METHOD ARTIFACT.** EAGLE shows the inversion when analyzed with adapted radii.

### Summary Table: Simulation Results

| Metric | SPARC | EAGLE fixed | EAGLE adapted | TNG star-only | TNG star+gas |
|--------|-------|-------------|---------------|---------------|--------------|
| Radii | 15-50 var | 10 fixed | 2 adapted | 2-3 adapted | 2-3 adapted |
| g† fitted | 1.20e-10 | 6.97e-11 | **1.27e-10** | 2.92e-10 | 6.10e-11 |
| Scatter | 0.12 | 0.115 | **0.071** | 0.072 | 0.075 |
| Inv. % (bw=0.15) | 99.9% | 62.6% | **99.7%** | 99.9% | **100%** |
| Inv. % (bw=0.30) | 99.9% | 6.2% | **95.0%** | 89.9% | **100%** |
| Inv. median | -9.89 | -10.37 | **-9.887** | -9.943 | -9.956 |
| α (obs. g†) | — | 1.056 | 1.275 | 1.276 | 1.094 |

---

## G. What Still Stands

After 35 tests, here is what is empirically established:

### Confirmed (robust across methods):
1. The RAR scatter has a minimum (inversion) near log g = -9.9 — in SPARC, EAGLE, and TNG
2. The inversion is robust to binning, smoothing method, jackknife, and split-half replication
3. RAR residuals have strong radial coherence (ACF lag-1 = 0.70, 100% positive)
4. 37% of galaxies show significant periodicity in RAR residuals (λ ≈ 6 kpc)
5. Environmental scatter is universal (no field vs cluster difference) in SPARC
6. The BEC functional form (occupation number) provides the best fit to simulation RAR residuals (BIC test P4)

### Refuted or inconclusive:
1. The scatter inversion is NOT a discriminant between ΛCDM and observations — ΛCDM simulations produce it
2. Soliton core fits are NOT preferred over NFW (ΔAIC = -20.3 favoring NFW)
3. Inner quantum regime (X < 1) shows deficit, not excess
4. Healing length scaling is confounded by galaxy size
5. ξ/R → ACF correlation may be a distance-quality artifact
6. Cluster-scale RAR has a distinctly different acceleration parameter (14.4× g†)

### The open puzzle:
The mathematical identity between the RAR and Bose-Einstein statistics remains **unexplained**. ΛCDM simulations reproduce the RAR (including the scatter inversion at g†), which means either:
- (a) The BEC interpretation is a mathematical coincidence and ΛCDM naturally produces RAR-like relations from baryonic feedback
- (b) The BEC statistics emerge from the ΛCDM framework itself (the condensate IS the CDM halo in some limit)
- (c) Both ΛCDM simulations and observations are approximating the same underlying quantum physics

No current test distinguishes between these three possibilities.

---

## File Locations

| Directory | Contents |
|-----------|----------|
| `/Users/russelllicht/bec-dark-matter/analysis/pipeline/` | All 60+ test scripts |
| `/Users/russelllicht/bec-dark-matter/analysis/results/` | All `summary_*.json` result files |
| `/Users/russelllicht/bec-dark-matter/data/sparc/` | SPARC galaxy data |
| `/Users/russelllicht/bec-dark-matter/data/eagle_rar/` | EAGLE aperture masses (29,716 galaxies) |
| `/Users/russelllicht/eagle_bec_results/` | EAGLE BEC pipeline + adapted radii results |
| `/Users/russelllicht/Desktop/tng_cross_validation/` | TNG Phase 1 + Phase 1B results |
| `/Users/russelllicht/Desktop/tng_cross_validation/tng_gas_comparison/` | TNG stars+gas comparison |
| `/Users/russelllicht/eagle_bec_results/eagle_adapted_results/` | EAGLE fixed vs adapted radii |

---

## Chronological Narrative

1. **Foundation:** Built RAR from SPARC 175 galaxies; confirmed BEC functional form fits with α ≈ 1.0
2. **Environmental test:** Found field > cluster scatter at 99.8% confidence — then discovered it was a distance artifact (UGC06786)
3. **Definitive environmental battery (7 tests):** Universal scatter confirmed; BEC-consistent but not discriminating
4. **Inversion stability (5 tests):** The scatter minimum at g† is real, robust, and model-independent
5. **Quantum signatures (5 tests):** Mixed — soliton cores not preferred, inner regime fails, but BEC form fits best
6. **Healing length series (7 tests):** Strong radial coherence and periodicity exist, but ξ scaling confounded by size
7. **EAGLE simulation (v1→v2):** Fixed aperture analysis shows 0.9% inversion — appears to discriminate ΛCDM from observations
8. **TNG simulation (Phase 1→1B):** Group catalog radii + gas → 100% inversion at g†; feature strengthens with baryonic completeness
9. **EAGLE adapted radii:** The bombshell — EAGLE shows 95-99.7% inversion with adapted radii, matching SPARC and TNG. The "discriminant" was a method artifact.

---

*This document is self-contained and can be shared with external AI systems for context.*
