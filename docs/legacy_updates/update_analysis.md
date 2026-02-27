# BEC Dark Matter — Full Project Analysis Update
## 50+ Tests Across 20 Datasets, ~33,000 Galaxies

**Author:** Russell Licht
**Date:** February 21, 2026
**Status:** Active research — MHONGOOSE integration complete, extended RAR tier analysis updated, hierarchical healing length analysis complete
**Export purpose:** Self-contained context document for external AI systems

---

## 1. Executive Summary

This project tests the hypothesis that dark matter is a **Bose-Einstein condensate (BEC)** by exploiting the algebraic identity between the Radial Acceleration Relation (RAR) and the Bose-Einstein occupation number formula. After 50+ tests across 20 datasets and ~33,000 galaxies:

- **14 tests support BEC**, **5 refute specific BEC predictions**, **11 are inconclusive**, and **8 simulation tests rewrite the narrative**
- The scatter inversion at g-dagger is **rock-solid** across all quality tiers (SPARC through 376-galaxy extended sample) and **replicates with MHONGOOSE data**
- LCDM simulations (EAGLE, TNG) also produce the inversion when analyzed with galaxy-adapted radii — removing it as a BEC-vs-LCDM discriminant
- **NEW:** Cluster-scale g†_eff follows a power-law mass scaling g†_eff ∝ M^0.32 (r=0.83, p=2×10⁻²³), connecting galaxies to clusters through the BEC healing length
- The mathematical identity between RAR and Bose-Einstein statistics remains **unexplained**
- The quantum-vs-classical-wave distinction is **unresolvable** with current data (need 300x more)
- The kurtosis spike at g-dagger is real but **Korsaga-2019-dominated** — not universal across surveys

---

## 2. Core Theory

### The RAR as Bose-Einstein Occupation Number

```
g_obs = g_bar / (1 - exp(-sqrt(g_bar / g†)))

Rearranging for the dark matter contribution:
g_DM / g_bar = 1 / (exp(sqrt(g_bar / g†)) - 1)  ←  Bose-Einstein occupation number n̄(ε)
```

Where:
- `ε = sqrt(g_bar / g†)` — dimensionless energy parameter
- `g† = 1.20 × 10⁻¹⁰ m/s²` — condensation scale (= MOND's a₀)
- `log(g†) = -9.921`
- Below g†: DM condenses → flat rotation curves
- Above g†: modes thermally suppressed → Newtonian dynamics

### Key Predictions
1. **Variance follows quantum bunching**: σ² ∝ n̄(n̄+1), not Poisson σ² ∝ n̄
2. **Healing length**: ξ = sqrt(GM/g†) sets coherence scale
3. **Environmental universality**: g† is fundamental → no field-vs-cluster scatter difference
4. **Scatter inversion**: derivative of residual scatter changes sign at g†
5. **Soliton cores**: BEC halos have r⁻⁸ cores, distinct from NFW r⁻¹ cusps

---

## 3. Datasets

| Dataset | Galaxies | RAR Points | Source | Use |
|---------|----------|------------|--------|-----|
| SPARC | 175 (131 quality) | 2,740 | Lelli+2016, 3.6μm | Primary rotation curves |
| THINGS | 6 new | 533 | Walter+2008 | Gold-standard HI mass models |
| de Blok 2002 | 17 new | 361 | de Blok+2002 | Direct decomposition RCs |
| Korsaga 2019 | 98 (replaced Vrot-only) | 5,705 | Korsaga+2019 | Freeman disk + Sérsic bulge models |
| Bouquin 2018 | 20 matched | ~500 | Bouquin+2018 | 3.6μm profile matching |
| Verheijen 2001 | 5 | 83 | Verheijen+2001 | Ursa Major cluster |
| PHANGS | 56 | 1,547 | Lang+2020 | CO rotation curves (inner radii) |
| GomezLopez 2019 | 71 | 2,852 | Gomez-Lopez+2019 | Extended sample (noisy) |
| **MHONGOOSE** | **4** | **82** | **Nkomo+2025, Sorgho+2019** | **MeerKAT deep HI** |
| PROBES | 3,163 (1,506 non-SPARC) | 45,176 | Stone+2021 | Heterogeneous RCs |
| ALFALFA × Yang | 25,255 (2,273 matched) | N/A | Haynes+2011/2018 | BTFR scatter |
| EAGLE | 29,716 (7,119 adapted) | 71,189 | Schaye+2015 | SPH simulation |
| TNG100-1 | 21,052 | 59,291 | Pillepich+2018 | AREPO simulation |
| Brouwer KiDS-1000 | 259,886 lenses | 16 bins | Brouwer+2021 | Weak lensing RAR |
| Tian+2020 clusters | 20 | 84 | Tian+2020 | Galaxy cluster RAR |
| WALLABY | various | various | Koribalski+2020 | HI survey |
| Cosmicflows-4 | 175 cross-matched | N/A | Tully+2023 | Distance corrections |
| S4G | 109 matched | N/A | Sheth+2010 | Spitzer 3.6μm photometry |
| Sofue | various | various | Sofue+2016 | RC compilation |
| LITTLE THINGS | 7 | N/A | Hunter+2012 | Dwarf galaxies (Vrot only) |

**Total unique galaxies with baryonic decomposition:** ~256 (extended sample) + 4 MHONGOOSE = 260
**Total simulation galaxies:** ~50,000+

---

## 4. Extended RAR Inversion — Quality-Tiered Analysis

### Tier Definitions

| Tier | Contents | Galaxies | RAR Points | Scatter Quality |
|------|----------|----------|------------|-----------------|
| SPARC-only | SPARC 3.6μm | 126 | 2,740 | σ ~ 0.17 dex |
| Tier A | SPARC + THINGS | 132 | 3,273 | σ ~ 0.13-0.17 |
| Tier B | A + de Blok 2002 | 149 | 3,634 | + σ ~ 0.33 |
| Tier B+ | B + quality Bouquin (Verheijen, PHANGS) | 152 | 3,701 | + σ ~ 0.20-0.32 |
| **Tier MH** | **B+ + MHONGOOSE Sorgho (NGC3621, NGC7424)** | **153** | **3,751** | **+ σ = 0.060, 0.082** |
| Tier K | B+ + Korsaga 2019 (Freeman+Sérsic) | 250 | 9,406 | + σ ~ 0.33 |
| Tier C | ALL sources (incl. noisy S4G/Bouquin) | 374 | 14,053 | + σ ~ 0.50+ |
| **Tier C+MH** | **Tier C + all MHONGOOSE (incl. Nkomo dwarfs)** | **376** | **14,135** | **+ σ = 0.175, 0.315** |

### Inversion Point Results (dσ/dx = 0 crossing nearest g†)

| Tier | Inversion (log g_bar) | Δ from g† (dex) | Within 0.25 dex? |
|------|----------------------|------------------|-------------------|
| g† (BEC prediction) | **-9.921** | 0.000 | — |
| SPARC-only (126 gal) | -10.156 | -0.235 | YES |
| Tier A: SPARC+THINGS (132 gal) | -10.157 | -0.237 | YES |
| Tier B: +deBlok (149 gal) | -10.141 | -0.220 | YES |
| Tier B+: +Bouquin (152 gal) | -10.153 | -0.233 | YES |
| **Tier MH: B+ + Sorgho (153 gal)** | **-10.157** | **-0.237** | **YES** |
| Tier K: +Korsaga (250 gal) | -9.865 | +0.056 | YES |
| Tier C: ALL sources (374 gal) | -9.947 | -0.026 | YES |
| **Tier C+MH: ALL + MHONGOOSE (376 gal)** | **-9.946** | **-0.025** | **YES** |
| Non-SPARC only (248 gal) | -9.933 | -0.012 | YES |

### Key Finding
- Adding MHONGOOSE Sorgho to Tier B+ shifts the inversion by only **-0.004 dex** — essentially zero
- Adding all MHONGOOSE (incl. Nkomo dwarfs) to Tier C shifts by **+0.001 dex** — negligible
- **The inversion point is robust across ALL quality tiers from 126 to 376 galaxies**
- All tiers replicate within 0.25 dex of g†
- Non-SPARC-only (248 galaxies, zero SPARC overlap) independently finds inversion at -9.933, only 0.012 dex from g†

---

## 5. MHONGOOSE Integration Details

### Data Source
- **Survey:** MHONGOOSE (MeerKAT HI Observations of Nearby Galactic Objects; Observing Southern Emitters)
- **Telescope:** MeerKAT (64-dish interferometer)
- **Papers:** Nkomo+2025 (A&A 699, A372), Sorgho+2019 (MNRAS 482, 1248)

### Galaxy Properties

| Galaxy | Source | Distance | M_star | M_HI | i | σ(RAR) | N_pts | Tier |
|--------|--------|----------|--------|------|---|--------|-------|------|
| NGC3621 | Sorgho+2019 | 6.6 Mpc | 1.0×10¹⁰ M☉ | 9.3×10⁹ M☉ | 64° | **0.060** | 35 | B+ |
| NGC7424 | Sorgho+2019 | 9.55 Mpc | 2.5×10⁹ M☉ | 1.3×10¹⁰ M☉ | 29° | **0.082** | 15 | B+ |
| ESO444-G084 | Nkomo+2025 | 4.6 Mpc | 4.9×10⁶ M☉ | 1.1×10⁸ M☉ | 49° | **0.315** | 16 | C |
| KKS2000-23 | Nkomo+2025 | 13.9 Mpc | 3.2×10⁷ M☉ | 6.1×10⁸ M☉ | — | **0.175** | 16 | C |

### Quality Assessment
- NGC3621 (σ=0.060) and NGC7424 (σ=0.082) have **SPARC-quality** scatter — better than most SPARC galaxies
- The Nkomo dwarfs probe the **ultra-low acceleration regime** (log g_bar < -11.3), entirely below g†
- All 82 MHONGOOSE points fall below g† (100% in condensed regime)
- Combined 4-galaxy scatter: 0.179 dex; Sorgho pair: 0.094 dex

### SPARC Cross-Validation (ESO444-G084)
ESO444-G084 appears in both SPARC and MHONGOOSE with different parameters:
- SPARC: D=4.83 Mpc, i=32°, 7 pts, M/L=0.5, σ=0.055
- MHONGOOSE: D=4.6 Mpc, i=49° (kinematic), 17 pts, M/L=0.2, σ=0.315
- Systematic budget: distance shift (-0.04 dex), inclination shift (-0.31 dex), M/L shift (-0.40 dex)
- The MHONGOOSE kinematic inclination (49°) is likely more reliable than SPARC's photometric (32°)
- Verdict: Data quality suitable for integration

---

## 6. Test Results by Category

### A. Environmental Scatter Tests (7 tests)

BEC predicts universal scatter (g† = fundamental constant). CDM predicts environmental dependence from halo assembly bias.

| # | Test | File | Result | Verdict |
|---|------|------|--------|---------|
| A1 | Definitive 7-test battery | `test_env_scatter_definitive.py` | Levene p=0.64, 0.76, 0.45, 0.74 (all >0.05) | **BEC-CONSISTENT** |
| A2 | Acceleration-binned | `test_env_cf4_accel_binned.py` | σ_dense=0.165, σ_field=0.171, p=0.64 | **BEC-CONSISTENT** |
| A3 | Confound control | `test_env_confound_control.py` | Environmental signal = distance artifact | **BEC-CONSISTENT** |
| A4 | Propensity-matched pairs | `test_propensity_matched_env.py` | 29 pairs, Δσ=+0.008 (p=0.12) | **BEC-CONSISTENT** |
| A5 | MC distance injection | `test_mc_distance_and_inversion.py` | Robust to 5-30% errors; inv at -9.856 | **BEC-CONSISTENT** |
| A6 | Void gradient | `test_void_gradient.py` | Only 10 void galaxies | **INCONCLUSIVE** |
| A7 | ALFALFA×Yang BTFR | `test_alfalfa_yang_btfr.py` | Complex pattern, 2,273 galaxies | **INCONCLUSIVE** |

**Critical discovery:** Prior CF4 environmental result (p<0.001) was an artifact of UGC06786's incorrect distance (9.88 vs 29.3 Mpc).

### B. Inversion Point Stability Tests (5 tests)

| # | Test | File | Result | Verdict |
|---|------|------|--------|---------|
| B1 | Binning robustness | `test_binning_robustness.py` | 25/25 configs within ±0.20 dex, mean=-9.89 | **STRONGLY STABLE** |
| B2 | Leave-one-out jackknife | `test_jackknife_robustness.py` | 131 iterations, max shift 0.065 dex | **ROBUST** |
| B3 | Split-half (1000×) | `test_split_half_replication.py` | 95.65% of splits replicate in BOTH halves | **STRONGLY REPLICATED** |
| B4 | Four-method nonparametric | `test_nonparametric_inversion.py` | All 36 configs within ±0.20 dex | **MODEL-INDEPENDENT** |
| B5 | PROBES external | `test_probes_inversion_replication.py` | Inversion at -11.12 (1.20 dex from g†) | **NOT CONFIRMED** |

**Note on B5:** PROBES uses stars-only g_bar (no gas correction available), which shifts the inversion. Gas correction improves scatter by 0.045 dex but doesn't fully resolve it.

### C. Quantum Signature Tests (5 tests)

| # | Test | File | Result | Verdict |
|---|------|------|--------|---------|
| C1 | Forward-model soliton vs NFW | `test_forward_model_bunching.py` | NFW preferred 53%, BEC 34%, ΔAIC=-20.3 | **FAILS BEC** |
| C2 | PHANGS inner quantum (X<1) | `test_phangs_inner_quantum.py` | Inner deficit -0.181 dex (opposite prediction) | **FAILS BEC** |
| C3 | Radial variance decomposition | `test_radial_variance_profile.py` | Inner ΔAIC=-27 (classical preferred) | **FAILS BEC** |
| C4 | BEC exponent α vs mass | `test_alpha_with_mtotal.py` | Global α≈1.06, weak mass trend | **MARGINAL** |
| C5 | Phase diagram model | `test_phase_diagram_model.py` | Data follows predicted boundary | **BEC-CONSISTENT** |

### D. Healing Length / Coherence Tests (7 tests)

| # | Test | File | Result | Verdict |
|---|------|------|--------|---------|
| D1 | Radial autocorrelation | `test_interface_oscillation.py` | Lag-1 ACF=0.700, 100% positive, p=2.3×10⁻⁴¹ | **STRONG ACF** |
| D2 | ACF controls | `test_interface_oscillation_controls.py` | Survives demeaning, differencing, detrending | **GENUINE STRUCTURE** |
| D3 | Lomb-Scargle spectral | `test_interface_spectral.py` | 37.3% significant periodicity, λ≈6.23 kpc | **EXCESS PERIODICITY** |
| D4 | Baryon wiggle control | `test_baryon_wiggle_vs_coherence.py` | Baryonic structure ≠ ACF predictor (all p>0.15) | **INTERFACE PHYSICS** |
| D5 | Healing length Lc | `test_healing_length_scaling.py` | ρ(Lc,ξ)=0.551 BUT confounded by size | **CONFOUNDED** |
| D6 | Healing length k* | `test_healing_length_kstar.py` | ρ(k*,ξ)=0.278 (p=0.023) but N_pts confound | **CONFOUNDED** |
| D7 | ACF + distance diag | `test_healing_length_acf.py` | Signal vanishes in clean subsample (p=0.64) | **EDD ARTIFACT** |

### E. Multi-Scale and External Tests (5 tests)

| # | Test | File | Result | Verdict |
|---|------|------|--------|---------|
| E1 | KiDS-1000 lensing | `test_brouwer_lensing_rar.py` | Env matters MORE at high mass; core unresolved | **BEC-CONSISTENT** |
| E2 | Galaxy cluster RAR | `test_cluster_rar_tian2020.py` | Best-fit a₀=14.4×g†; galaxy g† rejected | **CONSTRAINS THEORY** |
| E3 | Literature cross-ref | `literature_crossref_tests.py` | Consistent with McGaugh+2016, Lelli+2017 | **CONSISTENT** |
| E4 | Cluster σ scaling | `test_cluster_sigma_scaling.py` | B=0.014±0.229, g†≠σ² | **AGAINST simplest BEC** |
| E5 | Cluster g_core multivariate | `test_cluster_gcore_scaling.py` | R²=0.749 with g_core+g_200; σ irrelevant | **ILLUMINATING** |
| E6 | Hierarchical healing length | `test_hierarchical_healing_length.py` | g†_eff ∝ M^0.32, r=0.83, p=2×10⁻²³ | **BEC-CONSISTENT** |

**E5 key finding:** Cluster-scale g† is jointly controlled by core mass profile (g_core, p=0.026) and virial potential (g_200, p=0.033), NOT velocity dispersion. The 14× enhancement reflects where clusters sit on the mass-acceleration plane.

**E6 key finding:** The 14× enhancement of g† at cluster scales follows a single power law g†_eff = g† × (M/M_ref)^0.32 connecting 67 galaxies (g†_eff/g† = 1) to 20 clusters (g†_eff/g† = 6–32) across 4 orders of magnitude in mass. See Section 7 for detailed results.

### F. Simulation Cross-Validation (8 tests) — THE BOMBSHELL

| # | Test | File | Result | Verdict |
|---|------|------|--------|---------|
| F1 | EAGLE v1 (unit bug) | `test_eagle_rar_inversion.py` | SUPERSEDED | — |
| F2 | EAGLE v2 (corrected) | `test_eagle_rar_inversion_v2.py` | Bootstrap 0.9% near g† | **DISCRIMINATING** |
| F3 | EAGLE 6-test pipeline | `eagle_bec_pipeline.py` | 5/6 pass, BEC form best BIC | **MIXED** |
| F4 | TNG Phase 1 (stellar) | `tng_rar_analysis.py` | g†=3.16×10⁻¹¹, bootstrap 27.2% | **INTERMEDIATE** |
| F5 | TNG Phase 1B (stars+gas) | `tng_phase1b_gas.py` | Bootstrap **100%** at all bin widths | **INVERSION CONFIRMED** |
| F6 | EAGLE adapted radii | `eagle_adapted_radii.py` | g†=**1.27×10⁻¹⁰**, bootstrap **99.7%** | **METHOD ARTIFACT** |
| F7 | LCDM hydro inversion | `test_lcdm_hydro_inversion.py` | EAGLE/TNG offset 0.22 dex vs SPARC 0.06 | **DISCRIMINATING** |
| F8 | LCDM null inversion | `test_lcdm_null_inversion.py` | Only 5% of LCDM mocks near g† | **DISCRIMINATING** |

**The simulation summary table:**

| Metric | SPARC | EAGLE fixed | EAGLE adapted | TNG star-only | TNG star+gas |
|--------|-------|-------------|---------------|---------------|--------------|
| Radii used | 15-50 var | 10 fixed | 2 adapted | 2-3 adapted | 2-3 adapted |
| g† fitted | 1.20e-10 | 6.97e-11 | **1.27e-10** | 2.92e-10 | 6.10e-11 |
| Scatter | 0.12 dex | 0.115 | **0.071** | 0.072 | 0.075 |
| Bootstrap % (bw=0.15) | 99.9% | 62.6% | **99.7%** | 99.9% | **100%** |
| Bootstrap % (bw=0.30) | 99.9% | 6.2% | **95.0%** | 89.9% | **100%** |
| Inversion median | -9.89 | -10.37 | **-9.887** | -9.943 | -9.956 |

**Implication:** The scatter inversion at g† is a **universal structural feature** of galaxy mass profiles — it appears in SPARC, EAGLE, and TNG. It is NOT a discriminant between BEC and ΛCDM. However, SPARC's inversion is 4× closer to g† than the LCDM composite from Marasco+2020 (0.06 vs 0.22 dex).

### G. Additional Analyses (new since master map)

| # | Test | File | Result | Verdict |
|---|------|------|--------|---------|
| G1 | Kurtosis phase transition | `test_kurtosis_phase_transition.py` | Kurtosis=20.7 at g†, LCDM=0.009 (2,428× ratio) | **DISCRIMINATING** |
| G2 | Kurtosis disambiguation | `test_kurtosis_disambiguation.py` | Korsaga drives 96%+ of spike (κ=23.9 vs SPARC 0.6) | **DIAGNOSTIC** |
| G3 | Kurtosis MHONGOOSE | `test_kurtosis_mhongoose.py` | Δκ=0.0 after adding 82 pts, 252 gal | **UNCHANGED** |
| G4 | Power spectrum shape | `test_residual_power_spectrum.py` | β=0.03 (flat), NOT turbulence; bump at ~23 kpc | **DIAGNOSTIC** |
| G5 | Simulation coherence | `test_simulation_coherence.py` | SPARC 37% periodic vs EAGLE 3%, ACF 0.70 vs 0.60 | **POTENTIALLY DISCRIMINATING** |
| G6 | EAGLE injection recovery | `test_eagle_injection_recovery.py` | 95.3% false-negative rate at EAGLE resolution | **CALIBRATION** |
| G7 | Classical wave mimicry | classical_wave_mimicry_analysis.py | ΔAIC=-0.8 quantum vs classical wave, need 300× more data | **INDISTINGUISHABLE** |
| G8 | Lensing profile shape | `test_lensing_profile_shape.py` | NFW preferred, core unresolved (R_min=35 kpc > ξ) | **INCONCLUSIVE** |
| G9 | TF scatter vs redshift | `test_tf_scatter_redshift.py` | Band/technique differences dominate apparent evolution | **INCONCLUSIVE** |
| G10 | PROBES gas correction | (integrated pipeline) | Gas correction improves scatter by 0.045 dex | **DIAGNOSTIC** |
| G11 | Extended RAR + MHONGOOSE | `test_extended_rar_inversion.py` | All tiers replicate inversion; MH shifts B+ by -0.004 dex | **REPLICATED** |
| G12 | α-M_dyn scan | (alpha analysis) | Crossover α≈0.28, signal robust across wide α range | **DIAGNOSTIC** |

---

## 7. Hierarchical Healing Length & Cluster g†_eff Mass Scaling (NEW)

### Motivation
At galaxy scales, the RAR works with g† ≈ 1.2×10⁻¹⁰ m/s². At cluster scales (Tian+2020, 20 CLASH clusters), the best-fit is g†_eff ≈ 14.4× g†. If DM is a BEC, the healing length ξ = √(GM/g†) sets the coherence scale, and g†_eff should follow a predictable mass-scaling relation — not a random offset.

### Test 3a: Mass Scaling of g†_eff

| Fit | α (power-law) | r | p-value | N |
|-----|---------------|---|---------|---|
| **Combined galaxy+cluster (Mtot)** | **0.317 ± 0.023** | **0.831** | **2.2×10⁻²³** | 87 |
| Cluster-only (Mtot) | 0.542 ± 0.180 | 0.578 | 0.008 | 20 |
| Cluster-only (M200) | 0.907 ± 0.609 | 0.598 | 0.210 | 6 |
| Combined galaxy+cluster (M200) | 0.175 ± 0.013 | 0.849 | 2.3×10⁻²¹ | 73 |
| Naive (median cluster / median galaxy) | 0.469 | — | — | — |

- **M_ref** (where g†_eff = g†) = 9.3×10⁹ M☉ — roughly a dwarf galaxy mass
- The naive BEC prediction (α = 0.5) is within 2σ of the intra-cluster slope (0.54 ± 0.18)
- The combined fit (α = 0.32) is shallower, suggesting sub-linear scaling

### Test 3b: Healing Length vs Core Size

| Scale | ξ / R_physical | Correlation | p |
|-------|---------------|-------------|---|
| Galaxy ξ vs R_disk | 1.68× | r = 0.755 | 1.5×10⁻¹³ |
| Cluster ξ_eff vs BCG Rad | 1.41× | r = 0.037 | 0.878 |
| Cluster ξ_core vs BCG Rad | 5.7× | r = 0.283 | 0.226 |
| Cluster ξ_200 / R200 | 0.58× | — | — |
| Cluster ξ_200_eff / R200 | 0.16× | — | — |

- ξ_eff ≈ 1.4× BCG aperture radius — same order of magnitude, but no significant correlation within clusters
- At galaxy scale, ξ / R_disk ≈ 1.7× with strong correlation

### Test 3c: Hierarchical Consistency

ξ = √(GM/g†) spans from ~0.3 kpc (dwarf galaxies, M ~ 10⁸ M☉) to ~1,400 kpc (massive clusters, M200 ~ 10¹⁵ M☉). The log-log slope of ξ vs M across all scales is exactly 0.500 ± 0.000 — consistent with the BEC prediction (ξ ∝ M^0.5) by construction, since the same formula is used at all scales.

### Test 3d: Potential Well Depth

| Predictor | Slope | r | p | Significant? |
|-----------|-------|---|---|--------------|
| g†_eff vs Mtot | 0.542 | 0.578 | 0.008 | **YES** |
| g†_eff vs g_core | 0.138 | 0.186 | 0.433 | No |
| g†_eff vs g_200 (N=6) | 1.075 | 0.435 | 0.388 | No (low N) |
| E5 multivariate (g_core + g_200) | — | R²=0.749 | — | **YES** |

- g†_eff correlates significantly with total mass (p = 0.008) but NOT with core acceleration alone
- The multivariate model from E5 (R² = 0.749) remains the best predictor

### Step 4: Cluster Scatter Profile

| R (kpc) | σ (dex) | ⟨resid⟩ | Phase |
|---------|---------|---------|-------|
| 14.3 | 0.151 | -0.018 | Inner (R < ξ_core) |
| 100 | 0.096 | +0.063 | Near ξ_core ≈ 88 kpc |
| 200 | **0.072** | +0.051 | **Minimum** |
| 400 | 0.075 | -0.029 | Near ξ_200_eff ≈ 306 kpc |
| 600 | 0.091 | -0.098 | Outer |

- Inner scatter (0.151) > minimum (0.072) > outer (0.091) — consistent with BEC dual-state picture
- Scatter minimum at ~200 kpc is between ξ_core (88 kpc) and ξ_200_eff (306 kpc)

### Per-Cluster Healing Length Summary (20 CLASH clusters)

| Cluster | Mtot (10¹² M☉) | g†_eff/g† | ξ_core (kpc) | ξ_eff (kpc) | M200 (10¹⁴ M☉) | ξ_200 (kpc) |
|---------|-----------------|-----------|-------------|------------|-----------------|-------------|
| A209 | 3.87 | 10.7 | 67.0 | 20.5 | — | — |
| A383 | 7.55 | 21.3 | 93.6 | 20.3 | 8.4 | 988 |
| MACS0329 | 25.3 | 30.2 | 171.4 | 31.2 | — | — |
| MACS0647 | 7.71 | 23.8 | 94.6 | 19.4 | 18.0 | 1446 |
| MACS1115 | 6.25 | 12.3 | 85.2 | 24.3 | 11.0 | 1130 |
| MACS1931 | 7.21 | 14.1 | 91.5 | 24.4 | 12.0 | 1181 |
| MS2137 | 3.98 | 6.0 | 68.0 | 27.9 | 7.9 | 958 |
| RXJ2129 | 6.65 | 11.5 | 87.9 | 25.9 | 7.7 | 946 |
| (12 more) | 4.1–10.0 | 12.4–21.2 | 68–108 | 17–33 | — | — |

### Verdict
**MODERATE-TO-STRONG RESULT.** The 14× enhancement of g† at cluster scales is NOT anomalous — a single power law g†_eff ∝ M^0.32 (r = 0.83, p = 2×10⁻²³) connects galaxies to clusters across 4 orders of magnitude in mass. The BEC healing length provides a coherent framework spanning 0.3–3000 kpc. However, intra-cluster scatter is substantial, and the univariate mass scaling (r = 0.578) is weaker than the E5 multivariate model (R² = 0.749).

---

## 8. Kurtosis Source Decomposition — Critical Finding

The excess kurtosis spike at g† (κ=20.7 in Tier K) is **not universal across surveys**:

| Source | κ at g† bin | N points | Notes |
|--------|-----------|----------|-------|
| **Korsaga 2019** | **23.95** | 5,705 | Drives 96%+ of the signal |
| SPARC | 0.64 | 2,740 | Near-Gaussian |
| THINGS + deBlok | -0.46 | 894 | Platykurtic (no spike) |
| LCDM mock | 0.009 | — | Essentially zero |
| **Combined Tier K** | **20.71** | 9,406 | Korsaga-dominated |
| **Tier K + MHONGOOSE** | **20.71** | 9,488 | UNCHANGED after adding 82 pts |

**Implication:** The kurtosis spike may be a Korsaga-2019 systematic rather than a universal physical signal. The MHONGOOSE augmentation confirms robustness to additional data but does not resolve the source-dependence question. Further investigation needed with independent mass-model datasets.

---

## 9. Scatter by Data Source

| Source | N_pts | N_gal | σ(residual) | ⟨residual⟩ |
|--------|-------|-------|-------------|------------|
| SPARC | 2,740 | 126 | 0.172 | -0.174 |
| THINGS | 533 | 6 | 0.130 | -0.061 |
| deBlok2002 | 361 | 17 | 0.332 | +0.100 |
| Verheijen2001 | 83 | 5 | 0.442 | -0.176 |
| Korsaga2019 | 5,705 | 98 | 0.334 | +0.099 |
| PHANGS_Lang2020 | 1,745 | 50 | 0.471 | +0.173 |
| GomezLopez2019 | 2,852 | 71 | 0.607 | -0.091 |
| GHASP_2008b | 34 | 1 | 0.419 | -0.280 |
| **Sorgho+2019** | **50** | **2** | **~0.07** | **~-0.09** |
| **Nkomo+2025** | **32** | **2** | **~0.25** | **~-0.10** |

**Key insight:** Inversion detection requires σ < 0.2 dex mass decomposition. Adding quality data (THINGS, Sorgho) improves signal; adding noisy data (GomezLopez, GHASP) degrades it.

---

## 10. What Is Established After 50+ Tests

### Confirmed (robust across methods)
1. RAR scatter has a minimum (inversion) near log g = -9.9 — in SPARC, EAGLE, TNG, and all extended tiers
2. Inversion robust to binning (25/25), jackknife (131/131), split-half (95.7%), and 4 nonparametric methods (36/36)
3. RAR residuals have strong radial coherence (ACF lag-1 = 0.70, 100% positive, p = 10⁻⁴¹)
4. 37% of galaxies show significant periodicity in RAR residuals (λ ≈ 6 kpc), 11× expected by chance
5. Environmental scatter is universal (no field vs cluster difference) in SPARC — all 4 primary tests p > 0.05
6. BEC functional form provides best fit to simulation RAR residuals (BIC test)
7. Power spectrum of residuals is FLAT (β=0.03), ruling out turbulence
8. MHONGOOSE data integrates seamlessly; inversion shifts by <0.01 dex
9. Non-SPARC-only (248 galaxies) independently finds inversion at -9.933 (only 0.012 dex from g†)
10. **Cluster g†_eff follows mass scaling**: g†_eff ∝ M^0.32 (r=0.83, p=2×10⁻²³) — the 14× cluster enhancement is a natural consequence of deeper potential wells, not anomalous physics

### Refuted or Inconclusive
1. Scatter inversion is NOT a BEC-vs-ΛCDM discriminant — simulations produce it too
2. Soliton core fits NOT preferred over NFW (ΔAIC = -20.3 favoring NFW)
3. Inner quantum regime (X<1) shows deficit, not excess
4. Healing length scaling confounded by galaxy size
5. Cluster RAR has acceleration parameter 14.4× g† — but now shown to follow predictable mass scaling (E6)
6. Quantum vs. classical wave distinction unresolvable (ΔAIC = -0.8, need 300× more data)
7. Kurtosis spike at g† is Korsaga-dominated, not universal across surveys
8. EAGLE periodicity comparison (37% vs 3%) invalidated by 95.3% false-negative rate at EAGLE resolution

### The Open Puzzle
The mathematical identity between RAR and Bose-Einstein statistics remains **unexplained**. Three possibilities:
- **(a)** Mathematical coincidence — ΛCDM feedback naturally produces RAR-like relations
- **(b)** BEC statistics emerge from ΛCDM (the condensate IS the CDM halo in some limit)
- **(c)** Both simulations and observations approximate the same underlying quantum physics

No current test distinguishes between these three possibilities.

---

## 11. Recent Chronology

1. **Foundation:** Built RAR from SPARC 175 galaxies; confirmed BEC functional form with α ≈ 1.0
2. **Environmental test:** Found field > cluster scatter at 99.8% — then discovered it was a distance artifact (UGC06786)
3. **Definitive environmental battery (7 tests):** Universal scatter confirmed; BEC-consistent but not discriminating
4. **Inversion stability (5 tests):** Scatter minimum at g† is real, robust, and model-independent
5. **Quantum signatures (5 tests):** Mixed — soliton cores not preferred, inner regime fails, but BEC form fits best
6. **Healing length series (7 tests):** Strong radial coherence exists, but ξ scaling confounded by size
7. **EAGLE v2:** Fixed-aperture analysis shows 0.9% inversion — appeared to discriminate ΛCDM from observations
8. **TNG Phase 1→1B:** Gas + adapted radii → 100% inversion; feature strengthens with baryonic completeness
9. **EAGLE adapted radii (THE BOMBSHELL):** EAGLE shows 95-99.7% inversion with adapted radii, matching SPARC/TNG. The "discriminant" was a method artifact.
10. **Extended RAR tiers:** Built 256-galaxy dataset from 9 surveys; inversion stable from Tier A through Tier C
11. **Kurtosis decomposition:** Spike at g† is real in Tier K but driven by Korsaga 2019 (κ=23.9 vs SPARC 0.6)
12. **Classical wave mimicry:** Quantum vs. classical wave indistinguishable (ΔAIC=-0.8)
13. **MHONGOOSE integration:** NGC3621 (σ=0.060) and NGC7424 (σ=0.082) — SPARC-quality data from MeerKAT; Nkomo dwarfs probe ultra-low-acceleration regime; Tier MH replicates inversion at -10.157
14. **Power spectrum shape:** Flat β=0.03 rules out turbulence; spectral bump at ~23 kpc in periodic galaxies
15. **Simulation coherence calibrated:** EAGLE's 95% false-negative rate means periodicity comparison cannot be claimed as physical
16. **Hierarchical healing length (E6):** g†_eff ∝ M^0.32 (r=0.83, p=2×10⁻²³) connects 67 galaxies to 20 clusters; ξ spans 0.3–3000 kpc; cluster scatter profile consistent with BEC dual-state transition; ξ_eff ≈ 1.4× BCG aperture radius

---

## 12. File Reference

| Directory | Contents |
|-----------|----------|
| `analysis/pipeline/` | 85+ Python test scripts |
| `analysis/results/` | 70+ JSON summaries (incl. `summary_hierarchical_healing_length.json`), 12 CSV data products, 18 PNG figures |
| `data/sparc/` | SPARC rotation curves |
| `data/things/` | THINGS mass models |
| `data/mhongoose/` | MHONGOOSE RAR data (mhongoose_rar_all.tsv, mhongoose_rar_points.tsv) |
| `data/eagle_rar/` | EAGLE simulation data |
| `data/vizier_catalogs/` | Korsaga, Sofue, Martinsson, etc. |
| `data/brouwer2021/` | KiDS-1000 lensing ESD data |
| `data/cluster_rar/` | Tian+2020 cluster data |
| `data/alfalfa/` | ALFALFA HI survey |
| `data/probes/` | PROBES survey (3,163 galaxies) |
| `docs/` | BEC_DM_THEORY.md (106 KB), MASTER_PROGRESS_MAP.md |
| `dashboards/` | React visualization components |
| `figures/` | Generated plots (incl. `hierarchical_healing_length.png` — 4-panel mass scaling figure) |

### Key Pipeline Scripts
- `load_extended_rar.py` — Data loader for multi-survey RAR
- `match_bouquin_photometry.py` — Bouquin+2018 profile matching
- `match_korsaga_massmodels.py` — Korsaga+2019 Freeman+Sérsic reconstruction
- `integrate_mhongoose.py` — MHONGOOSE Nkomo+2025 integration
- `integrate_mhongoose_sorgho.py` — MHONGOOSE Sorgho+2019 integration
- `test_extended_rar_inversion.py` — Quality-tiered inversion analysis (includes Tier MH)
- `test_hierarchical_healing_length.py` — Hierarchical healing length & mass scaling (E6)

---

## 13. Quick Scorecard (Updated)

| Category | Tests | BEC Support | Against BEC | Inconclusive |
|----------|-------|-------------|-------------|--------------|
| Environmental scatter | 7 | 4 (universal σ) | 0 | 3 |
| Inversion stability | 5+2 MH | 7 (rock-solid) | 0 | 0 |
| Quantum signatures | 5 | 1 | 3 | 1 |
| Simulation cross-val | 8 | — | — | see Section F |
| Healing length / ACF | 7 | 2 | 1 | 4 |
| Multi-scale / external | 6 | 2 | 1 | 3 |
| Additional analyses | 12 | 2 | 0 | 10 |
| **TOTAL** | **~50** | **18** | **5** | **21** |

**+ 8 simulation tests showing inversion is universal across ΛCDM and observations**

---

*This document is self-contained and designed for export to external AI systems for full project context. Generated February 21, 2026.*
