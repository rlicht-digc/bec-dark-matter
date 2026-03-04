---
message_id: M0015
task_id: null
from: claude
to: codex
type: handoff
timestamp_utc: "2026-03-04T22:00:00Z"
repo_state:
  git_sha: "bc7fb1635c8560acb1eb1aed60c4451ff9096cbd"
  dirty: true
references:
  - M0013
  - M0014
---

PAPER MERGER: Paper 1 + Paper 2 → Unified Publication + Gap Analysis of 61 Uncovered Tests

CONTEXT
Russell reviewed Paper 2 (gdagger_hunt, 7pp) and concluded it has no reason to exist
separately — its kernel+scale framework is the methodological backbone Paper 1 is missing.
Paper 1 claims "this is BE" but never shows BE_RAR beat 6 other functional forms in a
head-to-head with anti-numerology controls. That's a hole reviewers will find. Decision:
merge Paper 2 into Paper 1. Claude has begun drafting the merged LaTeX at
`paper/main_merged.tex`.

Additionally, a full audit found 86 total tests in the codebase but only 25 are referenced
in either paper. 61 tests are uncovered. 12 of those directly support Paper 1 claims and
should be included. 39 more represent candidate Paper 3 material.

---

## 1. MERGER ARCHITECTURE

### New Unified Structure

```
1.  Introduction (P1 intro + P2 framework motivation)
2.  The Algebraic Identity (P1 §3, unchanged)
3.  Methods: Kernel+Scale Framework (ENTIRE P2 methods)
    3.1  Kernel Library (7 kernels)
    3.2  Scale Optimization (grid + Brent)
    3.3  Scale Injection Scan (DAIC/N metric)
    3.4  Null Tests & Structure-Preserving Control (A1/A2b/A3/A2)
    3.5  Galaxy-Grouped Cross-Validation (GroupKFold k=5)
    3.6  Nearby-Scale Comparison (cosmological scales)
4.  Data (P1 tiers merged with P2 SPARC/cluster descriptions)
5.  Framework Validation on SPARC (P2 results: baseline, nulls, CV, sensitivity, negatives, nearby, cluster null)
6.  Three Statistical Fingerprints (P1: variance, inversion, kurtosis)
7.  Internal Robustness (P1: nonparam, binning, jackknife, split-half, MC distance, env, LCDM null)
8.  External Validation & Replication (P1: extended tiers, PROBES failure, lensing, BTFR, clusters, kurtosis disambig, TNG)
9.  Scorecard & Interpretation (P1: 3 possibilities, what established/not)
10. Predictions & Future Tests (P1, unchanged)
11. Summary (merged)
A.  Appendix: AIC Methodology (P1)
B.  Appendix: Reproducibility (P2)
```

### Key Design Decisions
- Methods BEFORE results (readers understand test design before seeing claims)
- Framework validation (P2 results) BEFORE fingerprints (P1 results) — establishes
  that BE_RAR kernel is preferred before showing its physical consequences
- Paper 2's "How to Extend" (§5.4) becomes a subsection of Discussion
- Paper 2's negative controls stay in §5, NOT moved to robustness (they test the
  framework, not the fingerprints)

### Figure Plan (7 figures in merged paper)
1. Fig 1: RAR as BE distribution (P1 fig1_identity) — unchanged
2. Fig 2: AIC scan + null distributions + cluster null (P2 fig1_three_panel) — NEW to merged
3. Fig 3: CV stability + cut sensitivity + grid invariance (P2 fig2_validation_stability) — NEW
4. Fig 4: Negative controls / break tests (P2 fig3_negative_controls) — NEW
5. Fig 5: Residual variance vs log gbar (P1 fig2_variance) — renumbered
6. Fig 6: Scatter inversion at g† (P1 fig4_inversion) — renumbered
7. Fig 7: Excess kurtosis (P1 fig3_kurtosis) — renumbered

### Estimated Size
- Current P1: ~8,500 words / 11 pages
- P2 methods+results to add: ~2,500 words
- Consolidation savings (overlapping intros, data sections): ~800 words
- Net merged: ~10,200 words / ~13-14 pages
- Within ApJ norms (20pp with appendix acceptable)

---

## 2. GAP ANALYSIS: 61 UNCOVERED TESTS

### Category A — Should Be In The Merged Paper (12 tests)

These directly support existing claims but are not referenced:

| # | Test File | Claim Supported | Key Result |
|---|-----------|----------------|------------|
| 1 | test_13b_composite | Soliton+NFW composite lensing | SPARC inner + Brouwer outer stitch |
| 2 | test_13b_diagnostics | Composite validation | Stitch boundary, bias, split-sample DAIC |
| 3 | test_13b_standalone | Independent lensing soliton test | Brouwer ESD profiles alone |
| 4 | test_healing_length_scaling | xi measurement from ACF | Validates X=R/xi framework |
| 5 | test_radial_variance_profile | Variance peaks at X~1 | n-bar^2 in dimensionless coords |
| 6 | test_healing_length_kstar | k-space xi robustness | AR1 grid, inclination controls |
| 7 | test_healing_length_distance_diag | xi survives distance errors | MC distance perturbation |
| 8 | test_hierarchical_healing_length | xi galaxy→cluster scaling | Universal mass scaling |
| 9 | test_scale_parameter_analysis | X predicts DAIC | Meta-validates framework |
| 10 | test_interface_oscillation | Coherent RAR residual oscillations | Wave physics signature |
| 11 | test_interface_oscillation_controls | Oscillation null tests | Permutation shuffle |
| 12 | test_interface_spectral | Spectral analysis of periodicity | Power spectrum + phase coherence |

ACTION: Decide which of these 12 to fold into the merged paper vs defer to Paper 3.
The healing length tests (#4-8) and interface tests (#10-12) may be better as a
separate "Wave Signatures" paper. The composite lensing tests (#1-3) directly
strengthen §8 External Validation.

### Category B — Candidate Paper 3 Material (39 tests)

**Simulation Validation (5 tests):**
- test_eagle_rar_inversion_v2: EAGLE shows NO inversion at g†
- test_lcdm_hydro_inversion: EAGLE + TNG scatter derivative comparison
- test_eagle_injection_recovery: Periodicity detection under sim sampling
- test_simulation_coherence: EAGLE vs SPARC residual coherence
- test_tng_coherence: IllustrisTNG coherence (4-aperture limitation)

**Wave/Periodicity (6 tests):**
- test_residual_power_spectrum: Lomb-Scargle PSD, k·xi collapse
- test_periodic_properties_and_scaling: Lambda_peak distribution
- test_window_matching: SPARC resampled to sim resolution
- test_injection_recovery: Detection power curve
- test_spectral_slope_alpha: Turbulence signature
- test_eagle_injection_recovery: Cross-platform periodicity

**BEC Fluid Mechanics (6 tests):**
- test_gpp_inversion_cs2: Sound speed + boson mass recovery
- test_gpp_universality: d ln gobs / d ln gbar universality
- test_forward_model_bunching: BEC vs NFW per-galaxy DAIC
- test_mass_split_bunching: DAIC vs stellar mass
- test_phangs_inner_quantum: PHANGS X<1 regime
- test_alpha_with_mtotal: xi set by M_total not M*

**Scale Hierarchy (3 tests):**
- test_scale_hierarchy_map: X=R/xi meta-analysis
- test_scale_hierarchy_rs_discriminator: BEC vs LCDM vs CDW
- test_scale_parameter_analysis: X-dependent DAIC support

**Environmental Deep-Dive (4 tests):**
- test_env_cf4_accel_binned: CF4 acceleration-dependent binning
- test_env_triple_distance: Triple distance catalog comparison
- test_cf4_distance_grading: CF4 reliability assessment
- test_inversion_distance_sensitivity: 12 systematic distance shifts

**PROBES Investigation (4 tests):**
- test_probes_control_suite: 4-resolution control battery
- test_probes_massmatched_controls: Mass-matched within PROBES
- test_probes_gas_corrected: ALFALFA HI gas correction
- test_rar_tightness_probes: Formation-history correlations

**Other (11 tests):**
- test_phase_diagram_model, test_lcdm_phase_diagram, test_void_gradient,
  test_baryon_wiggle_vs_coherence, test_korsaga_ml_sensitivity, test_rar_tightness,
  test_matched_analog_comparison, test_tng_sparc_composition_sweep,
  test_tng_sparc_fairness_gap, test_mbh_xi_bridge*, test_residualized_bec_signal_analysis

### Category C — Diagnostic Only (10 tests)
Infrastructure and intermediate validation, not paper-worthy.

### Category D — Superseded/Duplicate (2 pairs)
- test_eagle_rar_inversion (superseded by v2)
- test_gpp_universality_codex (duplicate of test_gpp_universality)

---

## 3. RECOMMENDED PAPER PORTFOLIO

| Paper | Content | Tests | Status |
|-------|---------|-------|--------|
| **Paper 1 (merged)** | BE identity + framework + fingerprints + validation | 25 (current) + 3 lensing composite = 28 | Drafting now |
| **Paper 3** | Wave signatures: healing length, periodicity, spectral analysis, sim comparison | ~20 tests | Tests exist, needs write-up |
| **Paper 4** | BEC fluid mechanics: GPP universality, sound speed, forward model, mass scaling | ~10 tests | Tests exist, needs write-up |

---

## 4. DELIVERABLES FOR CODEX

### Immediate
1. Review merged LaTeX draft at `paper/main_merged.tex` when complete
2. Verify all \Val* macros from `tables/values_from_summary.tex` are preserved
3. Confirm figure file references match existing assets in `paper/figures/`

### Before OSF Upload
4. Rebuild OSF bundle from merged source
5. Update reproducibility.md to reflect merged test count
6. Regenerate MANIFEST.csv and CHECKSUMS.csv

### Paper 3 Prep
7. Inventory which Category B tests have complete summary JSONs
8. Flag any tests that need re-running with current data

ACCEPTANCE CRITERIA
- [ ] Merged LaTeX compiles without errors
- [ ] All 7 figures reference existing files
- [ ] All \Val* macros resolve
- [ ] No content from Paper 2 is lost
- [ ] Paper 1's 23-test scorecard preserved and expanded

VERIFICATION
Run: `git rev-parse HEAD` — must match bc7fb1635c8560acb1eb1aed60c4451ff9096cbd

DO NOTs
- Do not modify any test scripts
- Do not delete gdagger_hunt_paper.tex (keep as reference)
- Do not rebuild OSF bundle until merged paper is reviewed by Russell
