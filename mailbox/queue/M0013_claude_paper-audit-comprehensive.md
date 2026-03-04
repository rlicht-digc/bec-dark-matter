---
message_id: M0013
task_id: null
from: claude
to: codex
type: handoff
timestamp_utc: "2026-03-04T20:00:00Z"
repo_state:
  git_sha: "bc7fb1635c8560acb1eb1aed60c4451ff9096cbd"
  dirty: true
references: []
---

COMPREHENSIVE AUDIT: Paper 1 vs Paper 2, Test Infrastructure, and Consolidation Needs

CONTEXT
Russell flagged that `paper1.pdf` (OSF bundle, Mar 1) doesn't look like the same paper he was working on and may be incomplete. This audit inventories ALL paper artifacts in the repo, compares them, and maps every test to the claims they support. The goal: nail down what's what, what's missing, and what needs to happen today.

---

## 1. PAPER INVENTORY — TWO DISTINCT PAPERS EXIST

### Paper 1: "A Bose-Einstein-Form Mapping of the Radial Acceleration Relation and Its Residual Statistics"

| Attribute | Value |
|-----------|-------|
| **LaTeX source** | `paper/main.tex` (917 lines) |
| **Compiled PDF (paper dir)** | `paper/main.pdf` (560 KB) |
| **OSF bundle PDF** | `outputs/paper1_osf_bundle/20260301_131035/paper1.pdf` (558 KB) |
| **Date** | March 1, 2026 |
| **Pages** | 10 |
| **Figures** | 4 (fig1_identity, fig2_variance, fig3_kurtosis, fig4_inversion) |
| **Scope** | Comprehensive 23-test statistical battery across 6 datasets |

**Sections**: Introduction, Data (sample tiers, env classification), The Algebraic Identity (derivation, variable mapping, cosmological connection), Three Statistical Fingerprints (variance scaling, scatter inversion, kurtosis spike), Internal Robustness (8 tests), External Validation (PROBES failure, lensing, ALFALFA, clusters), Scorecard, Predictions, Summary.

**Key results**:
- Variance scaling: wave (n-bar^2) preferred over Poisson, Delta-AIC_eta = +10.5
- Scatter inversion at g-dagger within 0.05 dex (4 nonparametric methods agree)
- Kurtosis spike kappa_4 = 20.7 at g-dagger (BUT B-band Korsaga dependent; SPARC-only kappa_4 = 0.64)
- PROBES replication FAILURE: 1.20 dex offset (strongest negative result)
- Scorecard: 10 supporting, 5 inconclusive, 1 failed, 2 diagnostic, 3 different regime

### Paper 2: "A Kernel+Scale Identifiability Framework with Anti-Numerology Controls: Localizing g-dagger in the Radial Acceleration Relation"

| Attribute | Value |
|-----------|-------|
| **LaTeX source** | `paper/gdagger_hunt_paper.tex` (592 lines) |
| **Compiled PDF** | `paper/gdagger_hunt_paper.pdf` (643 KB) |
| **arXiv submission** | `paper/arxiv_submission_20260224/main.pdf` (655 KB) |
| **arXiv bundle dir** | `paper/arxiv_submission_20260224/` (full submission package) |
| **Date** | February 24, 2026 |
| **Pages** | 7 |
| **Figures** | 3 (three_panel AIC/nulls, validation_stability, negative_controls) |
| **Scope** | Methodological framework — 7 kernels, anti-numerology controls |

**Sections**: Introduction, Methods (kernel library, scale optimization, scale injection scan, null tests, cross-validation, nearby-scale comparison), Data (SPARC + cluster pilot), Results (baseline, nulls, CV, sensitivity, negative controls, nearby scales, cluster null), Discussion, Summary.

**Key results**:
- BE RAR kernel best, optimal scale 0.0139 dex from g-dagger
- Global shuffle: 0/500 hits at +/-0.10 dex (CP bound 0.006)
- Galaxy-grouped 5-fold CV: log10 s = -9.9195 +/- 0.0437
- Variable warping completely destroys optimum (confirms genuine feature)
- Cluster pilot correctly returns null (framework validation)

### Relationship Between Papers
Paper 2 is the **methodology paper** that develops the kernel+scale identifiability framework. Paper 1 is the **comprehensive analysis paper** that applies the framework and explores physical interpretations (variance scaling, scatter inversion, kurtosis). They are complementary — Paper 2 establishes the tooling, Paper 1 uses it alongside 20+ additional tests.

### POTENTIAL ISSUE
The file `paper/main.tex` (Paper 1 source) was recently modified (dirty in working tree). The OSF bundle `paper1.pdf` may be stale relative to the current `.tex` source. The `paper/main.pdf` was also modified. Need to verify whether the OSF bundle was rebuilt after the latest `.tex` edits.

---

## 2. COMPLETE TEST-TO-PAPER MAPPING

### Tests Supporting Paper 1 Claims

**Claim: Algebraic Identity (Section 3)**
- No dedicated test — this is a mathematical derivation, not empirical

**Claim: Variance Scaling / Fingerprint 1 (Section 4.1)**
- `analysis/pipeline/test_rar_tightness.py` (529 lines) — SPARC core variance model comparison
- `analysis/pipeline/test_rar_tightness_probes.py` (779 lines) — PROBES variant
- Results: `summary_rar_tightness.json`

**Claim: Scatter Inversion / Fingerprint 2 (Section 4.2)**
- `analysis/pipeline/test_nonparametric_inversion.py` — 4 nonparametric methods
- `analysis/pipeline/test_mc_distance_and_inversion.py` — Monte Carlo distance injection
- `analysis/pipeline/test_binning_robustness.py` — 25 binning configurations
- `analysis/pipeline/test_jackknife_robustness.py` — leave-one-out stability
- `analysis/pipeline/test_split_half_replication.py` (700 lines) — 1000 random half-splits
- Results: `summary_nonparametric_inversion.json`, `summary_mc_distance_and_inversion.json`, `summary_binning_robustness.json`, `summary_jackknife_robustness.json`, `summary_split_half_replication.json`

**Claim: Kurtosis Spike / Fingerprint 3 (Section 4.3)**
- `analysis/pipeline/test_kurtosis_phase_transition.py` — kurtosis vs log gbar
- `analysis/pipeline/test_kurtosis_disambiguation.py` — geometric artifact rule-out
- `analysis/pipeline/test_kurtosis_mhongoose.py` — MHONGOOSE robustness check
- Results: `summary_kurtosis_phase_transition.json`, `summary_kurtosis_disambiguation.json`

**Claim: Environmental Scatter (Section 5.6)**
- `analysis/pipeline/test_env_scatter_definitive.py` — field vs dense Levene tests
- `analysis/pipeline/test_env_confound_control.py` — covariate/propensity matching
- `analysis/pipeline/test_propensity_matched_env.py` — propensity-score matched pairs
- `analysis/pipeline/test_env_cf4_accel_binned.py` — CF4 acceleration-binned
- `analysis/pipeline/test_env_triple_distance.py` — triple distance metric
- Results: `summary_env_definitive.json`, `summary_env_confound_control.json`, `summary_propensity_matched_env.json`

**Claim: Structure-Only Null (Section 5.7)**
- `analysis/pipeline/test_lcdm_null_inversion.py` — NFW+disk semi-analytic null
- `analysis/pipeline/test_lcdm_phase_diagram.py` — LCDM phase diagram
- Results: `summary_lcdm_null_inversion.json`

**Claim: Extended RAR Tiers (Section 6.1)**
- `analysis/pipeline/test_extended_rar_inversion.py` — multi-tier inversion
- Results: `summary_extended_rar_inversion.json`

**Claim: PROBES Replication (Section 6.2)**
- `analysis/pipeline/test_probes_inversion_replication.py` — independent replication attempt
- Results: `summary_probes_inversion_replication.json`

**Claim: Weak Lensing (Section 6.3)**
- `analysis/pipeline/test_brouwer_lensing_rar.py` — KiDS-1000 environment effect
- `analysis/pipeline/test_lensing_profile_shape.py` — NFW vs cored profile comparison
- Results: `summary_brouwer_lensing_rar.json`, `summary_lensing_profile_shape.json`

**Claim: ALFALFA BTFR (Section 6.4)**
- `analysis/pipeline/test_alfalfa_yang_btfr.py` — Baryonic TF scatter by environment
- Results: `summary_alfalfa_yang_btfr.json`

**Claim: Cluster Scale (Section 6.5)**
- `analysis/pipeline/test_cluster_rar_tian2020.py` — cluster RAR pilot
- `analysis/pipeline/test_cluster_gcore_scaling.py` — cluster core scaling
- `analysis/pipeline/test_cluster_sigma_scaling.py` — cluster sigma vs g-dagger
- Results: `summary_cluster_rar_tian2020.json`

**Additional Pipeline Tests (supporting but not directly in paper sections)**:
- `test_scale_hierarchy_map.py` (1,153 lines) — multi-scale hierarchy
- `test_scale_hierarchy_rs_discriminator.py` (775 lines) — discriminant analysis
- `test_scale_parameter_analysis.py` (552 lines) — parameter sensitivity
- `test_spectral_slope_alpha.py` (560 lines) — spectral analysis
- `test_simulation_coherence.py` (1,115 lines) — coherence validation
- `test_soliton_nfw_composite.py` (1,393 lines) — soliton+NFW composite
- `test_residual_power_spectrum.py` (882 lines) — power spectrum
- `test_residualized_bec_signal_analysis.py` (890 lines) — residualized signal
- `test_unified_model_adjudication.py` (508 lines) — model comparison
- `test_void_gradient.py` (1,159 lines) — void gradient analysis
- `test_healing_length_scaling.py` — healing length
- `test_injection_recovery.py` — injection recovery
- `test_tf_scatter_redshift.py` (514 lines) — TF redshift evolution

### Tests Supporting Paper 2 Claims

**Core Framework Validation**:
- `analysis/tests/test_gdagger_hunt_pilot.py` (484 lines) — 3 end-to-end experiments (synthetic, SPARC, cluster)
- `analysis/tests/test_gdagger_hunt_refereeproof.py` (1,450+ lines) — 7-suite adversarial battery (A-G)

**Regression Guards**:
- `analysis/tests/test_unified_pipeline_regression.py` (77 lines) — SPARC loader + unified CSV
- `analysis/tests/test_tng_sparc_feature_repro_smoke.py` (117 lines) — TNG-SPARC reproduction

### TNG/Simulation Validation Tests
- `test_tng_coherence.py` (428 lines) — IllustrisTNG coherence
- `test_tng_sparc_composition_sweep.py` (522 lines) — TNG-SPARC composition
- `test_tng_sparc_fairness_gap.py` (577 lines) — fairness gap analysis
- `test_eagle_rar_inversion.py` — EAGLE simulation
- `test_eagle_rar_inversion_v2.py` — EAGLE v2
- `test_xi_massmatched_pairs.py` (983 lines) — mass-matched pair analysis
- `test_matched_analog_comparison.py` — matched analog
- `test_window_matching.py` (647 lines) — window matching

---

## 3. INFRASTRUCTURE SUMMARY

| Category | Count | Total Lines |
|----------|-------|-------------|
| Pipeline test files | 82 | ~62,317 |
| Core test files | 4 | ~2,128 |
| Result summary JSONs | 50+ | — |
| Result CSVs | 20+ | — |
| Tool/utility scripts | 6 | — |
| Runner scripts | 3 | — |
| Paper 3 analysis files | 13 | — |

**Build chain**: `scripts/build_paper1.sh` -> `paper/make_figures.py` (20 KB) -> LaTeX compilation -> `paper/main.pdf`

**OSF packaging**: `tools/osf_packaging/stage_for_osf.sh` + `upload_to_osf.sh` -> `outputs/paper1_osf_bundle/`

**External audit**: `analysis/tools/external_audit_generator.py` (1,800+ lines) generates reproducibility reports with SHA256 hashing

---

## 4. OPEN QUESTIONS AND ACTION ITEMS

### Critical Questions
1. **Which paper is Russell's primary focus today?** Paper 1 (comprehensive, 10 pages) or Paper 2 (methodology, 7 pages)?
2. **Is `paper/main.tex` ahead of the OSF bundle?** The working tree shows `paper/main.tex` and `paper/main.pdf` modified but uncommitted. The OSF bundle may be stale.
3. **What's missing from paper1.pdf?** Russell said "it doesn't look like it has everything in it." Need Russell to specify what sections/results/figures he expected but didn't see.

### Possible Issues
- **Paper 1 vs Paper 2 confusion**: `paper/main.pdf` is Paper 1 but the `arxiv_submission_20260224/main.pdf` is Paper 2 — both named `main.pdf` in different directories. Easy to open the wrong one.
- **OSF bundle staleness**: Built Mar 1, but `.tex` modified since. Rebuild needed?
- **82 pipeline tests, no CI**: Tests are run manually. No automated regression gate to catch breakage.
- **Kurtosis spike fragility**: kappa_4 = 20.7 in Tier K but 0.64 in SPARC-only. This is acknowledged in the paper but remains the weakest fingerprint claim.
- **PROBES failure**: The strongest negative result. Paper 1 handles it honestly but it's the main vulnerability for reviewers.

### Suggested Actions
1. Russell clarifies which paper and what's missing
2. Rebuild OSF bundle from current `main.tex` if source has changed
3. Diff `paper/main.tex` against `outputs/paper1_osf_bundle/20260301_131035/source/main.tex` to see what diverged
4. Commit dirty working tree changes
5. Run `test_unified_pipeline_regression.py` as smoke test to confirm nothing broke

VERIFICATION
Run: `git rev-parse HEAD` — must match bc7fb1635c8560acb1eb1aed60c4451ff9096cbd

DO NOTs
- Do not modify any test files or analysis scripts without explicit instruction
- Do not rebuild the OSF bundle until Russell confirms what's missing
- Do not commit the dirty working tree changes — that's Russell's call
