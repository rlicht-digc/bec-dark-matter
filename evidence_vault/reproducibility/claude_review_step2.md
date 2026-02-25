# Step 2 Review Memo — Claude (Reviewer)

**Reviewing**: `evidence_vault/reproducibility/alignment_inventory.csv` from commit 81c016f
**Date**: 2026-02-25
**Scope**: State classification accuracy, verdict quality, misclassifications, priorities

---

## 1. Executive Summary

The inventory correctly classifies **791 artifacts** across 4 roots. The deterministic classifier is sound in design but has **3 structural issues** and **several specific misclassifications** that need correction before the inventory drives the move plan.

**Key numbers**:
- 8 of 16 core scripts are stuck at State 3/4 due to a single fixable schema gap (missing `description` field)
- 47 State 4 summaries are referenced by live OSF HTML pages — these are either broken evidence being cited, or valid evidence with a trivially fixable schema gap
- 75 `.pyc` bytecache files are inventoried as "table" artifacts — these are build junk

---

## 2. State Misclassifications

### 2a. CRITICAL — 8 Core Scripts Wrongly Demoted to State 3

These scripts are in `CORE_SCRIPTS` but their summaries lack the `description` field (or in one case, use `test` instead of `test_name`), so they fall to State 3 (script) / State 4 (summary):

| Script | Summary | Actual Verdict | Issue |
|--------|---------|---------------|-------|
| `test_env_scatter_definitive.py` | `summary_env_definitive.json` | BEC-CONSISTENT | Missing `description` |
| `test_nonparametric_inversion.py` | `summary_nonparametric_inversion.json` | ALL_MATCH | Missing `description` |
| `test_env_confound_control.py` | `summary_env_confound_control.json` | INVERSION_FRAGILE | Missing `description` |
| `test_split_half_replication.py` | `summary_split_half_replication.json` | STRONGLY_REPLICATED | Missing `description` |
| `test_propensity_matched_env.py` | `summary_propensity_matched_env.json` | INCONCLUSIVE | Missing `description` |
| `test_lcdm_null_inversion.py` | `summary_lcdm_null_inversion.json` | DISCRIMINATING | Uses `test` instead of `test_name` |
| `test_kurtosis_disambiguation.py` | `summary_kurtosis_disambiguation.json` | DIAGNOSTIC | Missing `description` |
| `test_probes_inversion_replication.py` | `summary_probes_inversion_replication.json` | NOT_REPLICATED | Missing `description` |

**Fix**: Add `description` field to 7 summaries; rename `test` to `test_name` in 1 summary. After fix, all 8 become State 1. This is the single highest-value fix in the entire repo.

### 2b. 75 `.pyc` Files Classified as "table" (Should Be Excluded)

All files under `analysis/pipeline/__pycache__/` are Python bytecache. They are not artifacts. They should be:
- Added to `.gitignore`
- Excluded from the inventory entirely (or given a new type `junk` at State 4)
- Deleted from tracking if committed

### 2c. `detect_type` Catch-All Misclassifies Non-Table Files

The inventory's type detection falls through to "table" for anything that isn't `.py`, `summary_*.json`, or a figure extension. This incorrectly labels:
- 86 non-summary `.json` files as "table" (includes config files, data files, nested summaries)
- 18 `.md` files as "table" (these are documentation/reports)
- 7 `.log` files as "table"
- 2 `.sh` scripts as "table"
- 2 `.html` files as "table" — including the **OSF HTML pages themselves** (`tests_results_osf.html`, `references_osf.html`), which are map files per the schema contract

**Recommendation**: Add types `documentation`, `config`, `junk` to the classifier. Low priority — doesn't affect state classification, only reporting clarity.

### 2d. State 3 Summaries That Should Be State 2

9 summaries are State 3 (missing verdict) but have valid `test_name` + `description`. These are structurally sound, just missing a verdict line:

- `summary_cf4_distance_grading.json`
- `summary_env_cf4_accel_binned.json`
- `summary_env_triple_distance.json`
- `summary_hierarchical_healing_length.json`
- `summary_inversion_distance_sensitivity.json`
- `summary_mass_split_bunching.json`
- `summary_void_gradient.json`
- `summary_distance_catalogs.json`
- `summary_pathological_galaxies.json`

These are legitimately quarantined per the classification rules (missing verdict = State 3), but they are **close to promotable**. Adding a verdict to each would move them to State 2.

---

## 3. Core Test Confirmation (15-25 Target)

### Confirmed Core (16 scripts — all correct)

The `CORE_SCRIPTS` list in `build_alignment_inventory.py` maps well to the alignment plan Section 6 claims:

**Claim: Environmental scatter in condensate regime**
1. `test_env_scatter_definitive.py` — primary evidence
2. `test_env_confound_control.py` — confound controls
3. `test_propensity_matched_env.py` — propensity matching

**Claim: Inversion point near g-dagger**
4. `test_mc_distance_and_inversion.py` — Monte Carlo distance + inversion
5. `test_nonparametric_inversion.py` — method-agnostic inversion
6. `test_binning_robustness.py` — binning sweep
7. `test_jackknife_robustness.py` — leave-one-out stability

**Claim: Discriminating null tests**
8. `test_lcdm_null_inversion.py` — LCDM mock comparison
9. `test_split_half_replication.py` — split-half replication

**Claim: Kurtosis spike / phase boundary**
10. `test_kurtosis_phase_transition.py` — kurtosis peak at g-dagger
11. `test_kurtosis_disambiguation.py` — instrumental vs physical kurtosis

**Claim: External validation**
12. `test_alfalfa_yang_btfr.py` — ALFALFA x Yang BTFR
13. `test_brouwer_lensing_rar.py` — Brouwer lensing RAR
14. `test_lensing_profile_shape.py` — NFW vs cored profile
15. `test_probes_inversion_replication.py` — PROBES dataset replication
16. `test_extended_rar_inversion.py` — multi-tier quality inversion

**Assessment**: The list is complete. No additions recommended. The 16 scripts cover all 5 claim categories with appropriate redundancy (multiple methods per claim).

### Tests That Should Stay State 2 (Supporting)

These 5 are correctly classified as supporting — valid but not core to the main narrative:
- `test_cluster_scale_rar.py` — cluster-scale RAR (different scale regime, interesting but not central)
- `test_cluster_sigma_scaling.py` — cluster sigma scaling
- `test_kurtosis_mhongoose.py` — kurtosis with independent MHONGOOSE data (corroborative)
- `test_soliton_nfw_composite.py` — model fitting exercise
- `test_tf_scatter_redshift.py` — Tully-Fisher systematics check

---

## 4. Verdict Quality Review

### Verdicts That Are Appropriate
- **BEC-CONSISTENT** (env_scatter, mc_distance, brouwer_lensing): Correctly states consistency rather than proof. Good.
- **REPLICATED** (extended_rar_inversion): Direct, factual. Good.
- **ROBUST** (jackknife, binning): Quantitative thresholds stated. Good.
- **NFW ADEQUATE** (lensing_profile_shape): Honest null result with appropriate caveats. Excellent.
- **COMPLEX** (alfalfa_yang_btfr): Acknowledges mismatch with simple BEC prediction. Honest.

### Verdicts That Need Attention

| Summary | Current Verdict | Concern | Recommended Action |
|---------|----------------|---------|-------------------|
| `summary_kurtosis_phase_transition.json` | DISCRIMINATING | Strong claim. Data shows 2428x ratio vs LCDM mock — impressive but depends on mock fidelity. | Consider softening to "STRONGLY_DISCRIMINATING" or add qualifier. Low priority — the data does support it. |
| `summary_env_confound_control.json` | INVERSION_FRAGILE | This is a CORE test with a negative/fragile result. **This is actually GOOD** — it shows intellectual honesty. | Keep as-is. Must appear on OSF to show the analysis acknowledges limitations. |
| `summary_propensity_matched_env.json` | INCONCLUSIVE | Another core test with a null result. | Keep as-is. Same reasoning — honest reporting strengthens credibility. |
| `summary_probes_inversion_replication.json` | NOT_REPLICATED | Core test that explicitly fails replication. | Keep as-is. This is the most important verdict in the entire suite for credibility. Hiding non-replication would be scientific misconduct. |
| `summary_split_half_replication.json` | STRONGLY_REPLICATED | Positive. No concern. | Keep. |
| `summary_jackknife_robustness.json` | Contains nested verdicts: inversion_point=ROBUST, environmental_scatter=FRAGILE | The top-level verdict says "ROBUST" but the nested verdict reveals environmental scatter is fragile under single-galaxy removal. | The top-level verdict should acknowledge BOTH outcomes. Consider: "MIXED — inversion robust, env scatter fragile under jackknife" |

### Key Observation on Negative Verdicts

Three core tests return negative/inconclusive results (INVERSION_FRAGILE, INCONCLUSIVE, NOT_REPLICATED). **These must appear in the OSF Core Tests section.** Omitting them would:
1. Misrepresent the strength of evidence
2. Undermine credibility if discovered during peer review
3. Violate scientific norms around reporting null results

The correct framing on OSF is: "Our analysis includes [N] core tests. [M] support BEC-consistency, [K] are inconclusive or show fragility, and [J] fail to replicate in independent datasets. We report all results."

---

## 5. Prioritized Fix List

### MUST FIX BEFORE OSF GOES PUBLIC

1. **Add `description` field to 7 core summaries + rename `test` to `test_name` in 1** — This unblocks 8 core scripts from State 3→1. Without this, half the core evidence is technically non-conforming.
   - Files: `summary_env_definitive.json`, `summary_nonparametric_inversion.json`, `summary_env_confound_control.json`, `summary_split_half_replication.json`, `summary_propensity_matched_env.json`, `summary_lcdm_null_inversion.json` (also rename `test`→`test_name`), `summary_kurtosis_disambiguation.json`, `summary_probes_inversion_replication.json`

2. **Include negative/inconclusive core verdicts on OSF** — INVERSION_FRAGILE, INCONCLUSIVE, NOT_REPLICATED must be visible in the OSF Core Tests page. Suppressing them is not an option.

3. **Fix jackknife verdict ambiguity** — Top-level verdict says ROBUST but nested verdict shows env scatter is FRAGILE. Either:
   - Change top-level to "MIXED — inversion robust, environmental scatter fragile"
   - Or add a `caveats` field the OSF page can display

4. **Remove `.pyc` files from tracking** — Add `__pycache__/` and `*.pyc` to `.gitignore`. Delete from git index.

5. **Verify the 47 State 4 summaries referenced by OSF HTML** — Many of these are on the OSF pages but have broken schemas. For each: either fix the summary (add description) to promote to State 2+, or remove the reference from OSF pages. Cannot have broken summaries cited as evidence.

### MUST FIX BEFORE JOURNAL SUBMISSION

6. **Add verdicts to 9 State 3 summaries** (cf4_distance_grading, env_cf4_accel_binned, env_triple_distance, hierarchical_healing_length, inversion_distance_sensitivity, mass_split_bunching, void_gradient, distance_catalogs, pathological_galaxies) — These are structurally sound but missing verdicts. Each needs a 1-line verdict.

7. **Audit the ~40 remaining State 4 summaries** that are referenced by OSF pages but NOT in the core list — determine which are worth fixing vs. which should be dereferenced from OSF. Many may be legacy/superseded tests that should be explicitly deprecated.

8. **Tighten kurtosis_phase_transition verdict** — "DISCRIMINATING" is defensible but strong. Consider qualifying with mock-fidelity caveat.

### NICE-TO-HAVE

9. **Improve `detect_type` classifier** — Add `documentation`, `config`, `build_artifact` types so the inventory is more informative. Doesn't affect state classification.

10. **Add `description` to all State 4 summaries** — Even for deprecated tests, having a description helps future readers understand what the test was for.

11. **Create a verdict taxonomy** — Standardize verdict strings across all summaries (e.g., BEC-CONSISTENT, REPLICATED, ROBUST, FRAGILE, INCONCLUSIVE, NOT_REPLICATED, DISCRIMINATING) so the OSF page can programmatically color-code them.

---

## 6. Inventory Methodology Notes

### What Codex Got Right
- The `CORE_SCRIPTS` list is well-chosen — maps cleanly to the 5 claim categories
- The deterministic classification avoids subjective judgment at this stage
- Using the OSF index mapping to link scripts→summaries is the correct approach
- Checking for both `test_name`+`description` (required) and `verdict` (recommended) is schema-aligned

### What Could Be Improved
- The `detect_type` catch-all ("table" for everything else) is too coarse
- No handling for `__pycache__/` or other build artifacts
- The `DEPRECATED_HINTS` set doesn't catch `__pycache__` or `.pyc`
- No distinction between "referenced by OSF pages" (public-facing) vs "referenced only by ARTIFACTS_INDEX" (internal) — this matters for prioritization

---

## 7. Recommended Next Steps

1. Codex fixes the 8 core summaries (add `description`, rename `test`→`test_name`)
2. Re-run `build_alignment_inventory.py` to verify State 1 count rises from 16 to 32
3. Codex produces the move plan using corrected inventory
4. Claude reviews the move plan before execution
