# TNG↔SPARC Feature Reproduction Audit (Forensic)

Generated: 2026-03-01T21:38:07Z (UTC)
Repo: `/Users/russelllicht/bec-dark-matter`
Scope: code/read-only audit only (no expensive re-runs)

## Executive determination
- **Strict test status:** **Does not exist in full**.
- **Partial coverage exists:** yes (inversion + matching/composition/coherence fragments).
- **Missing for referee-proof symmetric discriminant:** unified apples-to-apples pipeline that computes **both** inversion and kurtosis features on **SPARC and SPARC-matched TNG analogs** using the **same residual/bins/statistics codepath**, with split-half/null diagnostics and provenance hashes.

## What exists (with paths)

### 1) Inversion-related TNG/SPARC code (partial)
1. `analysis/pipeline/test_lcdm_hydro_inversion.py`
- Uses Marasco+2020 EAGLE/TNG property tables (`data/eagle_rar/tablea1e.dat`, `tablea1t.dat`) and computes scatter-derivative inversion with offset sweep.
- Outputs `analysis/results/summary_lcdm_hydro_inversion.json`.
- Gap vs strict target: builds semi-analytic profiles from catalog properties, not a matched TNG analog sample under the same canonical feature pipeline.

2. `analysis/pipeline/test_matched_analog_comparison.py`
- Matches SPARC galaxies to EAGLE/TNG analogs (M* + Rhalf), computes inversion and offset robustness; includes ACF/scatter comparisons.
- Outputs `analysis/results/summary_matched_analog_comparison.json`, `analysis/results/eagle_matched_analogs.csv`, `analysis/results/tng_matched_analogs.csv`, and figures under `figures/matched_analog_*.png`.
- Gap vs strict target:
  - No kurtosis spike (κ4) feature + bootstrap CI.
  - No split-half replication / null-shuffle proximity test for TNG feature.
  - Uses duplicated local implementations rather than a single canonical shared feature module.
  - TNG branch depends on 4-aperture HDF5 (`~/Desktop/tng_cross_validation/aperture_masses.hdf5`) and even notes resolution limitations.

3. `analysis/pipeline/test_tng_sparc_composition_sweep.py`
- Composition discrimination sweep using residual-distribution metrics (Mann-Whitney, Cliff’s delta, median CI).
- Outputs `analysis/results/tng_sparc_composition_sweep/composition_ranking.csv` and `summary_tng_sparc_composition_sweep.json`.
- Gap: not inversion/kurtosis feature reproduction test.

4. `analysis/pipeline/test_tng_sparc_fairness_gap.py`
- Mass-matched fairness sweep (per-galaxy residual scatter).
- Outputs `analysis/results/tng_sparc_composition_sweep/fairness_gap_vs_threshold.csv` and `summary_fairness_gap_vs_threshold.json` plus plots.
- Gap: not inversion/kurtosis reproduction.

### 2) Kurtosis code (SPARC/Tier-K/LCDM mock; no TNG reproduction)
1. `analysis/pipeline/test_kurtosis_phase_transition.py`
2. `analysis/pipeline/test_kurtosis_disambiguation.py`
3. `analysis/pipeline/test_kurtosis_mhongoose.py`

These compute κ4 profiles and bootstrap CIs for SPARC/Tier-K and mock controls, but **do not run a SPARC-matched TNG analog κ4 reproduction test**.

### 3) Coherence code (different feature family)
1. `analysis/pipeline/test_simulation_coherence.py` (EAGLE vs SPARC; TNG noted as insufficient in-script)
2. `analysis/pipeline/test_tng_coherence.py` (15-radii TNG profiles from external extraction)

These target ACF/periodicity, not inversion+kurtosis parity.

## Feature-by-feature strictness check

### Feature A: Inversion point near g† with bin/offset stability
- **Implemented partially.**
- Evidence:
  - `test_lcdm_hydro_inversion.py` computes inversion + 10 offsets.
  - `test_matched_analog_comparison.py` computes inversion + 10 offsets for SPARC/EAGLE/TNG matched analog sets.
- Why not full strict test:
  - Not one canonical shared pipeline module for both datasets.
  - Matching and data-generation choices differ across scripts.
  - No referee-proof provenance bundle for this exact symmetric question.

### Feature B: Kurtosis spike κ4 near g† with bootstrap CI
- **Missing for TNG/SPARC matched analog comparison.**
- Existing kurtosis scripts are SPARC/Tier-K/mock centric; no matched TNG analog κ4 run artifact found.

### Feature C (optional): Split-half replication (galaxy-block)
- **Missing for TNG analog feature test.**
- SPARC-only split-half exists: `analysis/pipeline/test_split_half_replication.py`.

### Feature D (optional): Null/shuffle proximity tests to g†
- **Missing for TNG analog feature test.**
- Null/shuffle machinery exists in SPARC-focused g† hunt tests (`analysis/tests/test_gdagger_hunt_refereeproof.py`) but not integrated for TNG matched analog inversion/kurtosis comparison.

## Existing runnable entrypoints (current code)

### Command set 1: hydro inversion fragment
```bash
python3 analysis/pipeline/test_lcdm_hydro_inversion.py
```
Required inputs:
- `data/eagle_rar/tablea1e.dat`
- `data/eagle_rar/tablea1t.dat`
- `data/sparc/SPARC_table2_rotmods.dat`
- `data/sparc/SPARC_Lelli2016c.mrt`

Outputs:
- `analysis/results/summary_lcdm_hydro_inversion.json`

### Command set 2: matched analog inversion fragment
```bash
python3 analysis/pipeline/test_matched_analog_comparison.py
```
Required inputs:
- `data/sparc/SPARC_table2_rotmods.dat`
- `data/sparc/SPARC_Lelli2016c.mrt`
- `data/eagle_rar/eagle_aperture_masses.json`
- `~/Desktop/tng_cross_validation/aperture_masses.hdf5` (for TNG branch)

Outputs:
- `analysis/results/summary_matched_analog_comparison.json`
- `analysis/results/eagle_matched_analogs.csv`
- `analysis/results/tng_matched_analogs.csv` (if TNG available)
- `figures/matched_analog_*.png`

### Command set 3: composition/fairness discriminant fragments
```bash
python3 analysis/pipeline/test_tng_sparc_composition_sweep.py \
  --tng-input rar_points.parquet --bootstrap 400

python3 analysis/pipeline/test_tng_sparc_fairness_gap.py \
  --tng-input rar_points.parquet \
  --tng-profiles tng_mass_profiles.npz \
  --massmatch-caliper 0.20
```
Required inputs:
- TNG points: parquet/csv/npz (default in-script fallbacks)
- SPARC tables in `data/sparc/`
- `tng_mass_profiles.npz` for fairness mass matching

Outputs:
- `analysis/results/tng_sparc_composition_sweep/*`

## Local availability check (this workspace)
- Present:
  - `data/sparc/SPARC_table2_rotmods.dat`
  - `data/sparc/SPARC_Lelli2016c.mrt`
  - `data/eagle_rar/eagle_aperture_masses.json`
  - `data/eagle_rar/tablea1e.dat`
  - `data/eagle_rar/tablea1t.dat`
  - `rar_points.parquet`
- Missing locally:
  - `tng_mass_profiles.npz`
  - `~/Desktop/tng_cross_validation/aperture_masses.hdf5`
  - `~/Desktop/tng_cross_validation/LATEST_GOOD/rar_points.parquet`

## What is missing for a true referee-proof apples-to-apples test

### Required minimal additions (skeleton only)
1. `analysis/tng/adapters.py`
- `load_tng_points(...) -> DataFrame[galaxy_id, source, log_gbar, log_gobs, R_kpc, logMstar, Rhalf_kpc]`
- `load_sparc_points(...) -> same schema`
- Include deterministic file-hash capture (`sha256`) and row-count checks.

2. `analysis/tng/matching.py`
- Galaxy-level matching (nearest-neighbor or propensity) with explicit covariates:
  - `logMstar`, `Rhalf` (or size proxy), `n_radii`, `gbar_min/max` coverage.
- Emit balance diagnostics (SMD/KS per covariate) and matched index tables.

3. `analysis/tng/tng_sparc_feature_repro.py`
- One canonical runner using shared functions from `analysis/pipeline/analysis_tools.py`:
  - inversion via `find_inversion_point`
  - kurtosis profile via `binned_stats`
  - kurtosis CI via `bootstrap_kurtosis`
- Same residual definition and same bin configuration for SPARC and TNG analogs.
- Optional diagnostics toggles: split-half, shuffle/null.
- CLI:
  - `--smoke`
  - `--seed`
  - `--n-bootstrap`
  - `--offset-grid`
  - `--match-method`/`--caliper`
  - `--outdir`

4. `analysis/tests/test_tng_sparc_feature_repro_smoke.py`
- Fast deterministic smoke test (small matched sample, small bootstrap).
- Verifies schema, hashes, reproducibility (repeat run equality), and no NaNs in key metrics.

5. `analysis/results/summary_tng_sparc_feature_repro.json` (produced artifact)
- Must include:
  - dataset hashes (SPARC + TNG)
  - git head
  - run timestamp/config
  - matching diagnostics
  - inversion metrics (location, Δ from g†, offset stability)
  - kurtosis metrics (κ4 peak location, κ4@g†, bootstrap CIs)
  - optional split-half and null/shuffle stats if enabled
  - explicit verdict string and caveats

### Data products required
1. TNG per-galaxy radial table with at minimum:
- `galaxy_id`, `log_gbar`, `log_gobs`, `R_kpc`
- plus matching covariates (`logMstar`, size proxy `Rhalf_kpc`)

2. SPARC points in the same canonical schema.

3. Matched analog map:
- SPARC galaxy ↔ TNG analog IDs
- distance/caliper metrics
- balance diagnostics

4. Exact shared pipeline parameters:
- bin edges/range, bin width, min points per bin
- residual definition function (single imported function)
- offset grid used for inversion stability

### Referee-proof acceptance criteria
1. **Pipeline identity check:** SPARC and TNG branches call the same imported feature functions (not duplicated custom math).
2. **Matching transparency:** balance table included; unmatched fractions reported.
3. **Reproducibility:** fixed seed + hashes + git head + run stamp.
4. **Smoke mode:** completes quickly and writes full summary schema.
5. **No hardcoded narrative:** summary JSON is source of truth for report text/figures.

## Bottom line
- The repository has strong partial groundwork, but **does not yet contain the full symmetric “TNG reproduces SPARC features?” test in one referee-proof pipeline**.
- Minimum safe path is to add a thin, canonical wrapper layer around existing reusable components (`analysis_tools`) plus explicit matching/provenance/test harness.
