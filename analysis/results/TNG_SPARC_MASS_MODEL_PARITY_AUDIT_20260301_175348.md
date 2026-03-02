# TNG↔SPARC Mass-Model Parity Audit
Generated (UTC): 2026-03-01T22:53:49.380227Z
## 1) Executive Summary
Current TNG `rar_points` products are **not yet mass-model-parity-equivalent** to SPARC for a strict apples-to-apples discriminant. Evidence indicates truth-level enclosed-mass accelerations with fixed 50-point radial grids and no observer-mimic transform. Recommendation: **do not treat current products as parity-clean for final discriminant**; first fix parity definition in-repo, then run Tier A strict, then Tier B observer-mimic as sensitivity.
## 2) What SPARC gbar/gobs Means (Code-Referenced)
In the unified pipeline, SPARC is built from observed RC decomposition: `gobs = Vobs^2 / R * conv` and `gbar = Vbar^2 / R * conv`, where `Vbar^2 = sign(Vgas)*Vgas^2 + Y_disk*Vdisk^2 + Y_bulge*Vbul^2` (see `analysis/pipeline/09_unified_rar_pipeline.py`, around lines ~986-988 and ~715-717). This is disk-decomposition style, with explicit M/L assumptions.
## 3) What TNG gbar/gobs Means Today
### 3.1 Current TNG Inputs Located
- `/Users/russelllicht/bec-dark-matter/raw_data/tng/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/rar_points_CLEAN.parquet`
  - sha256: `98c08968891ed21b5bc0e8dec5715b3863da4b21966ceffa54783afa4b6974fe`
  - rows=150000, cols=6, n_gal=3000
  - points/gal min/med/max = 50/50.0/50
  - R_kpc global min/max = 1.5/200
- `/Users/russelllicht/bec-dark-matter/raw_data/tng/TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED/rar_points.csv`
  - sha256: `9c07d54fce9dd2839244481ae5a135bbdedfa4485469d52a8258f52d3b9438a4`
  - rows=150000, cols=6, n_gal=3000
  - points/gal min/med/max = 50/50.0/50
  - R_kpc global min/max = 1.5/200
- `/Users/russelllicht/bec-dark-matter/raw_data/tng/TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE/rar_points.parquet`
  - sha256: `25d2169920829bc21fa48302c06351f208a3e18246a1f82d185d6499c1ffdbb3`
  - rows=2406650, cols=6, n_gal=48133
  - points/gal min/med/max = 50/50.0/50
  - R_kpc global min/max = 1.5/200
- `/Users/russelllicht/bec-dark-matter/raw_data/tng/quarantine_contaminated/rar_points_20899x15_CONTAMINATED.parquet`
  - sha256: `e3af2cc1c45959597c91d2904b0cb236be1b6f233667d5380baee0d6722d3b28`
  - rows=313485, cols=6, n_gal=20899
  - points/gal min/med/max = 15/15.0/15
  - R_kpc global min/max = 0.250012/652.041
- `/Users/russelllicht/bec-dark-matter/raw_data/tng/quarantine_contaminated/tng_mass_profiles_CONTAMINATED_20899_session.npz`
  - sha256: `b11cfa2b779181c169877fe331bab935a6173b7d2bf5b3b641913cf1c6d21c60`

### 3.2 Provenance Trace
- Staging/provenance files point to an external extractor: `big_extraction_29122.py` via `raw_data/tng/ingestion_staging/repro_commands.sh` and `registry_summary.json`.
- That writer script is **not present in this repo**, so exact production code for current `rar_points.parquet` cannot be audited line-by-line here.
- In-repo TNG profile code (`analysis/pipeline/tng_extract_profiles.py` + `analysis/pipeline/test_tng_coherence.py`) uses enclosed-mass accelerations: `gbar = G*M_bar(<R)/R^2`, `gobs = G*M_tot(<R)/R^2`. Current column set (`log_gbar`, `log_gobs`, `log_gDM`, `lowres_flag`) is consistent with this truth-style construction.
## 4) Comparability Scorecard
| Criterion | Status | Audit determination |
|---|---|---|
| Disk geometry parity with SPARC Vbar decomposition | FAIL | Current TNG products appear enclosed-mass/spherical-style truth accelerations; no verified disk-plane decomposition writer in-repo. |
| Observer-mimic (projection, M/L scatter, RC sampling/noise) | FAIL | No evidence in current production lineage for observer-side transform. |
| Radial sampling comparability | FAIL (raw), PARTIAL MITIGATION | Raw TNG is fixed 50 pts/gal; SPARC is irregular. New strict driver has pairwise K-matching mitigation. |
| gobs definition consistency | PASS_WITH_CAVEAT | Truth-level total acceleration is internally consistent, but not observer-measured Vobs under projection/systematics. |
## 5) Run-Order Decision (Tier A vs Tier B)
**Decision: `fix_then_A`**
- Do **not** treat current TNG points as final parity-clean input for the discriminant yet.
- First implement minimal Tier-A parity correction (disk-consistent `gbar`/`gobs` definitions and in-repo writer provenance).
- Then run Tier A strict (same estimator + matched sampling + K matching).
- Then run Tier B observer-mimic as an additional realism/sensitivity tier.
## 6) Minimal Required Changes (Actionable)
1. **Version-control the canonical TNG writer** (`big_extraction_29122.py` equivalent) inside repo; stamp exact formulas and units in manifest.
2. **Add disk-consistent TNG acceleration build path** (`gbar=Vbar^2/R`, `gobs=Vtot^2/R`) before strict feature comparison.
3. **Keep pairwise K matching mandatory** (already in `analysis/tng/matching.py` and `analysis/tng/tng_sparc_feature_repro.py`) and report K_eff distribution.
4. **Add Tier-B observer-mimic module** (projection + inclination + M/L scatter + RC sampling/noise) for sensitivity analysis, after Tier-A parity run.
