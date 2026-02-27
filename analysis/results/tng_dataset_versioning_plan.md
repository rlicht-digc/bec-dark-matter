# TNG Dataset Versioning Plan (In Progress)

Date: 2026-02-22 (local)

## Objective
Run TNG vs SPARC comparisons without mixed-run contamination by enforcing dataset IDs, manifests, and lineage audit.

## Canonical dataset targets
- `TNG_RAR_3000x50_SOFT1p5_RUN201626` (dev/sanity, fast iteration)
- `TNG_RAR_29122x50_SOFT1p5_RUNYYYYMMDD_HHMMSS` (final large sample, pending generation)

## Completed now
- Added manifest + lineage tool:
  - `/Users/russelllicht/bec-dark-matter/analysis/pipeline/tng_dataset_lineage.py`
- Wrote manifests for currently available datasets:
  - `/Users/russelllicht/TNG_RAR_LATEST_GOOD/meta/dataset_manifest.json`
  - `/Users/russelllicht/bec-dark-matter/meta/dataset_manifest.json`
- Ran lineage audit on analysis outputs:
  - `/Users/russelllicht/bec-dark-matter/analysis/results/dataset_lineage_audit.csv`
  - `/Users/russelllicht/bec-dark-matter/analysis/results/dataset_lineage_audit.json`

## Current findings
- Dataset `3000x50` exists and is internally consistent for `rar_points + master_catalog`.
- `galaxy_scatter_dm.csv` in `3000x50` covers 2402/3000 subhalos (598 missing); likely due to an internal minimum-point or quality filter.
- Dataset `20899x15` exists and is consistent (`20899 * 15 = 313485` points).
- Most existing outputs are missing explicit `dataset_id` lineage fields (audit found 139/139 scanned files missing ID tagging).

## Pending (blocked on incoming data)
- Build final large dataset:
  - `TNG_RAR_29122x50_SOFT1p5_RUN...`
  - required files:
    - `rar_points.parquet`
    - `galaxy_scatter_dm.csv`
    - `meta/master_catalog.csv`
    - `meta/dataset_manifest.json`
- Confirm frozen extraction/selection settings before full run:
  - `SOFT_KPC`
  - radii grid policy (0.5x–5x R_half)
  - galaxy selection/quality cuts
  - DM-dominated threshold policy for downstream scatter metrics

## Next actions after data arrives
1. Generate/validate manifest for `29122x50`.
2. Re-run fairness/composition sweeps with filename lineage:
   - `...__<dataset_id>.csv`
3. Re-run matched tests with explicit columns:
   - `dataset_id`, `log_gbar_cut`, `MIN_DM_PTS`, `rmin_kpc`
4. Re-run lineage audit in strict mode and fail any output without dataset ID.

