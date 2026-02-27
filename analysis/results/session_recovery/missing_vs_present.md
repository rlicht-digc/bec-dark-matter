# Missing vs Present Dataset Audit
Generated: 2026-02-23T12:36:38.793405

## Present and Verified

| Dataset | Path | Rows | Galaxies | Pts/Gal | Status |
|---------|------|------|----------|---------|--------|
| DEV Mixed | rar_profiles/20260222_201626/rar_points.parquet | 313,485 | 20,899 | 15 | CONTAMINATED |
| DEV Clean | rar_profiles/20260222_201626/rar_points_CLEAN.parquet | 150,000 | 3,000 | 50 | CLEAN_DEV |
| BIG Base | rar_profiles/20260223_061026_big_base/rar_points.parquet | 2,406,650 | 48,133 | 50 | CLEAN_BIG_BASE |
| DEV Catalog | rar_profiles/20260222_201626/meta/master_catalog.csv | 3,000 | - | - | OK |
| BIG Catalog | rar_profiles/20260223_061026_big_base/meta/master_catalog.csv | 48,133 | - | - | OK |
| Quality Counts | rar_profiles/20260222_201626/meta/galaxy_quality_counts.csv | 3,000 | - | - | OK |
| Selected IDs | rar_profiles/20260222_201626/meta/selected_base_subhalos_bins8_dm10.csv | 2,334 | - | - | OK |
| Scatter DM (big) | rar_profiles/20260223_061026_big_base/galaxy_scatter_dm.csv | 48,133 | - | - | OK |
| Mass Profiles | tng_mass_profiles.npz | - | - | - | OK |
| Fairness CSV | fairness_gap_vs_threshold.csv | - | - | - | OK |
| Extraction Log | big_extraction.log | - | - | - | COMPLETE |

## Missing or Not Needed

| Item | Status | Notes |
|------|--------|-------|
| 29k galaxy dataset | NOT NEEDED | 29122 was binned rows, not galaxies. Big base has 48133. |
| big_base_complete/ | EMPTY | Directory exists but contains 0 files. Was likely a staging dir. |
| BIG base quality counts | AVAILABLE | galaxy_scatter_dm.csv has dm_ok=45907 of 48133 |
| BIG base per-galaxy binning QC | NOT YET COMPUTED | Need to run bins_ok analysis on big_base |

## 29122 Ambiguity Resolution

The number 29122 has been used inconsistently. Resolution:

- In earlier analysis on the 3000-galaxy DEV set: 10 log-spaced bins in gbar were created.
- Total binned rows = 31,319 (some galaxies appear in multiple bins).
- After n-cut (requiring n>=3 per bin): 29,122 binned rows remained.
- This is BINNED ROWS, not unique galaxies.
- The big_extraction_29122.py script was named after this number but actually extracts 48,133 galaxies.
- The docstring explicitly states: Exhaustive search showed 29122 is not achievable from any simple selection.
- Actual selection: SubhaloFlag=1, Mstar>=1e8, n_star>=100, n_dm>=100, rhalf>0 => 48,133 galaxies.
