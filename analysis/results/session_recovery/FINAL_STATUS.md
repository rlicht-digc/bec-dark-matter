# FINAL STATUS - Session Recovery Audit
Generated: 2026-02-23T12:50:29.717106

## Summary Table

| Category | Item | Status |
|----------|------|--------|
| VERIFIED | DEV Clean (3000x50) | rar_profiles/20260222_201626/rar_points_CLEAN.parquet - 150,000 rows |
| VERIFIED | BIG Base (48133x50) | rar_profiles/20260223_061026_big_base/rar_points.parquet - 2,406,650 rows |
| VERIFIED | DEV Master Catalog | 3,000 rows, 12 columns |
| VERIFIED | BIG Master Catalog | 48,133 rows, 7 columns |
| VERIFIED | Quality Counts (dev) | 3,000 rows, bins_ok + n_dm_pts |
| VERIFIED | Selected IDs (bins8_dm10) | 2,334 galaxies from dev set |
| VERIFIED | DM Scatter (big) | 48,133 rows, dm_ok=45,907 |
| VERIFIED | Extraction Log | 48133 ok, 0 failed, 5891.5s |
| FLAGGED | DEV Mixed parquet | CONTAMINATED: 20899x15, kept in place |
| RESOLVED | 29122 ambiguity | Was binned rows from dev set, NOT galaxy count |
| QUARANTINED | big_base_complete/ | Empty dir moved |
| QUARANTINED | Temp scripts | Session recovery temp files moved |
| NOT NEEDED | 29k galaxy dataset | Big base has 48,133 galaxies (exceeds target) |

## Newly Generated Files (in outputs/session_recovery/)

- registry_runs.csv, registry_summary.json
- quality_cut_reconstruction.csv
- selected_ids_catalog.csv
- missing_vs_present.md
- repro_commands.sh
- cleanup_plan.csv, cleanup_executed.csv
- FINAL_STATUS.md

## Quarantined (in _quarantine_session_recovery/)

- big_base_complete/ (empty dir)
- rar_profiles/Untitled.ipynb
- .ipynb_checkpoints from data dirs
- Temporary phase scripts

## Long-Running Jobs: None. Big base extraction completed.

## Next Command:
python3 /home/tnguser/scripts/session_recovery/compute_big_base_qc.py
