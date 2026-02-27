# Big Dataset Validation Report

**Timestamp:** 2026-02-23T17:53:32.807209

**Candidate:** `/home/tnguser/rar_profiles/20260223_061026_big_base/rar_points.parquet`

**Result:** VALID ✅

## Counts

| Metric | Expected | Actual | Match |
|--------|----------|--------|-------|
| Total rows | 2406650 | 2406650 | ✅ |
| Unique galaxies | 48133 | 48133 | ✅ |
| All 50 pts/gal | True | True | ✅ |
| Required cols | All present | All present | ✅ |

## Points Per Galaxy

- Min: 50
- Median: 50.0
- Max: 50
- Mean: 50.00

## Columns

- `r_kpc` (✅ required)
- `log_gbar` (✅ required)
- `log_gobs` (✅ required)
- `log_gDM` (✅ required)
- `lowres_flag` (✅ required)
- `SubhaloID` (✅ required)
