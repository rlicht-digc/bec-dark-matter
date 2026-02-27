# cf4

## What this dataset is
- Distance-cache products for Cosmicflows-style environment classification workflows.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `catinella2005_cf4_cache.json`
- `cf4_distance_cache.json`
- `vogt2004_cf4_cache.json`
- `wallaby_cf4_cache.json`

## Preprocessing scripts involved
- `analysis/pipeline/01_resolve_pgc_numbers.py`
- `analysis/pipeline/02_cf4_rar_pipeline.py`
- `analysis/pipeline/03_fetch_cf4_distances.py`
- `analysis/pipeline/04_yang_crossmatch.py`
- `analysis/pipeline/05_wallaby_rar_pipeline.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
