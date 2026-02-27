# environment

## What this dataset is
- Derived environment catalogs for SPARC and WALLABY objects.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `sparc_environment_catalog.csv`
- `sparc_environment_catalog.json`
- `wallaby_environment_catalog.csv`
- `wallaby_environment_catalog.json`

## Preprocessing scripts involved
- `analysis/pipeline/02_cf4_rar_pipeline.py`
- `analysis/pipeline/04_yang_crossmatch.py`
- `analysis/pipeline/05_wallaby_rar_pipeline.py`
- `analysis/pipeline/06_wallaby_environments.py`
- `analysis/pipeline/07_comprehensive_comparison.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
