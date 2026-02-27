# yang_catalogs

## What this dataset is
- Group/galaxy catalogs used for ALFALFA×Yang environment matching.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `DESIDR9.y1.v1_galaxy.fits`
- `DESIDR9.y1.v1_group.fits`
- `SDSS7`
- `SDSS7_ID`
- `SDSS7_INFO`
- `SDSS7_SFR`
- `SDSS7_ST`
- `galaxy_DR7.tar.gz`

## Preprocessing scripts involved
- `analysis/pipeline/02_cf4_rar_pipeline.py`
- `analysis/pipeline/04_yang_crossmatch.py`
- `analysis/pipeline/05_wallaby_rar_pipeline.py`
- `analysis/pipeline/09_unified_rar_pipeline.py`
- `analysis/pipeline/load_extended_rar.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
