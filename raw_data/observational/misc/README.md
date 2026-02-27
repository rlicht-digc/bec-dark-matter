# misc

## What this dataset is
- Auxiliary catalogs (group lists, inclinations, survey properties) supporting pipeline joins.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `hyperleda_inclinations.tsv`
- `kourkchi2017_all_groups.tsv`
- `kourkchi2017_galaxies.tsv`
- `kourkchi2017_groups.tsv`
- `kourkchi2017_massive_groups.tsv`
- `manga_nsa_properties.tsv`
- `z0mgs_leroy2019_masses.tsv`

## Preprocessing scripts involved
- `analysis/pipeline/test_env_cf4_accel_binned.py`
- `analysis/pipeline/test_env_scatter_definitive.py`
- `analysis/pipeline/test_void_gradient.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
