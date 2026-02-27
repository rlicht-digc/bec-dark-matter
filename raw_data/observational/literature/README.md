# literature

## What this dataset is
- Extracted literature tables used in replication and supplementary analyses.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `arXiv-2111.00937v1_extracted/extracted_tables_index.csv`
- `arXiv-2111.00937v1_extracted/tab_VERTICO-sample.csv`
- `arXiv-2111.00937v1_extracted/tab_co_props.csv`
- `arXiv-2111.00937v1_extracted/tab_co_radii.csv`
- `arXiv-2111.00937v1_extracted/tab_mass-size_bestfit.csv`

## Preprocessing scripts involved
- `analysis/pipeline/04_yang_crossmatch.py`
- `analysis/pipeline/05_wallaby_rar_pipeline.py`
- `analysis/pipeline/06_wallaby_environments.py`
- `analysis/pipeline/build_quality_subsample.py`
- `analysis/pipeline/expand_rotation_curves.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
