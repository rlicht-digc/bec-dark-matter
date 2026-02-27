# wallaby

## What this dataset is
- WALLABY source and kinematic catalogs used for environment and cross-check analyses.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `wallaby_desi_crossmatch.json`
- `wallaby_dr1_kinematic_catalogue.csv`
- `wallaby_dr1_source_catalogue.csv`
- `wallaby_dr2_high_res_catalogue.csv`
- `wallaby_dr2_kinematic_catalogue.csv`
- `wallaby_dr2_source_catalogue.csv`
- `wallaby_hydra_wang2021.csv`
- `wallaby_reynolds2022_detected.csv`

## Preprocessing scripts involved
- `analysis/pipeline/05_wallaby_rar_pipeline.py`
- `analysis/pipeline/06_wallaby_environments.py`
- `analysis/pipeline/07_comprehensive_comparison.py`
- `analysis/pipeline/08_expanded_rar_pipeline.py`
- `analysis/pipeline/09_unified_rar_pipeline.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
