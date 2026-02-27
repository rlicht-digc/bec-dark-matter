# hi_surveys

## What this dataset is
- Aggregated H I survey and rotation-curve tables for cross-validation checks.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `Catinella2005_Palomar_lowV_galaxies.tsv`
- `Fathi2009_NGC4294_Halpha_RC.tsv`
- `Fathi2009_NGC4519_Halpha_RC.tsv`
- `Koopmann2006_all_Virgo_galaxies.tsv`
- `Koopmann2006_target_galaxies.tsv`
- `Lang2020_NGC4293_RC.tsv`
- `Lang2020_Virgo_galaxies_RC.tsv`
- `Lang2020_all_Virgo_RC.tsv`

## Preprocessing scripts involved
- `analysis/pipeline/08_expanded_rar_pipeline.py`
- `analysis/pipeline/09_unified_rar_pipeline.py`
- `analysis/pipeline/fetch_catinella2005_cf4.py`
- `analysis/pipeline/fetch_vogt2004_cf4.py`
- `analysis/pipeline/test_cluster_scale_rar.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
