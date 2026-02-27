# alfalfa

## What this dataset is
- ALFALFA source catalogs used for BTFR and environment cross-match analyses.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `alfalfa_alpha100_haynes2011.tsv`
- `alfalfa_alpha100_haynes2018.tsv`

## Preprocessing scripts involved
- `analysis/pipeline/09_unified_rar_pipeline.py`
- `analysis/pipeline/load_extended_rar.py`
- `analysis/pipeline/match_bouquin_photometry.py`
- `analysis/pipeline/match_korsaga_massmodels.py`
- `analysis/pipeline/match_photometry.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
