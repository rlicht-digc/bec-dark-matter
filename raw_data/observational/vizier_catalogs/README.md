# vizier_catalogs

## What this dataset is
- General VizieR-ingested tables used across preprocessing and validation steps.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `boselli1997_table6_sample.tsv`
- `bouquin2018_table1.tsv`
- `bouquin2018_table1_sample.tsv`
- `bouquin2018_table3.tsv`
- `bouquin2018_table3_sample.tsv`
- `deblok2002_processed.tsv`
- `deblok2002_raw.tsv`
- `diazgarcia2016_s4g_sample.tsv`

## Preprocessing scripts involved
- `analysis/pipeline/assess_vizier_catalogs.py`
- `analysis/pipeline/load_extended_rar.py`
- `analysis/pipeline/match_bouquin_photometry.py`
- `analysis/pipeline/match_korsaga_massmodels.py`
- `analysis/pipeline/match_photometry.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
