# sparc

## What this dataset is
- Primary galaxy sample and rotation/mass-model data for core RAR analyses.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `MassModels.zip`
- `SPARC_Lelli2016c.mrt`
- `SPARC_ReadMe.txt`
- `SPARC_table1_vizier.dat`
- `SPARC_table2_rotmods.dat`
- `rotmods_vizier.zip`
- `sparc_coordinates.json`
- `sparc_pgc_crossmatch.csv`

## Preprocessing scripts involved
- `analysis/classical_wave_mimicry_analysis.py`
- `analysis/gdagger_hunt.py`
- `analysis/pipeline/01_resolve_pgc_numbers.py`
- `analysis/pipeline/02_cf4_rar_pipeline.py`
- `analysis/pipeline/03_fetch_cf4_distances.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
