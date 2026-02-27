# probes

## What this dataset is
- Expanded galaxy profile bundle used for extended inversion replication tests.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `README_profiles`
- `README_tables`
- `main_table.csv`
- `model_fits.csv`
- `profiles/profiles/2MASSJ00385299-5529210_f.aux`
- `profiles/profiles/2MASSJ00385299-5529210_f.prof`
- `profiles/profiles/2MASSJ00385299-5529210_g.aux`
- `profiles/profiles/2MASSJ00385299-5529210_g.prof`

## Preprocessing scripts involved
- `analysis/pipeline/09_unified_rar_pipeline.py`
- `analysis/pipeline/expand_rotation_curves.py`
- `analysis/pipeline/probes_reader/load_galaxy.py`
- `analysis/pipeline/probes_reader/read_galaxies.py`
- `analysis/pipeline/test_extended_rar_inversion.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
