# mhongoose

## What this dataset is
- MHONGOOSE RAR point sets and per-galaxy extraction files.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `mhongoose_rar_all.tsv`
- `mhongoose_rar_points.tsv`
- `nkomo2025_eso444g084.tsv`
- `nkomo2025_kks2000-23.tsv`
- `sorgho2019_ngc3621.tsv`
- `sorgho2019_ngc7424.tsv`

## Preprocessing scripts involved
- `analysis/pipeline/integrate_mhongoose.py`
- `analysis/pipeline/integrate_mhongoose_sorgho.py`
- `analysis/pipeline/test_extended_rar_inversion.py`
- `analysis/pipeline/test_kurtosis_mhongoose.py`
- `analysis/pipeline/test_ngc3621_single_galaxy.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
