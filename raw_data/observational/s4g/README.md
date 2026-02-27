# s4g

## What this dataset is
- S4G stellar maps used in morphology and profile cross-checks.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `IC1954.stellar.fits`
- `IC5273.stellar.fits`
- `NGC0628.stellar.fits`
- `NGC0685.stellar.fits`
- `NGC1087.stellar.fits`
- `NGC1097.stellar.fits`
- `NGC1300.stellar.fits`
- `NGC1365.stellar.fits`

## Preprocessing scripts involved
- `analysis/pipeline/load_extended_rar.py`
- `analysis/pipeline/match_bouquin_photometry.py`
- `analysis/pipeline/match_korsaga_massmodels.py`
- `analysis/pipeline/match_photometry.py`
- `analysis/pipeline/test_extended_rar_inversion.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
