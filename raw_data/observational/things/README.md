# things

## What this dataset is
- Rotation-curve profiles used for extended observational consistency checks.

## What it is used for in this project
- Input data for RAR↔BEC analysis workflows and reproducibility checks.
- Referenced by summary outputs in `analysis/results/` and related figures.

## Key files used
- `curves/DDO154.curve.02`
- `curves/IC2574.curve.02`
- `curves/NGC2366.curve.02`
- `curves/NGC2403.curve.02`
- `curves/NGC2841.curve.02`
- `curves/NGC2903.curve.02`
- `curves/NGC2976.curve.02`
- `curves/NGC3031.curve.02`

## Preprocessing scripts involved
- `analysis/pipeline/09_unified_rar_pipeline.py`
- `analysis/pipeline/analysis_tools.py`
- `analysis/pipeline/expand_rotation_curves.py`
- `analysis/pipeline/integrate_mhongoose_sorgho.py`
- `analysis/pipeline/load_extended_rar.py`

## Known restrictions
- Third-party dataset terms apply.
- See `TERMS_OF_USE.md` for source links, required citations, and redistribution policy.
