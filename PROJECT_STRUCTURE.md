# Project Structure (Clean Layout)

## Main project (BEC/RAR)
- `analysis/`: analysis code + non-BH outputs
- `bec_rar_identity/`: canonical BEC<->RAR hub (tests, artifacts, manifests)
- `docs/`: active docs, reports, and legacy update notes
- `figures/`: non-BH figure outputs
- `rerun_outputs/`: referee rerun bundles

## Raw data (all source payloads)
- `raw_data/`
  - `tng/`: TNG source datasets and staging
  - `observational/`: SPARC, clusters, CF4, WALLABY, ALFALFA, etc.
  - `observational/by_system/`: quick links for Virgo/Coma/Sagittarius/Fornax
  - `archives/`: raw archive bundles

## BH singularity project (separate local repo)
- Local repo root: `/Users/russelllicht/bh-singularity`
- In this repo, `subprojects/bh_singularity` is a symlink to that external repo.

## Compatibility symlinks
These legacy paths are preserved so older scripts still resolve:
- `data` -> `raw_data/observational`
- `datasets` -> `raw_data/tng`
- `Remaining_TNG` -> `raw_data/tng/ingestion_staging`
- `subprojects/bh_singularity` -> `../../bh-singularity`
- `outputs_bh` -> `subprojects/bh_singularity/outputs/outputs_bh`
- `outputs_bh3` -> `subprojects/bh_singularity/outputs/outputs_bh3`
- `run_bh_viability.py` -> `subprojects/bh_singularity/scripts/run_bh_viability.py`
- `run_bh_viability_modes.py` -> `subprojects/bh_singularity/scripts/run_bh_viability_modes.py`
