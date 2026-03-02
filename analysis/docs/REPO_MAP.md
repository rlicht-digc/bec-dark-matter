# REPO_MAP

## Purpose by Repository
- `/Users/russelllicht/bec-dark-matter`
  - Primary RAR=BEC science repo.
  - Owns Paper3 bridge code and `rar_points_unified.csv`.
  - **Branch C** and **Branch D** must run here.
- `/Users/russelllicht/bh-singularity`
  - BH-specific work only (thin-shell / BH traversal / singularity workflows).
  - Must not be used as runtime root for Paper3 or RAR=BEC Branch C/D.
- `/Users/russelllicht/Documents/New project`
  - Legacy/older workspace. Not part of active Branch C/D execution.

## Branch Entrypoints (Single Source of Truth)
- Branch C (Paper3 density bridge C1–C3):
  - `analysis/run_branchC_paper3_experiments.py`
  - Underlying science script: `analysis/paper3/paper3_branchC_experiments.py`
  - Output root: `outputs/paper3_high_density/BRANCHC_C1C2C3_<timestamp>/`
- Branch D (RAR control / SPARC / g† identifiability):
  - `analysis/run_branchD_rar_control.py`
  - Underlying science script: `analysis/pipeline/run_referee_required_tests.py`
  - Output root: `outputs/branchD_rar_control/<timestamp>/`

## Hard Execution Rules
- Run Branch C/D **only** from cwd `/Users/russelllicht/bec-dark-matter`.
- Every run must print and save a run stamp containing:
  - repo root path
  - git HEAD
  - dataset SHA256
  - output folder
- No silent overwrite:
  - outputs are timestamp-versioned
  - if existing CSV files are present in a target output directory, wrappers back them up before run
  - sidecar run metadata JSON is always written (`run_stamp_branchC.json` / `run_stamp_branchD.json`)

## BH Isolation / Vendor Policy
- Do **not** copy BH files into `bec-dark-matter` directly.
- If BH-origin code is required in `bec-dark-matter`:
  - place it under `vendor/bh_singularity/<original_relative_path>`
  - include provenance README (origin repo path, commit hash if known, date, reason)
  - update imports to use `vendor/` explicitly.
- Use `analysis/tools/repo_provenance_scan.py` before major Branch C/D runs to audit leakage risk.

## Operational Checklist
1. `cd /Users/russelllicht/bec-dark-matter`
2. Run provenance scan and review report in `analysis/results/`.
3. Run Branch C or Branch D entrypoint scripts (above).
4. Archive/run-review with stamped sidecar metadata.
