Drop files from Jupyter here, then run:

python3 /Users/russelllicht/bec-dark-matter/analysis/pipeline/organize_remaining_tng.py --dry-run
python3 /Users/russelllicht/bec-dark-matter/analysis/pipeline/organize_remaining_tng.py

Expected inputs include:
- session recovery outputs (registry_runs.csv, registry_summary.json, ...)
- compute_big_base_qc.py
- optional archives (.tgz/.tar/.zip)
- optional big-base dataset files (rar_points.parquet, master_catalog.csv, etc)

This organizer copies files into canonical repo locations and writes logs under:
/Users/russelllicht/bec-dark-matter/analysis/results/session_recovery/
