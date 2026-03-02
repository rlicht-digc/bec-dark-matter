# AWS Drop Bundle (bec-dark-matter)

This folder is a self-contained staging bundle for reproducible EC2 runs.

## What This Includes
- EC2 bootstrap scripts for Ubuntu + Python venv.
- Deterministic run wrappers for:
  - Branch C (Paper3 C1-C3)
  - Branch D (RAR refereeproof)
  - strict TNG↔SPARC feature reproduction (if TNG input is present)
- Input/output verification scripts.
- S3 pull/push helpers.
- Config template (`RUN_CONFIG.template.json`).
- SHA256 manifest (`MANIFEST.sha256`).

## Preconditions
1. Clone repo on EC2.
2. Place this `aws_drop/` folder at repo root.
3. Ensure AWS credentials are available (`aws configure` or IAM role).

## Quick Start on EC2
```bash
cd /path/to/bec-dark-matter
bash aws_drop/bootstrap/ubuntu_bootstrap.sh
source .venv/bin/activate
source aws_drop/bootstrap/set_threads.sh

export S3_BUCKET="your-bucket"
export S3_PREFIX="bec-dark-matter"
bash aws_drop/sync/s3_pull_inputs.sh

bash aws_drop/verify/verify_inputs.sh
bash aws_drop/run/run_smoke_all.sh
# full runs:
bash aws_drop/run/run_branchC_full.sh
bash aws_drop/run/run_branchD_full.sh
bash aws_drop/run/run_tng_sparc_feature_repro_full.sh

bash aws_drop/verify/verify_outputs.sh
bash aws_drop/sync/s3_push_outputs.sh
```

## Configuration
You can override parameters using environment variables or edit/copy:
- `aws_drop/RUN_CONFIG.template.json`

Supported env vars:
- global: `RUN_CONFIG`, `RUN_ID`, `OUTPUT_ROOT`, `REQUIRE_CSV_SHA`, `REQUIRE_MIN_SPARC_POINTS`
- Branch C: `SEED`, `N_PERM`, `N_BOOT_BIN`, `N_BINS`
- Branch D: `SEED`, `N_SHUFFLES`
- TNG↔SPARC: `SEED`, `MODE`, `N_BOOTSTRAP`, `N_SHUFFLES`, `N_BINS`, `K`, `TNG_INPUT_PATH`

## S3 Layout Convention
Inputs:
- `s3://$S3_BUCKET/$S3_PREFIX/inputs/rar_points_unified.csv`
- `s3://$S3_BUCKET/$S3_PREFIX/inputs/rar_points.parquet` (optional TNG points)

Outputs:
- `s3://$S3_BUCKET/$S3_PREFIX/runs/<RUN_ID>/...`

Bundle uploads:
- `s3://$S3_BUCKET/$S3_PREFIX/bundles/`

## Determinism + Provenance
- Fixed seeds are passed through run scripts.
- Each run writes to a timestamped output directory.
- Run wrappers write `aws_run_stamp_*.json` with git HEAD, dataset hashes, and command.
