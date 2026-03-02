#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

bash "$REPO_ROOT/aws_drop/verify/verify_inputs.sh"

(
  export N_PERM=200
  export N_BOOT_BIN=120
  export N_BINS=10
  export SEED=271828
  bash "$REPO_ROOT/aws_drop/run/run_branchC_full.sh"
)

(
  export N_SHUFFLES=80
  export SEED=42
  bash "$REPO_ROOT/aws_drop/run/run_branchD_full.sh"
)

if [[ -f "$REPO_ROOT/aws_drop/run/run_tng_sparc_feature_repro_full.sh" ]]; then
  if [[ -f "${TNG_INPUT_PATH:-/mnt/data/rar_points.parquet}" ]]; then
    (
      export MODE=smoke
      export N_BOOTSTRAP=200
      export N_SHUFFLES=80
      export N_BINS=10
      export K=12
      export SEED=42
      bash "$REPO_ROOT/aws_drop/run/run_tng_sparc_feature_repro_full.sh"
    )
  else
    echo "[run_smoke_all] skipping TNG smoke; missing ${TNG_INPUT_PATH:-/mnt/data/rar_points.parquet}"
  fi
fi

bash "$REPO_ROOT/aws_drop/verify/verify_outputs.sh"
echo "[run_smoke_all] done"
