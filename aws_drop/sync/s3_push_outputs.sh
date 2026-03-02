#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${S3_BUCKET:?Set S3_BUCKET before running}"
S3_PREFIX="${S3_PREFIX:-bec-dark-matter}"
RUN_ID="${RUN_ID:-RUN_$(date -u +%Y%m%d_%H%M%S)}"
RUN_PATH="${1:-}"
BASE_URI="s3://$S3_BUCKET/$S3_PREFIX/runs/$RUN_ID"

if [[ -n "$RUN_PATH" ]]; then
  if [[ ! -d "$RUN_PATH" ]]; then
    echo "[s3_push_outputs] RUN_PATH not found: $RUN_PATH" >&2
    exit 1
  fi
  aws s3 sync "$RUN_PATH" "$BASE_URI/custom_run_$(basename "$RUN_PATH")/"
  echo "[s3_push_outputs] synced custom run path: $RUN_PATH"
  exit 0
fi

if [[ -d "$REPO_ROOT/outputs/branchD_rar_control" ]]; then
  aws s3 sync "$REPO_ROOT/outputs/branchD_rar_control" "$BASE_URI/branchD_rar_control/"
fi

if [[ -d "$REPO_ROOT/outputs/paper3_high_density" ]]; then
  aws s3 sync "$REPO_ROOT/outputs/paper3_high_density" "$BASE_URI/paper3_high_density/"
fi

shopt -s nullglob
for d in "$REPO_ROOT"/analysis/results/tng_sparc_feature_repro_*; do
  [[ -d "$d" ]] || continue
  aws s3 sync "$d" "$BASE_URI/analysis_results/$(basename "$d")/"
done
shopt -u nullglob

echo "[s3_push_outputs] synced to $BASE_URI"
