#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

check_latest() {
  local base="$1"
  local must1="$2"
  local must2="${3:-}"
  if [[ ! -d "$base" ]]; then
    echo "[verify_outputs] missing directory: $base"
    return 1
  fi
  local latest
  latest="$(ls -1dt "$base"/* 2>/dev/null | head -n1 || true)"
  if [[ -z "$latest" ]]; then
    echo "[verify_outputs] no run folders in: $base"
    return 1
  fi
  if [[ ! -e "$latest/$must1" ]]; then
    echo "[verify_outputs] missing $must1 in $latest"
    return 1
  fi
  if [[ -n "$must2" && ! -e "$latest/$must2" ]]; then
    echo "[verify_outputs] missing $must2 in $latest"
    return 1
  fi
  echo "[verify_outputs] ok: $latest"
}

ok=0
check_latest "$REPO_ROOT/outputs/paper3_high_density" "summary_C1C2C3.json" "report_C1C2C3.md" || ok=1
check_latest "$REPO_ROOT/outputs/branchD_rar_control" "summary_referee_required.json" "report_referee_required.md" || ok=1

shopt -s nullglob
matches=("$REPO_ROOT"/analysis/results/tng_sparc_feature_repro_*)
shopt -u nullglob
if (( ${#matches[@]} > 0 )); then
  latest_tng="$(ls -1dt "$REPO_ROOT"/analysis/results/tng_sparc_feature_repro_* | head -n1)"
  [[ -f "$latest_tng/summary.json" ]] || { echo "[verify_outputs] missing summary.json in $latest_tng"; ok=1; }
  [[ -f "$latest_tng/report.md" ]] || { echo "[verify_outputs] missing report.md in $latest_tng"; ok=1; }
  echo "[verify_outputs] ok: $latest_tng"
else
  echo "[verify_outputs] no tng_sparc_feature_repro_* outputs found (optional)."
fi

if [[ "$ok" -ne 0 ]]; then
  exit 1
fi

echo "[verify_outputs] all checks passed"
