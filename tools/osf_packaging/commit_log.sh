#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ALLOW_DIRTY=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "Usage:"
  echo "  tools/osf_packaging/commit_log.sh [--allow-dirty] <log_run.py args...>"
  exit 2
fi

if [[ "${ALLOW_DIRTY}" -eq 0 ]]; then
  BAD_PATHS=()
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    path="${line:3}"
    if [[ "${path}" != logs/runs/* ]]; then
      BAD_PATHS+=("${path}")
    fi
  done < <(git status --porcelain)

  if [[ ${#BAD_PATHS[@]} -gt 0 ]]; then
    echo "Refusing to proceed: uncommitted changes outside logs/runs were found."
    echo "First paths:"
    printf ' - %s\n' "${BAD_PATHS[@]:0:20}"
    echo "Use --allow-dirty to bypass this guard."
    exit 1
  fi
fi

LOG_PATH="$(python3 tools/osf_packaging/log_run.py "${ARGS[@]}")"

if [[ -z "${LOG_PATH}" || ! -f "${LOG_PATH}" ]]; then
  echo "Failed to create run log. log_run.py did not return a valid path."
  exit 1
fi

git add "${LOG_PATH}"

META="$(
python3 - "${LOG_PATH}" <<'PY'
import json
import sys

path = sys.argv[1]
doc = json.load(open(path))
run_id = doc.get("run_id", "unknown")
tool = doc.get("tool", "unknown")
purpose = str(doc.get("purpose", "unknown")).replace("\n", " ").strip()
print(f"{run_id}\t{tool}\t{purpose}")
PY
)"

IFS=$'\t' read -r RUN_ID TOOL PURPOSE <<< "${META}"
git commit -m "log: ${TOOL} ${PURPOSE} (${RUN_ID})"

echo "${LOG_PATH}"
