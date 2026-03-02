#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$REPO_ROOT/analysis/pipeline/run_referee_required_tests.py" ]]; then
  echo "[run_branchD_full] missing analysis/pipeline/run_referee_required_tests.py" >&2
  exit 1
fi

RUN_CONFIG="${RUN_CONFIG:-$REPO_ROOT/aws_drop/RUN_CONFIG.template.json}"
CSV_PATH="$REPO_ROOT/analysis/results/rar_points_unified.csv"

SEED="${SEED:-42}"
N_SHUFFLES="${N_SHUFFLES:-1000}"
REQUIRE_CSV_SHA="${REQUIRE_CSV_SHA:-}"

if [[ -f "$RUN_CONFIG" ]]; then
  eval "$(python3 - "$RUN_CONFIG" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
try:
  c=json.loads(p.read_text())
except Exception:
  c={}
b=c.get('branchD',{})
print(f"SEED={int(b.get('seed',42))}")
print(f"N_SHUFFLES={int(b.get('n_shuffles',1000))}")
rcs=c.get('require_csv_sha','')
if isinstance(rcs,str) and rcs and '<' not in rcs and 'TODO' not in rcs:
  print(f"REQUIRE_CSV_SHA='{rcs}'")
PY
  )"
fi

python3 - "$CSV_PATH" "$REQUIRE_CSV_SHA" <<'PY'
import hashlib, pathlib, sys
csv=pathlib.Path(sys.argv[1])
req=sys.argv[2].strip().lower()
h=hashlib.sha256()
with csv.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):
    h.update(c)
got=h.hexdigest().lower()
print(f"[run_branchD_full] csv_sha256={got}")
if req and req != got:
  raise SystemExit(f"REQUIRE_CSV_SHA mismatch expected={req} got={got}")
PY

TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/outputs/branchD_rar_control/$TS"
mkdir -p "$OUT_DIR"

python3 - "$CSV_PATH" "$OUT_DIR/aws_run_stamp_branchD.json" <<'PY'
import datetime as dt, hashlib, json, pathlib, subprocess, sys
csv=pathlib.Path(sys.argv[1])
out=pathlib.Path(sys.argv[2])
def sha(p):
  h=hashlib.sha256()
  with p.open('rb') as f:
    for c in iter(lambda:f.read(1024*1024),b''):
      h.update(c)
  return h.hexdigest()
try:
  git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
except Exception:
  git='UNKNOWN'
obj={
  'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z'),
  'git_head': git,
  'dataset': str(csv),
  'dataset_sha256': sha(csv),
  'runner': 'aws_drop/run/run_branchD_full.sh'
}
out.write_text(json.dumps(obj,indent=2)+'\n')
PY

python3 analysis/pipeline/run_referee_required_tests.py \
  --project-root "$REPO_ROOT" \
  --output-dir "$OUT_DIR" \
  --seed "$SEED" \
  --n-shuffles "$N_SHUFFLES"

echo "[run_branchD_full] output=$OUT_DIR"
