#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$REPO_ROOT/analysis/paper3/paper3_branchC_experiments.py" ]]; then
  echo "[run_branchC_full] missing analysis/paper3/paper3_branchC_experiments.py" >&2
  exit 1
fi

RUN_CONFIG="${RUN_CONFIG:-$REPO_ROOT/aws_drop/RUN_CONFIG.template.json}"
CSV_PATH="$REPO_ROOT/analysis/results/rar_points_unified.csv"

SEED="${SEED:-271828}"
N_PERM="${N_PERM:-10000}"
N_BOOT_BIN="${N_BOOT_BIN:-800}"
N_BINS="${N_BINS:-15}"
REQUIRE_CSV_SHA="${REQUIRE_CSV_SHA:-}"
REQUIRE_MIN_SPARC_POINTS="${REQUIRE_MIN_SPARC_POINTS:-}"

if [[ -f "$RUN_CONFIG" ]]; then
  eval "$(python3 - "$RUN_CONFIG" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
try:
  c=json.loads(p.read_text())
except Exception:
  c={}
b=c.get('branchC',{})
val=lambda v,d: d if v is None or (isinstance(v,str) and v.startswith('RUN_')) else v
print(f"SEED={int(val(b.get('seed'),271828))}")
print(f"N_PERM={int(val(b.get('n_perm'),10000))}")
print(f"N_BOOT_BIN={int(val(b.get('n_boot_bin'),800))}")
print(f"N_BINS={int(val(b.get('n_bins'),15))}")
rcs=c.get('require_csv_sha','')
if isinstance(rcs,str) and rcs and '<' not in rcs and 'TODO' not in rcs:
  print(f"REQUIRE_CSV_SHA='{rcs}'")
rmp=c.get('require_min_sparc_points',None)
if isinstance(rmp,(int,float)):
  print(f"REQUIRE_MIN_SPARC_POINTS={int(rmp)}")
PY
  )"
fi

if [[ -n "$REQUIRE_MIN_SPARC_POINTS" ]]; then
  python3 - "$CSV_PATH" "$REQUIRE_MIN_SPARC_POINTS" <<'PY'
import pandas as pd,sys
csv=sys.argv[1]
min_n=int(sys.argv[2])
df=pd.read_csv(csv,usecols=['source'])
n=int((df['source'].astype(str)=='SPARC').sum())
if n < min_n:
    raise SystemExit(f"SPARC points below threshold: {n} < {min_n}")
print(f"[run_branchC_full] SPARC points: {n} (threshold {min_n})")
PY
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/outputs/paper3_high_density/BRANCHC_C1C2C3_${TS}"
mkdir -p "$OUT_DIR"

python3 - "$CSV_PATH" "$OUT_DIR/aws_run_stamp_branchC.json" <<'PY'
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
  'runner': 'aws_drop/run/run_branchC_full.sh'
}
out.write_text(json.dumps(obj,indent=2)+'\n')
PY

CMD=(
  python3 analysis/paper3/paper3_branchC_experiments.py
  --rar_points_file "$CSV_PATH"
  --out_dir "$OUT_DIR"
  --seed "$SEED"
  --n_perm "$N_PERM"
  --n_boot_bin "$N_BOOT_BIN"
  --n_bins "$N_BINS"
)
if [[ -n "$REQUIRE_CSV_SHA" ]]; then
  CMD+=(--require_csv_sha "$REQUIRE_CSV_SHA")
fi

"${CMD[@]}"

echo "[run_branchC_full] output=$OUT_DIR"
