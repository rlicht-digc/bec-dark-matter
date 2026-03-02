#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$REPO_ROOT/analysis/tng/tng_sparc_feature_repro.py" ]]; then
  echo "[run_tng_sparc] missing analysis/tng/tng_sparc_feature_repro.py" >&2
  exit 1
fi

RUN_CONFIG="${RUN_CONFIG:-$REPO_ROOT/aws_drop/RUN_CONFIG.template.json}"
TNG_INPUT_PATH="${TNG_INPUT_PATH:-/mnt/data/rar_points.parquet}"
MODE="${MODE:-full}"
SEED="${SEED:-42}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
N_SHUFFLES="${N_SHUFFLES:-1000}"
N_BINS="${N_BINS:-15}"
K="${K:-20}"

if [[ -f "$RUN_CONFIG" ]]; then
  eval "$(python3 - "$RUN_CONFIG" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
try:
  c=json.loads(p.read_text())
except Exception:
  c={}
t=c.get('tng_sparc',{})
mode=t.get('mode','full')
if isinstance(mode,str) and mode in {'smoke','full'}:
  print(f"MODE={mode}")
print(f"SEED={int(t.get('seed',42))}")
print(f"N_BOOTSTRAP={int(t.get('n_bootstrap',10000))}")
print(f"N_SHUFFLES={int(t.get('n_shuffles',1000))}")
print(f"N_BINS={int(t.get('n_bins',15))}")
print(f"K={int(t.get('K',20))}")
path=t.get('tng_input_path','')
if isinstance(path,str) and path:
  print(f"TNG_INPUT_PATH='{path}'")
PY
  )"
fi

if [[ ! -f "$TNG_INPUT_PATH" ]]; then
  echo "[run_tng_sparc] missing TNG input: $TNG_INPUT_PATH" >&2
  exit 1
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/analysis/results/tng_sparc_feature_repro_${MODE}_${TS}"

python3 analysis/tng/tng_sparc_feature_repro.py \
  --tng-input "$TNG_INPUT_PATH" \
  --mode "$MODE" \
  --seed "$SEED" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --n-shuffles "$N_SHUFFLES" \
  --n-bins "$N_BINS" \
  --K "$K" \
  --outdir "$OUT_DIR"

echo "[run_tng_sparc] output=$OUT_DIR"
