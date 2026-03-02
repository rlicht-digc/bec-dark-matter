#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CSV_PATH="$REPO_ROOT/analysis/results/rar_points_unified.csv"
TNG_PATH="${TNG_INPUT_PATH:-/mnt/data/rar_points.parquet}"

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[verify_inputs] missing $CSV_PATH" >&2
  exit 1
fi

LINE_COUNT="$(wc -l < "$CSV_PATH")"
if [[ "$LINE_COUNT" -lt 100 ]]; then
  echo "[verify_inputs] CSV seems too small: $LINE_COUNT lines" >&2
  exit 1
fi

python3 - "$CSV_PATH" "$TNG_PATH" <<'PY'
import hashlib
import pathlib
import pandas as pd
import sys

csv = pathlib.Path(sys.argv[1])
tng = pathlib.Path(sys.argv[2])

def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

print(f"[verify_inputs] csv_sha256={sha(csv)}")
df = pd.read_csv(csv, usecols=['source'])
if 'SPARC' not in set(df['source'].astype(str).unique()):
    raise SystemExit('[verify_inputs] SPARC source rows not found')
print(f"[verify_inputs] sparc_points={int((df['source'].astype(str) == 'SPARC').sum())}")

req = pathlib.os.environ.get('REQUIRE_CSV_SHA', '').strip().lower()
if req:
    got = sha(csv).lower()
    if req != got:
        raise SystemExit(f"[verify_inputs] REQUIRE_CSV_SHA mismatch expected={req} got={got}")

if tng.exists():
    print(f"[verify_inputs] tng_sha256={sha(tng)} path={tng}")
else:
    print(f"[verify_inputs] optional TNG input missing at {tng}")
PY
