#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${S3_BUCKET:?Set S3_BUCKET before running}"
S3_PREFIX="${S3_PREFIX:-bec-dark-matter}"

CSV_DST="$REPO_ROOT/analysis/results/rar_points_unified.csv"
TNG_DST="/mnt/data/rar_points.parquet"

mkdir -p "$(dirname "$CSV_DST")"
mkdir -p "$(dirname "$TNG_DST")"

aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/inputs/rar_points_unified.csv" "$CSV_DST"

if aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/inputs/rar_points.parquet" >/dev/null 2>&1; then
  aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/inputs/rar_points.parquet" "$TNG_DST"
else
  echo "[s3_pull_inputs] Optional TNG input not found; skipping rar_points.parquet"
fi

python3 - "$CSV_DST" "$TNG_DST" <<'PY'
import hashlib
import pathlib
import sys

def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

csv = pathlib.Path(sys.argv[1])
tng = pathlib.Path(sys.argv[2])
print(f"[s3_pull_inputs] {csv} sha256={sha(csv)}")
if tng.exists():
    print(f"[s3_pull_inputs] {tng} sha256={sha(tng)}")
PY
