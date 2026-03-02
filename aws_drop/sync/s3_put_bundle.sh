#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${S3_BUCKET:?Set S3_BUCKET before running}"
S3_PREFIX="${S3_PREFIX:-bec-dark-matter}"

BUNDLE_PATH="${1:-}"
if [[ -z "$BUNDLE_PATH" ]]; then
  BUNDLE_PATH="$(ls -1t "$REPO_ROOT"/dist/aws_drop_bundle_*.tar.gz 2>/dev/null | head -n1 || true)"
fi

if [[ -z "$BUNDLE_PATH" || ! -f "$BUNDLE_PATH" ]]; then
  echo "[s3_put_bundle] No bundle found. Pass bundle path explicitly." >&2
  exit 1
fi

DEST="s3://$S3_BUCKET/$S3_PREFIX/bundles/$(basename "$BUNDLE_PATH")"
aws s3 cp "$BUNDLE_PATH" "$DEST"
echo "[s3_put_bundle] uploaded: $DEST"
