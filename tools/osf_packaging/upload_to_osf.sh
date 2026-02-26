#!/usr/bin/env bash
# upload_to_osf.sh — Upload staged files to OSF using osfclient
# Prerequisites: pip3 install osfclient
# Usage:   OSF_PROJECT=abc12 bash tools/osf_packaging/upload_to_osf.sh
# Dry-run: OSF_PROJECT=abc12 DRY_RUN=1 bash tools/osf_packaging/upload_to_osf.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGING="$REPO_ROOT/public_osf/staging"
DRY_RUN="${DRY_RUN:-}"

# --- Preflight checks ---
if [ -z "${OSF_PROJECT:-}" ]; then
  echo "ERROR: Set OSF_PROJECT to your OSF project ID (e.g. export OSF_PROJECT=abc12)" >&2
  exit 1
fi

if [ -z "${OSF_TOKEN:-}" ]; then
  # Try loading from macOS Keychain
  if command -v security &>/dev/null; then
    OSF_TOKEN="$(security find-generic-password -s "osf-token" -w 2>/dev/null || true)"
  fi
  if [ -z "$OSF_TOKEN" ]; then
    echo "ERROR: Set OSF_TOKEN or store it in macOS Keychain under 'osf-token'" >&2
    exit 1
  fi
  export OSF_TOKEN
fi

if ! command -v osf &>/dev/null; then
  echo "osfclient not found. Install it with:"
  echo "  pip3 install osfclient"
  exit 1
fi

if [ ! -d "$STAGING" ]; then
  echo "Staging directory not found at $STAGING"
  echo "Run 'bash tools/osf_packaging/stage_for_osf.sh' first."
  exit 1
fi

upload() {
  local src="$1"
  local dest="$2"
  if [ -n "$DRY_RUN" ]; then
    echo "  [dry-run] osf -p $OSF_PROJECT upload $src osfstorage/$dest"
  else
    osf -p "$OSF_PROJECT" upload "$src" "osfstorage/$dest"
  fi
}

UPLOADED=0

# --- Skip wiki (manual paste) ---
echo "=== OSF Upload ==="
echo "Project: $OSF_PROJECT"
echo ""
echo "SKIP: wiki/ — paste HTML into OSF wiki pages manually:"
echo "  - tests_results_osf.html → wiki page 'Tests & Results'"
echo "  - references_osf.html   → wiki page 'References'"
echo ""

# --- Core summaries ---
echo "Uploading core/ (State 1 summaries)..."
for f in "$STAGING"/core/*.json; do
  upload "$f" "core/$(basename "$f")"
  ((UPLOADED++))
done

# --- Supporting summaries ---
echo "Uploading supporting/ (State 2 summaries)..."
for f in "$STAGING"/supporting/*.json; do
  upload "$f" "supporting/$(basename "$f")"
  ((UPLOADED++))
done

# --- Metadata ---
echo "Uploading metadata/..."
for f in "$STAGING"/metadata/*; do
  upload "$f" "metadata/$(basename "$f")"
  ((UPLOADED++))
done

# --- Datasets ---
echo "Uploading datasets/..."
for ds_dir in "$STAGING"/datasets/*/; do
  ds_name="$(basename "$ds_dir")"
  for f in "$ds_dir"*; do
    [ -f "$f" ] || continue
    upload "$f" "datasets/$ds_name/$(basename "$f")"
    ((UPLOADED++))
  done
done

# --- Summary ---
echo ""
echo "Done. Uploaded $UPLOADED files to OSF project $OSF_PROJECT."
if [ -n "$DRY_RUN" ]; then
  echo "(Dry run — no files were actually uploaded.)"
fi
echo ""
echo "Reminder: manually paste wiki HTML files into OSF wiki pages."
