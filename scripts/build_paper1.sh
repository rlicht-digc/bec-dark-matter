#!/usr/bin/env bash
# Build Paper 1 (main.tex -> main.pdf) using latexmk
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")/../paper" && pwd)"
cd "$PAPER_DIR"

echo "[build] Working directory: $PAPER_DIR"
echo "[build] Building main.tex ..."

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

echo "[build] Done. PDF at: $PAPER_DIR/main.pdf"
ls -lh main.pdf
