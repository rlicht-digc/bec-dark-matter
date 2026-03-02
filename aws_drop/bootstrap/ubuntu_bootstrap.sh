#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_PATH="$REPO_ROOT/aws_drop/bootstrap/bootstrap_env.log"

sudo apt-get update -y
sudo apt-get install -y git tmux htop python3 python3-venv python3-pip awscli

cd "$REPO_ROOT"
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel
if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
  python3 -m pip install -r "$REPO_ROOT/requirements.txt"
else
  python3 -m pip install numpy scipy pandas pyarrow matplotlib joblib scikit-learn pytest
fi

echo "[bootstrap] python: $(python3 --version)" | tee "$LOG_PATH"
echo "[bootstrap] pip: $(python3 -m pip --version)" | tee -a "$LOG_PATH"
python3 -m pip freeze | tee "$REPO_ROOT/aws_drop/pip_freeze.txt" | tee -a "$LOG_PATH" >/dev/null

echo "[bootstrap] complete. activate with: source .venv/bin/activate"
