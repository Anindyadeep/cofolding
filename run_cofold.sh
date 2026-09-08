#!/usr/bin/env bash
# Bootstrap: create a tiny orchestrator venv (huggingface_hub + hf_transfer),
# then hand off to run_cofold.py. Safe to call repeatedly (venv is reused).
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV="${COFOLD_VENV:-$SCRIPT_DIR/.venv-orchestrator}"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "[bootstrap] creating orchestrator venv at $VENV"
    python3 -m venv "$VENV"
    "$PY" -m pip install -q -U pip
    "$PY" -m pip install -q "huggingface_hub>=0.24" hf_transfer
fi

exec "$PY" "$SCRIPT_DIR/run_cofold.py" "$@"
