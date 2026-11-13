#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MAMMO_FM_CODE_DIR=$(cd "$ROOT/.." && pwd)
export PYTHONPATH="$ROOT/src/codebase:$MAMMO_FM_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "${SAE_PYTHON_BIN:-${PYTHON_BIN:-python}}" -m mammo_sae "$@"
