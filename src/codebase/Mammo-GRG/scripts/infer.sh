#!/usr/bin/env bash
set -euo pipefail

GRG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GRG_ROOT"

python scripts/infer.py "$@"
