#!/usr/bin/env bash
set -euo pipefail

GRG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GRG_ROOT"

deepspeed --master_port "${MASTER_PORT:-12434}" scripts/train.py --stage 1 "$@"
