#!/usr/bin/env bash
set -euo pipefail

GRG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GRG_ROOT"

python -m scripts.extract_image_embeddings \
  --checkpoint "${MAMMO_FM_CHECKPOINT:-Mammo-FM_BatmanlabTrained_CLIP_last.tar}" \
  "$@"
