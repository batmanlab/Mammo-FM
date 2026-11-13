#!/usr/bin/env bash

# Mammo-FM risk prediction: Stage 1 distillation -> Stage 2 outcome training
# -> inference on PREDICT_SPLIT.
#
# Required:
#   CSV=/path/to/mammogram_cancer_5yr_with_risk.csv
#   MAMMO_CHECKPOINT=/path/to/MammoFM-checkpoint.tar
#   OUT=/path/to/output
#
# Common optional settings:
#   PYTHON_BIN=/path/to/conda/env/bin/python
#   NPROC=3
#   FOLDS="1 2 3 4 5"
#   SPLIT_COL='split_fold_{fold}'
#   TRAIN_SPLIT='train_fold_{fold}'
#   VAL_SPLIT='val_fold_{fold}'
#   PREDICT_SPLIT='test_fold_{fold}'

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export STAGE2_INIT=stage1

exec bash "$SCRIPT_DIR/mayo_mammofm_stage1_stage2.sh"
