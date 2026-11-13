#!/usr/bin/env bash

# Corrected Mammo-FM five-horizon risk pipeline.
#
# This launcher deliberately uses the collaborator's existing Mayo dataset
# implementation and split values. It does not create folds or replace image
# preprocessing, view selection, or incomplete-exam handling.
# Required environment variables:
#   CSV=/path/to/mammogram_cancer_5yr_with_risk.csv
#   MAMMO_CHECKPOINT=/path/to/MammoFM-checkpoint.tar
#   OUT=/path/to/output
#
# Existing single-split defaults: train/valid, with prediction on valid.
# If the CSV already contains a held-out value, for example "test", set
# PREDICT_SPLIT=test. Existing fold columns can be selected with templates:
#   FOLDS="1 2 3 4 5" SPLIT_COL='split_fold_{fold}' \
#   TRAIN_SPLIT='train_fold_{fold}' VAL_SPLIT='val_fold_{fold}' \
#   PREDICT_SPLIT='test_fold_{fold}' bash mayo_mammofm_stage1_stage2.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CODE_DIR="$REPO_ROOT/src/codebase"
MAMMO_FM_CODE_DIR=$(cd "$REPO_ROOT/.." && pwd)

: "${CSV:?Set CSV to the Mayo image-level risk manifest}"
: "${MAMMO_CHECKPOINT:?Set MAMMO_CHECKPOINT to the Mayo Mammo-FM checkpoint}"
: "${OUT:?Set OUT to the output directory}"

PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-1}
FOLDS=${FOLDS:-1}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-6}
NUM_WORKERS=${NUM_WORKERS:-4}
SPLIT_COL=${SPLIT_COL:-split}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-valid}
PREDICT_SPLIT=${PREDICT_SPLIT:-valid}
STAGE2_INIT=${STAGE2_INIT:-stage1}
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}

export PYTHONPATH="$CODE_DIR:$MAMMO_FM_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

OPTIONAL=()
if [[ "${NO_AMP:-n}" == "y" ]]; then
  OPTIONAL+=(--no-amp)
fi

for FOLD in $FOLDS; do
  FOLD_DIR="$OUT/fold$FOLD"
  STAGE1_DIR="$FOLD_DIR/stage1"
  if [[ "$STAGE2_INIT" == "stage1" ]]; then
    STAGE2_DIR="$FOLD_DIR/stage2_from_stage1"
  elif [[ "$STAGE2_INIT" == "mayo" ]]; then
    STAGE2_DIR="$FOLD_DIR/stage2_only"
  else
    echo "STAGE2_INIT must be stage1 or mayo" >&2
    exit 2
  fi

  COMMON=(
    --csv "$CSV"
    --mammo-checkpoint "$MAMMO_CHECKPOINT"
    --fold "$FOLD"
    --split-col "$SPLIT_COL"
    --train-split "$TRAIN_SPLIT"
    --val-split "$VAL_SPLIT"
    --predict-split "$PREDICT_SPLIT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-epochs "$WARMUP_EPOCHS"
    "${OPTIONAL[@]}"
  )

  if [[ "$STAGE2_INIT" == "stage1" ]]; then
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
      -m mayo_mammofm_risk.run \
      --mode train --stage stage1 --output-dir "$STAGE1_DIR" \
      "${COMMON[@]}"

    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
      -m mayo_mammofm_risk.run \
      --mode train --stage stage2 --stage2-init stage1 \
      --stage1-checkpoint "$STAGE1_DIR/best.pt" --output-dir "$STAGE2_DIR" \
      "${COMMON[@]}"
  else
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
      -m mayo_mammofm_risk.run \
      --mode train --stage stage2 --stage2-init mayo --output-dir "$STAGE2_DIR" \
      "${COMMON[@]}"
  fi

  "$PYTHON_BIN" -m mayo_mammofm_risk.run \
    --mode predict --stage stage2 \
    --stage2-init "$STAGE2_INIT" \
    --checkpoint "$STAGE2_DIR/best.pt" --output-dir "$STAGE2_DIR" \
    "${COMMON[@]}"
done
