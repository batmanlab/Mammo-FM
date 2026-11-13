#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAMMO_FM_REPO_ROOT="${MAMMO_FM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MAMMO_FM_CLIP_CKPT="${MAMMO_FM_CLIP_CKPT:-/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check/Mammo-FM_BatmanlabTrained_CLIP.tar}"
: "${MAMMO_FM_EVAL_DIR:?Set MAMMO_FM_EVAL_DIR to an output directory outside the repository.}"
MAMMO_FM_CONDA_ENV="${MAMMO_FM_CONDA_ENV:-/restricted/projectnb/batmanlab/shawn24/Mammo-CLIP_modernBert}"
export MAMMO_FM_REPO_ROOT MAMMO_FM_EVAL_DIR MAMMO_FM_CONDA_ENV
[[ -f "${MAMMO_FM_CLIP_CKPT}" ]] || { echo "Missing Mammo-FM checkpoint: ${MAMMO_FM_CLIP_CKPT}" >&2; exit 1; }

CKPT_KEY="${MAMMO_FM_CKPT_KEY:-Mammo-FM}"
CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"

### Define global log base path
LOG_BASE="${MAMMO_FM_EVAL_DIR}/logs"
SUB_DIR="epoch4/batmanlab_trained_weightedBCE_n"
source "${SCRIPT_DIR}/configs/vindr_configs.sh"

VINDR_LABELS=("Mass" "Suspicious Calcification")
VINDR_DATA_FRACS=("1.0" "0.5" "0.25")
#VINDR_DATA_FRACS=("1.0")

###########################################
## VINDr DETECTION
###########################################

VINDR_DETECTION_ARCHS=("breast_clip_det_b5")
FREEZE_OPTIONS=("n" "y")

for LABEL in "${VINDR_LABELS[@]}"; do
  for FREEZE in "${FREEZE_OPTIONS[@]}"; do
    if [[ "$FREEZE" == "y" ]]; then
      FRACS=("1.0")
    else
      FRACS=("1.0" "0.5" "0.1")
    fi

    for FRAC in "${FRACS[@]}"; do
      for ARCH in "${VINDR_DETECTION_ARCHS[@]}"; do
        export CLIP_CKPT="$CLIP_CKPT"
        export ARCH="$ARCH"
        export DATA_FRAC="$FRAC"
        export FREEZE_BACKBONE="$FREEZE"
        export DATASET_NAME="$DATASET_NAME"
        export DATA_DIR="$DATA_DIR"
        export IMG_DIR="$IMG_DIR"
        export CSV_FILE="$CSV_FILE"
        export LABEL="$LABEL"

        echo $ARCH
        echo $DATASET_NAME
        echo $DATA_DIR
        echo $IMG_DIR
        echo $CSV_FILE
        echo $DATA_FRAC
        echo $LABEL
        echo $CLIP_CKPT

        timestamp=$(date +"%Y-%m-%d-%H-%M-%S-%N")
        label_tag="${LABEL// /_}"  # Replace spaces for safe jobname/log path
        jobname="${ARCH//[^a-zA-Z0-9]/_}_det_${label_tag}_${FREEZE}_${FRAC}_${CKPT_KEY}_${timestamp}"
        logdir="${LOG_BASE}/detection_vindr/${CKPT_KEY}/${SUB_DIR}"
        mkdir -p "$logdir"

        echo "Submitting VinDr DETECTION: $jobname"
        qsub -j y -N "$jobname" \
             -o "${logdir}/${jobname}.qlog" \
             -v ARCH,CLIP_CKPT,DATASET_NAME,DATA_DIR,IMG_DIR,CSV_FILE,DATA_FRAC,FREEZE_BACKBONE,LABEL,MAMMO_FM_REPO_ROOT,MAMMO_FM_EVAL_DIR,MAMMO_FM_CONDA_ENV \
             "${SCRIPT_DIR}/qsub_templates/base_detector.qsub"
      done
    done
  done
done
