#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAMMO_FM_REPO_ROOT="${MAMMO_FM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MAMMO_FM_CLIP_CKPT="${MAMMO_FM_CLIP_CKPT:-/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check/Mammo-FM_BatmanlabTrained_CLIP.tar}"
: "${MAMMO_FM_EVAL_DIR:?Set MAMMO_FM_EVAL_DIR to an output directory outside the repository.}"
MAMMO_FM_CONDA_ENV="${MAMMO_FM_CONDA_ENV:-Mammo-FM}"
export MAMMO_FM_REPO_ROOT MAMMO_FM_EVAL_DIR MAMMO_FM_CONDA_ENV
[[ -f "${MAMMO_FM_CLIP_CKPT}" ]] || { echo "Missing Mammo-FM checkpoint: ${MAMMO_FM_CLIP_CKPT}" >&2; exit 1; }

CKPT_KEY="${MAMMO_FM_CKPT_KEY:-Mammo-FM}"
CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"
LOG_BASE="${MAMMO_FM_EVAL_DIR}/logs"
SUB_DIR="epoch15/batmanlab_trained_weightedBCE_n"

source "${SCRIPT_DIR}/configs/rsna_configs.sh"

RSNA_LABELS=("cancer")
RSNA_DATA_FRACS=("1.0" "0.8" "0.5" "0.25" "0.1")
RSNA_ARCHS=("breast_clip_det_b5_period_n_lp" "breast_clip_det_b5_period_n_ft")

for LABEL in "${RSNA_LABELS[@]}"; do
  for ARCH in "${RSNA_ARCHS[@]}"; do
    gpu_capability=8.0
    memory_per_core=8G
    if [[ "${ARCH}" == *_lp ]]; then
      gpu_capability=7.0
      memory_per_core=4G
    fi
    for FRAC in "${RSNA_DATA_FRACS[@]}"; do
      export CLIP_CKPT ARCH
      export DATA_FRAC="${FRAC}"
      export DATASET_NAME DATA_DIR IMG_DIR CSV_FILE LABEL WEIGHTED_BCE
      export EXTRA_ARGS="--label '${LABEL}' --n_folds 1"

      timestamp=$(date +"%Y-%m-%d-%H-%M-%S-%N")
      jobname="${ARCH//[^a-zA-Z0-9]/_}_${DATASET_NAME}_${LABEL}_${FRAC}_${CKPT_KEY}_${timestamp}"
      logdir="${LOG_BASE}/classification_${DATASET_NAME,,}/${CKPT_KEY}/${SUB_DIR}"
      mkdir -p "${logdir}"

      echo "Submitting RSNA: ${jobname}"
      qsub -j y -N "${jobname}" \
        -l "gpu_c=${gpu_capability},mem_per_core=${memory_per_core}" \
        -o "${logdir}/${jobname}.qlog" \
        -v ARCH,DATASET_NAME,DATA_DIR,IMG_DIR,CSV_FILE,DATA_FRAC,CLIP_CKPT,LABEL,WEIGHTED_BCE,EXTRA_ARGS,MAMMO_FM_REPO_ROOT,MAMMO_FM_EVAL_DIR,MAMMO_FM_CONDA_ENV \
        "${SCRIPT_DIR}/qsub_templates/base_classifier.qsub"
    done
  done
done
