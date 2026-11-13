#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAMMO_FM_REPO_ROOT="${MAMMO_FM_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

ASSET_DIR="${MAMMO_FM_ASSET_DIR:-/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check}"
export MAMMO_FM_CLIP_CKPT="${MAMMO_FM_CLIP_CKPT:-${ASSET_DIR}/Mammo-FM_BatmanlabTrained_CLIP.tar}"
export MAMMO_FM_MODERNBERT_DIR="${MAMMO_FM_MODERNBERT_DIR:-${ASSET_DIR}/text_encoders/modernbert_mlm_5epochs}"
export MAMMO_FM_ZERO_SHOT_DIR="${MAMMO_FM_ZERO_SHOT_DIR:-${ASSET_DIR}/zero_shot/b5}"

export MAMMO_FM_CMMD_DATA_DIR="${MAMMO_FM_CMMD_DATA_DIR:-/restricted/projectnb/batmanlab/shawn24/Additional_Breast_data/ChineseMammoDataset}"
export MAMMO_FM_NL_BREAST_DATA_DIR="${MAMMO_FM_NL_BREAST_DATA_DIR:-/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data}"
export MAMMO_FM_LUMINA_DATA_DIR="${MAMMO_FM_LUMINA_DATA_DIR:-/restricted/projectnb/batmanlab/shared/Data/LUMINA}"
export MAMMO_FM_RSNA_DATA_DIR="${MAMMO_FM_RSNA_DATA_DIR:-/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

# flash-attn imports a CUDA 12 extension while ModernBERT is registered. The
# matching runtime is bundled in the active Python environment but is not
# automatically added to the dynamic-linker search path on interactive nodes.
PYTHON_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
CUDA_RUNTIME_LIB="${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib"
if [[ ! -f "${CUDA_RUNTIME_LIB}/libcudart.so.12" ]]; then
  echo "CUDA 12 runtime not found in the active Python environment: ${CUDA_RUNTIME_LIB}" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python -c 'import torch; from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM; print("ModernBERT import preflight passed")'

required_files=(
  "${MAMMO_FM_CLIP_CKPT}"
  "${MAMMO_FM_REPO_ROOT}/src/zs_prompts/zs_prompts.json"
  "${MAMMO_FM_CMMD_DATA_DIR}/cmmd_png/merged_final_cmmd.csv"
  "${MAMMO_FM_NL_BREAST_DATA_DIR}/nlbs_image_labels_v2.csv"
  "${MAMMO_FM_LUMINA_DATA_DIR}/lumina_metadata.csv"
  "${MAMMO_FM_RSNA_DATA_DIR}/RSNA_Cancer_Detection/train_folds.csv"
  "${MAMMO_FM_RSNA_DATA_DIR}/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv"
  "${MAMMO_FM_RSNA_DATA_DIR}/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds_abnormal.csv"
)

for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

if [[ ! -d "${MAMMO_FM_MODERNBERT_DIR}" ]]; then
  echo "ModernBERT checkpoint directory not found: ${MAMMO_FM_MODERNBERT_DIR}" >&2
  exit 1
fi

mkdir -p "${MAMMO_FM_ZERO_SHOT_DIR}"
cd "${MAMMO_FM_REPO_ROOT}"

python src/codebase/eval_zero_shot_clip.py --config-name zs_clip "$@"
