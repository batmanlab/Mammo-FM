#!/bin/bash -l

#$ -N mammofm_b5_modernbert_2048
#$ -P batmanlab
#$ -l h_rt=48:00:00
#$ -pe omp 8
#$ -l gpus=2
#$ -l gpu_c=8.0
#$ -j y
#$ -m bea
#$ -cwd

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MAMMO_FM_REPO_ROOT:-${SGE_O_WORKDIR:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}}"
NPROC_PER_NODE="${MAMMO_FM_NPROC_PER_NODE:-2}"

if [[ ! -f "${REPO_ROOT}/src/codebase/train.py" ]]; then
  echo "Mammo-FM repository root is invalid: ${REPO_ROOT}" >&2
  echo "Submit from the repository root or set MAMMO_FM_REPO_ROOT." >&2
  exit 1
fi

required_variables=(
  MAMMO_FM_MODERNBERT_DIR
  MAMMO_FM_IMAGE_TEXT_1_IMAGE_DIR
  MAMMO_FM_IMAGE_TEXT_1_CSV
  MAMMO_FM_IMAGE_LABEL_CSV
  MAMMO_FM_IMAGE_TEXT_2_IMAGE_DIR
  MAMMO_FM_IMAGE_TEXT_2_CSV
  MAMMO_FM_OUTPUT_DIR
)

for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Required environment variable is not set: ${variable_name}" >&2
    exit 2
  fi
done

for checkpoint_file in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json; do
  if [[ ! -f "${MAMMO_FM_MODERNBERT_DIR}/${checkpoint_file}" ]]; then
    echo "Missing ModernBERT file: ${MAMMO_FM_MODERNBERT_DIR}/${checkpoint_file}" >&2
    exit 3
  fi
done

for csv_file in \
  "${MAMMO_FM_IMAGE_TEXT_1_CSV}" \
  "${MAMMO_FM_IMAGE_LABEL_CSV}" \
  "${MAMMO_FM_IMAGE_TEXT_2_CSV}"; do
  if [[ ! -f "${csv_file}" ]]; then
    echo "Missing training CSV: ${csv_file}" >&2
    exit 4
  fi
done

for image_directory in \
  "${MAMMO_FM_IMAGE_TEXT_1_IMAGE_DIR}" \
  "${MAMMO_FM_IMAGE_TEXT_2_IMAGE_DIR}"; do
  if [[ ! -d "${image_directory}" ]]; then
    echo "Missing training image directory: ${image_directory}" >&2
    exit 5
  fi
done

if [[ "${NPROC_PER_NODE}" -ne 2 ]]; then
  echo "This launcher requests two GPUs; MAMMO_FM_NPROC_PER_NODE must therefore be 2." >&2
  exit 6
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is unavailable; activate the Mammo-FM environment before launching." >&2
  exit 7
fi

mkdir -p "${MAMMO_FM_OUTPUT_DIR}"
cd "${REPO_ROOT}"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${REPO_ROOT}/src/codebase/train.py" \
  --config-name pretrain_b5_det_modernbert_2048.yaml \
  "$@"
