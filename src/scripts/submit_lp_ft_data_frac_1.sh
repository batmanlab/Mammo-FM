#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAMMO_FM_REPO_ROOT="${MAMMO_FM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MAMMO_FM_EVAL_DIR="${MAMMO_FM_EVAL_DIR:-/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check/downstream_eval/data_frac_1.0}"
MAMMO_FM_CLIP_CKPT="${MAMMO_FM_CLIP_CKPT:-/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check/Mammo-FM_BatmanlabTrained_CLIP.tar}"
MAMMO_FM_CONDA_ENV="${MAMMO_FM_CONDA_ENV:-/restricted/projectnb/batmanlab/shawn24/Mammo-CLIP_modernBert}"
QSUB_BIN="${QSUB_BIN:-qsub}"
DATA_FRAC="1.0"
WEIGHTED_BCE="n"

[[ -f "${MAMMO_FM_CLIP_CKPT}" ]] || { echo "Missing checkpoint: ${MAMMO_FM_CLIP_CKPT}" >&2; exit 1; }
[[ -f "${SCRIPT_DIR}/qsub_templates/base_classifier.qsub" ]] || { echo "Missing qsub template" >&2; exit 1; }
mkdir -p "${MAMMO_FM_EVAL_DIR}/logs" "${MAMMO_FM_EVAL_DIR}/submission_manifests"

RSNA_ROOT="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
VINDR_PREFIX="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"

TASKS=(
  "RSNA|cancer|${RSNA_ROOT}|RSNA_Cancer_Detection/train_images_png|RSNA_Cancer_Detection/train_folds.csv"
  "VINDR|cancer|${RSNA_ROOT}|${VINDR_PREFIX}/images_png|${VINDR_PREFIX}/vindr_detection_v1_folds_abnormal.csv"
  "VINDR|mass|${RSNA_ROOT}|${VINDR_PREFIX}/images_png|${VINDR_PREFIX}/vindr_detection_v1_folds.csv"
  "VINDR|calc|${RSNA_ROOT}|${VINDR_PREFIX}/images_png|${VINDR_PREFIX}/vindr_detection_v1_folds.csv"
  "CMMD|cancer|/restricted/projectnb/batmanlab/shawn24/Additional_Breast_data/ChineseMammoDataset|cmmd_png|cmmd_png/merged_final_cmmd.csv"
  "CMMD|mass|/restricted/projectnb/batmanlab/shawn24/Additional_Breast_data/ChineseMammoDataset|cmmd_png|cmmd_png/merged_final_cmmd.csv"
  "CMMD|calc|/restricted/projectnb/batmanlab/shawn24/Additional_Breast_data/ChineseMammoDataset|cmmd_png|cmmd_png/merged_final_cmmd.csv"
  "NLBreast|cancer|/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data|NLBS_Data_png_v1|nlbs_image_labels_v2.csv"
  "LUMINA|cancer|/restricted/projectnb/batmanlab/shared/Data/LUMINA|LUMINA_PNG_Mammo_FM|lumina_metadata.csv"
)
ARCHS=("breast_clip_det_b5_period_n_lp" "breast_clip_det_b5_period_n_ft")

run_timestamp="$(date +'%Y%m%d_%H%M%S')"
manifest="${MAMMO_FM_EVAL_DIR}/submission_manifests/lp_ft_data_frac_1_${run_timestamp}.tsv"
printf 'job_id\tjob_name\tdataset\tlabel\tarch\tdata_frac\tlog\n' > "${manifest}"

export MAMMO_FM_REPO_ROOT MAMMO_FM_EVAL_DIR MAMMO_FM_CONDA_ENV
export DATA_FRAC WEIGHTED_BCE
export CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"

for task in "${TASKS[@]}"; do
  IFS='|' read -r DATASET_NAME LABEL DATA_DIR IMG_DIR CSV_FILE <<< "${task}"
  [[ -f "${DATA_DIR}/${CSV_FILE}" ]] || { echo "Missing CSV: ${DATA_DIR}/${CSV_FILE}" >&2; exit 1; }
  for ARCH in "${ARCHS[@]}"; do
    gpu_capability=8.0
    memory_per_core=8G
    if [[ "${ARCH}" == *_lp ]]; then
      gpu_capability=7.0
      memory_per_core=4G
    fi
    export ARCH DATASET_NAME DATA_DIR IMG_DIR CSV_FILE LABEL
    mode="${ARCH##*_}"
    job_name="mf_${DATASET_NAME,,}_${LABEL}_${mode}_df1_${run_timestamp}"
    log_dir="${MAMMO_FM_EVAL_DIR}/logs/${DATASET_NAME,,}/${LABEL}/${mode}"
    log_path="${log_dir}/${job_name}.qlog"
    mkdir -p "${log_dir}"
    submission="$(${QSUB_BIN} -j y -N "${job_name}" -o "${log_path}" \
      -l "gpu_c=${gpu_capability},mem_per_core=${memory_per_core}" \
      -v ARCH,DATASET_NAME,DATA_DIR,IMG_DIR,CSV_FILE,DATA_FRAC,CLIP_CKPT,LABEL,WEIGHTED_BCE,MAMMO_FM_REPO_ROOT,MAMMO_FM_EVAL_DIR,MAMMO_FM_CONDA_ENV \
      "${SCRIPT_DIR}/qsub_templates/base_classifier.qsub")"
    job_id="$(sed -nE 's/.*job ([0-9]+).*/\1/p' <<< "${submission}")"
    [[ -n "${job_id}" ]] || { echo "Could not parse qsub response: ${submission}" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${job_id}" "${job_name}" "${DATASET_NAME}" "${LABEL}" "${ARCH}" "${DATA_FRAC}" "${log_path}" >> "${manifest}"
    echo "${submission} (${DATASET_NAME}/${LABEL}/${mode})"
  done
done

echo "Submission manifest: ${manifest}"
