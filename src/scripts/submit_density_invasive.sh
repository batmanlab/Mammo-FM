#!/bin/bash
# Density and cancer-subtype LP/FT, plus corrected portrait-geometry CMMD reruns.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAMMO_FM_REPO_ROOT="${MAMMO_FM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
: "${MAMMO_FM_EVAL_DIR:?Set a NEW external output directory for these runs}"
: "${MAMMO_FM_CLIP_CKPT:?Set the path to Mammo-FM_BatmanlabTrained_CLIP.tar}"
: "${LUMINA_DENSITY_METADATA:?Run prepare_lumina_density.py and set its density CSV path}"
MAMMO_FM_CONDA_ENV="${MAMMO_FM_CONDA_ENV:-/restricted/projectnb/batmanlab/shawn24/Mammo-CLIP_modernBert}"
QSUB_BIN="${QSUB_BIN:-qsub}"
EPOCHS="${EPOCHS:-15}"
DATA_FRAC=1.0
WEIGHTED_BCE=n
CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"
[[ -f "${CLIP_CKPT}" && -f "${LUMINA_DENSITY_METADATA}" ]] || { echo "Missing backbone or LUMINA metadata" >&2; exit 1; }
case "$(realpath -m "${MAMMO_FM_EVAL_DIR}")/" in
  "${MAMMO_FM_REPO_ROOT}/"*) echo "Run artifacts must be outside the repository" >&2; exit 1;;
esac

RSNA_ROOT="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
VINDR_PREFIX="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
TASKS=(
  "RSNA|density|${RSNA_ROOT}|RSNA_Cancer_Detection/train_images_png|RSNA_Cancer_Detection/train_folds.csv"
  "VINDR|density|${RSNA_ROOT}|${VINDR_PREFIX}/images_png|${VINDR_PREFIX}/vindr_detection_v1_folds.csv"
  "LUMINA|density|/restricted/projectnb/batmanlab/shared/Data/LUMINA|LUMINA_PNG_Mammo_FM|lumina_metadata.csv"
  "RSNA|invasive|${RSNA_ROOT}|RSNA_Cancer_Detection/train_images_png|RSNA_Cancer_Detection/train_folds.csv"
)
if [[ "${RERUN_CMMD:-1}" == 1 ]]; then
  for label in cancer mass calc; do
    TASKS+=("CMMD|${label}|/restricted/projectnb/batmanlab/shawn24/Additional_Breast_data/ChineseMammoDataset|cmmd_png|cmmd_png/merged_final_cmmd.csv")
  done
fi
for task in "${TASKS[@]}"; do
  IFS='|' read -r DATASET_NAME LABEL DATA_DIR IMG_DIR CSV_FILE <<< "${task}"
  [[ -f "${DATA_DIR}/${CSV_FILE}" ]] || { echo "Missing source CSV: ${DATA_DIR}/${CSV_FILE}" >&2; exit 1; }
done

run_timestamp="$(date +'%Y%m%d_%H%M%S')"
mkdir -p "${MAMMO_FM_EVAL_DIR}/submission_manifests"
manifest="${MAMMO_FM_EVAL_DIR}/submission_manifests/density_invasive_portrait_${run_timestamp}.tsv"
printf 'job_id\tjob_name\tdataset\tlabel\tarch\tepochs\tlog\n' > "${manifest}"
export MAMMO_FM_REPO_ROOT MAMMO_FM_EVAL_DIR MAMMO_FM_CONDA_ENV
export CLIP_CKPT DATA_FRAC WEIGHTED_BCE EPOCHS LUMINA_DENSITY_METADATA
for task in "${TASKS[@]}"; do
  IFS='|' read -r DATASET_NAME LABEL DATA_DIR IMG_DIR CSV_FILE <<< "${task}"
  export DATASET_NAME LABEL DATA_DIR IMG_DIR CSV_FILE
  for mode in lp ft; do
    ARCH="breast_clip_det_b5_period_n_${mode}"
    gpu_capability=8.0
    memory_per_core=8G
    if [[ "${mode}" == "lp" ]]; then
      gpu_capability=7.0
      memory_per_core=4G
    fi
    export ARCH
    job_name="mf_${DATASET_NAME,,}_${LABEL}_${mode}_portrait_${run_timestamp}"
    log_dir="${MAMMO_FM_EVAL_DIR}/logs/${DATASET_NAME,,}/${LABEL}/${mode}"
    log_path="${log_dir}/${job_name}.qlog"
    mkdir -p "${log_dir}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      printf 'DRY RUN: %s (gpu_c=%s, mem_per_core=%s, omp=4) -> %s\n' "${job_name}" "${gpu_capability}" "${memory_per_core}" "${log_path}"
      continue
    fi
    submission="$("${QSUB_BIN}" -terse -j y -N "${job_name}" -o "${log_path}" \
      -pe omp 4 -l "mem_per_core=${memory_per_core},gpu_c=${gpu_capability}" \
      -v ARCH,DATASET_NAME,DATA_DIR,IMG_DIR,CSV_FILE,DATA_FRAC,CLIP_CKPT,LABEL,WEIGHTED_BCE,EPOCHS,LUMINA_DENSITY_METADATA,MAMMO_FM_REPO_ROOT,MAMMO_FM_EVAL_DIR,MAMMO_FM_CONDA_ENV \
      "${SCRIPT_DIR}/qsub_templates/base_classifier.qsub")"
    [[ "${submission}" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub response: ${submission}" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${submission}" "${job_name}" "${DATASET_NAME}" "${LABEL}" "${ARCH}" "${EPOCHS}" "${log_path}" >> "${manifest}"
    echo "Submitted ${submission}: ${DATASET_NAME}/${LABEL}/${mode}"
  done
done
echo "Submission manifest: ${manifest}"
