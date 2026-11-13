#!/bin/bash
# Submit three independent pooled-data full-finetuning jobs for Stage-3 grounding.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAMMO_FM_REPO_ROOT="${MAMMO_FM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
: "${MAMMO_FM_EVAL_DIR:?Set a new external run directory containing metadata/audit.json}"
: "${MAMMO_FM_CLIP_CKPT:?Set Mammo-FM_BatmanlabTrained_CLIP.tar}"
: "${MAMMO_FM_CONDA_ENV:?Set the existing classification conda environment path}"
QSUB_BIN="${QSUB_BIN:-qsub}"
EPOCHS="${EPOCHS:-15}"
CLIP_CKPT="$(realpath -- "${MAMMO_FM_CLIP_CKPT}")"
DATA_DIR="$(realpath -- "${MAMMO_FM_EVAL_DIR}/metadata")"
MAMMO_FM_EVAL_DIR="$(realpath -- "${MAMMO_FM_EVAL_DIR}")"
[[ -f "${CLIP_CKPT}" && -x "${MAMMO_FM_CONDA_ENV}/bin/python" ]] || {
  echo "Missing backbone or classification Python environment" >&2; exit 1;
}
case "${MAMMO_FM_EVAL_DIR}/" in
  "$(realpath -- "${MAMMO_FM_REPO_ROOT}")/"*) echo "Run artifacts must be outside the repository" >&2; exit 1;;
esac

"${MAMMO_FM_CONDA_ENV}/bin/python" -E - "${DATA_DIR}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
audit = json.loads((root / "audit.json").read_text())
assert audit["status"] == "PASS", "Grounding data audit did not pass"
assert set(audit["source_audits"]) == {"vindr", "embed", "bu", "upmc"}, "Incorrect source cohorts"
checks = audit["validation"]
assert checks["checked_unique_png_paths"] > 0
for name in ("missing_png_paths", "patient_split_overlaps", "exam_split_overlaps", "image_split_overlaps"):
    assert checks[name] == 0, f"Failed data audit: {name}"
for task in ("mass", "calcification", "asymmetry"):
    entry = audit["tasks"][task]
    path = root / f"grounding_{task}.csv"
    assert entry["filename"] == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["output_sha256"], f"Changed manifest: {task}"
    assert set(entry["by_source_and_split"]) == {"vindr", "embed", "bu", "upmc"}
print("All three audited grounding manifests verified; CMMD excluded.")
PY

DATASET_NAME=Grounding
ARCH=breast_clip_det_b5_period_n_ft
DATA_FRAC=1.0
WEIGHTED_BCE=n
IMG_DIR=.
export MAMMO_FM_REPO_ROOT MAMMO_FM_EVAL_DIR MAMMO_FM_CONDA_ENV
export CLIP_CKPT DATA_DIR DATASET_NAME ARCH DATA_FRAC WEIGHTED_BCE IMG_DIR EPOCHS
manifest="${MAMMO_FM_EVAL_DIR}/submissions.tsv"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  for artifact_dir in checkpoints outputs tensorboard; do
    [[ ! -e "${MAMMO_FM_EVAL_DIR}/${artifact_dir}" ]] || {
      echo "Existing ${artifact_dir} directory; select a fresh run directory" >&2; exit 1;
    }
  done
  # An interrupted or repeated launch cannot silently submit duplicate jobs.
  (set -o noclobber; printf 'job_id\ttask\tepochs\tlog\n' > "${manifest}") || {
    echo "Submission manifest already exists; inspect it before launching again" >&2; exit 1;
  }
fi
for LABEL in mass calcification asymmetry; do
  CSV_FILE="grounding_${LABEL}.csv"
  export LABEL CSV_FILE
  job_name="mf_grounding_${LABEL}_ft"
  log_dir="${MAMMO_FM_EVAL_DIR}/logs/${LABEL}"
  log_path="${log_dir}/${job_name}.qlog"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY RUN: %s | epochs=%s | 4 CPU, 32G host RAM, 1 GPU, gpu_c=8.0, 24h | %s\n' "${job_name}" "${EPOCHS}" "${log_path}"
    continue
  fi
  mkdir -p "${log_dir}"
  submission="$("${QSUB_BIN}" -terse -j y -N "${job_name}" -o "${log_path}" \
    -P batmanlab -pe omp 4 -l h_rt=24:00:00,mem_per_core=8G,gpus=1,gpu_c=8.0 \
    -v ARCH,DATASET_NAME,DATA_DIR,IMG_DIR,CSV_FILE,DATA_FRAC,CLIP_CKPT,LABEL,WEIGHTED_BCE,EPOCHS,MAMMO_FM_REPO_ROOT,MAMMO_FM_EVAL_DIR,MAMMO_FM_CONDA_ENV \
    "${SCRIPT_DIR}/qsub_templates/base_classifier.qsub")"
  [[ "${submission}" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub response: ${submission}" >&2; exit 1; }
  printf '%s\t%s\t%s\t%s\n' "${submission}" "${LABEL}" "${EPOCHS}" "${log_path}" >> "${manifest}"
  echo "Submitted ${submission}: grounding/${LABEL}/full-finetune"
done
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  echo "Submission manifest: ${manifest}"
fi
