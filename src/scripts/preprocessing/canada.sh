#!/bin/bash -l

#$ -P batmanlab
#$ -j y
#$ -N preprocess_NLBS
#$ -o preprocess.qlog

#$ -l h_rt=48:00:00
#$ -pe omp 8

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAMMO_FM_REPO_ROOT="${MAMMO_FM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
MAMMO_FM_CONDA_ENV="${MAMMO_FM_CONDA_ENV:-Mammo-FM}"
NL_BREAST_METADATA_CSV="${NL_BREAST_METADATA_CSV:-/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data/nlbs_image_labels_v2.csv}"

module load miniconda
conda activate "${MAMMO_FM_CONDA_ENV}"
python "${MAMMO_FM_REPO_ROOT}/src/preprocessing/preprocess_canada.py" \
  --metadata_csv "${NL_BREAST_METADATA_CSV}"
