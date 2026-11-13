# Mammo-FM: Breast-specific foundational model for Integrated Mammographic Diagnosis, Prognosis, and Reporting

[![PyPI v0.4.1](https://img.shields.io/badge/PyPI-v0.4.1-blue.svg)](https://pypi.org/project/mammofm/0.4.1/)
[![PyPI downloads: pending indexing](https://img.shields.io/badge/downloads-pending%20indexing-lightgrey.svg)](https://pypistats.org/faqs)
[![Paper](https://img.shields.io/badge/Paper-9cf)](https://arxiv.org/pdf/2512.00198)
[![Hugging Face](https://img.shields.io/badge/Checkpoints-Hugging%20Face-yellow)](https://huggingface.co/batmanLab/Mammo-FM)
[🇻🇳 ![VinDr PNG](https://img.shields.io/badge/VinDr%20Mammogram%20png%20images-lightblue)](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)
[🇺🇸 ![RSNA PNG](https://img.shields.io/badge/RSNA%20Mammogram%20png%20images-lightblue)](https://huggingface.co/datasets/shawn24/RSNA-Breast-Cancer-detection-from-mammograms-PNG)
[🇨🇳 ![CMMD PNG](https://img.shields.io/badge/CMMD%20Mammography%20PNG-lightblue)](https://www.kaggle.com/datasets/shantanughosh/cmmd-mammography-png)
[🇨🇦 ![NL-Breast PNG](https://img.shields.io/badge/NL--Breast%20Screen%20PNG-lightblue)](https://www.kaggle.com/datasets/shantanughosh/nl-breast-screen-png)
[🇹🇷 ![LUMINA PNG](https://img.shields.io/badge/LUMINA%20Mammography%20PNG-lightblue)](https://www.kaggle.com/datasets/shantanughosh/lumina-mammography-png)
![Visitor Badge](https://visitor-badge.laobi.icu/badge?page_id=batmanlab.FM&right_color=%23FFA500)

#### ⚠️ WARNING: Before running the commands in this README, replace `/restricted/projectnb/batmanlab/shawn24/PhD` with the path to your own project/data root. This is a lab-specific example path and will not exist on other systems.

#### ⚠️ WARNING: There is a plethora of pre-processing settings available for RSNA and VinDr Mammo datasets. We recommend using the pre-processing discussed in the following sections. We are not responsible for any discrepancies in the results due to different pre-processing settings. If you use the png dataset uploaded in kaggle or HF by us, it is fully pre-processed. Else you can use the pre-processing scripts provided in the following sections.

#### ⚠️ WARNING: If you find the `punkt_tab` error, run the following command in the python environment:

```python
import nltk

nltk.download('punkt_tab')
```

## FAQ

We follow our previous code [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP) strictly. The Mammo-FM pretraining
entry point, downstream diagnostic evaluation workflows (linear probing and full fine-tuning), public-dataset
zero-shot evaluation, five-year risk prediction and interpretation, and Mammo-GRG report-generation component are
included in this repository.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Python package](#python-package)
   1. [Custom zero-shot prompts](#custom-zero-shot-prompts)
3. [CLI](#cli)
4. [Data Download](#data-download)
5. [Pre-processing Images](#pre-processing-images)
6. [Pretraining Mammo-FM](#pretraining-mammo-fm)
7. [Mammo-FM (image-text alignment) Contrastive Checkpoints](#mammo-fm-image-text-alignment-contrastive-checkpoints)
8. [Evaluation (Diagnosis)](#evaluation-diagnosis)
   1. [Zero-shot evaluation of Mammo-FM](#zero-shot-evaluation-of-mammo-fm)
   2. [Linear probe/Full finetune scripts](#linear-probefull-finetune-scripts)
      1. [Released downstream classification checkpoints](#released-downstream-classification-checkpoints)
9. [Risk Prediction (Prognosis)](#risk-prediction-prognosis)
10. [Radiology Report Generation (Mammo-GRG)](#radiology-report-generation-mammo-grg)
11. [Citation](#citation)
12. [License and Copyright](#license-and-copyright)
13. [Contact](#contact)
14. [Contributing](#contributing)
15. [Acknowledgement](#acknowledgement)

## Environment Setup

Use [environment.yml](environment.yml) to setup the environment.

```bash
git clone git@github.com:batmanlab/Mammo-FM.git
cd Mammo-FM
conda env create --name Mammo-FM -f environment.yml
conda activate Mammo-FM
```

Mammo-FM is implemented with following specification:

* Python version: 3.9+
* PyTorch version: 2.2.2
* CUDA version: 11.8

## Python package

Use Mammo-FM for preprocessing, diagnosis, prognosis and report generation easily in **7 lines of Python**
(excluding comments and blank lines).

Install with `pip install 'mammofm[report,dicom]'` (Python 3.10+, PyTorch 2.6+).
Diagnosis and embeddings use one image; prognosis and reports use four views from the same exam.
Inference accepts DICOMs or preprocessed PNGs; standalone preprocessing accepts DICOMs only.

```python
import mammofm as mf
views = {"lmlo": "LMLO.dcm", "lcc": "LCC.dcm", "rmlo": "RMLO.dcm", "rcc": "RCC.dcm"}

## Preprocess a DICOM image:
image = mf.preprocess(views["lmlo"])

## Extract image embeddings:
embeddings = mf.features(views["lmlo"], "vindr")

## Diagnosis:
diagnosis = mf.diagnosis(views["lmlo"], "vindr")

## Prognosis:
risk = mf.prognosis(views)

## Report generation:
report = mf.report_gen(views)
```

`mf.preprocess(img)` returns an in-memory, 8-bit grayscale Pillow image, **1520 pixels high × 912 wide**,
using the unified converter. It runs on CPU without model downloads; save the result with
`image.save("preprocessed.png")`.

See the [Python package guide](docs/python-package.md) for supported datasets, outputs,
GPU settings, model access and offline use. For command-line inference, see [CLI](#cli).

### Custom zero-shot prompts

Install or upgrade with `pip install --upgrade 'mammofm>=0.4.0'`, then use your own
positive and negative prompts:

```python
import mammofm as mf

Pos_label = ["There is a mass in the breast."]
Neg_label = ["There is no mass in the breast."]
positive = mf.text_embeddings(Pos_label)
negative = mf.text_embeddings(Neg_label)
image = mf.image_embeddings("mammogram.dcm", dataset="rsna")
output = mf.zero_shot(image, positive, negative)
print(output)
```

`text_embeddings` returns one projected, normalized embedding per prompt;
`image_embeddings` returns the matching image embedding for a DICOM or preprocessed PNG.
Both use the image and text weights in `Mammo-FM_BatmanlabTrained_CLIP.tar`.
The output contains `positive_cosine`, `negative_cosine`, and `score` (positive minus
negative cosine), each a list with one value per image. No sigmoid, softmax, or
probability conversion is applied. Multiple prompts are averaged within each class
after row normalization, then the class average is normalized again.

Use `image_embeddings`, not the unprojected arrays from `mf.features`, for text comparison.
The existing `mf.diagnosis` scoring is unchanged. See the
[custom-prompt guide](docs/python-package.md#custom-zero-shot-prompts) for reusable
clients, cached embeddings, and input details.

## CLI

Run the complete report-generation and risk-prediction workflow on one
four-view DICOM examination. Supply exactly one correctly labeled DICOM from
the same examination for each canonical view. This section assumes the
`mammofm` command is already available in the active environment.

```bash
EXAM_DIR="/path/to/one/exam"
OUTPUT_DIR="./outputs/exam-001"

# Run both risk models and Mammo-GRG report generation:
mammofm exam \
  --lmlo "${EXAM_DIR}/LMLO.dcm" \
  --lcc  "${EXAM_DIR}/LCC.dcm" \
  --rmlo "${EXAM_DIR}/RMLO.dcm" \
  --rcc  "${EXAM_DIR}/RCC.dcm" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda

# After every branch succeeds, expect:
# ./outputs/exam-001/report.json      Mammo-GRG reports
# ./outputs/exam-001/risk_mayo.json   Mayo-trained 1- to 5-year probabilities
# ./outputs/exam-001/risk_bu.json     BMC-trained 1- to 5-year probabilities
#
# The same three result objects are also printed together as JSON in the
# terminal; progress messages are printed separately.
```

The risk values are direct, uncalibrated model probabilities. The default BMC
risk model is the Stage 1 MIRAI KD → Stage 2 BU outcomes checkpoint. Use
`--only risk` to run only the two risk models, `--only report` to run only
Mammo-GRG report generation, and `--overwrite` to intentionally replace
existing requested JSON files. The `--device cuda` option requires an available
CUDA GPU; omit `--device` to let the CLI select CUDA when available and CPU
otherwise.

See the [CLI guide](docs/python-package.md#four-view-report-and-risk-cli) for options,
output files and setup. All outputs are for research use only.

## Data Download

Download the preprocessed PNG images (image size: **1520x912 pixels**) for the OOD evaluation datasets, suitable for direct use with the Mammo-FM image encoder, from:

- [🇻🇳 VinDr](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)
- [🇺🇸 RSNA](https://huggingface.co/datasets/shawn24/RSNA-Breast-Cancer-detection-from-mammograms-PNG)
- [🇨🇳 CMMD](https://www.kaggle.com/datasets/shantanughosh/cmmd-mammography-png)
- [🇨🇦 NL-Breast Screen](https://www.kaggle.com/datasets/shantanughosh/nl-breast-screen-png)
- [🇹🇷 LUMINA](https://www.kaggle.com/datasets/shantanughosh/lumina-mammography-png)

Download the corresponding evaluation CSVs from the [`OOD_CSV`](https://huggingface.co/batmanLab/Mammo-FM/tree/main/OOD_CSV) directory. Use the split column shown below.

<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th align="left">Dataset / task</th>
      <th align="left">Evaluation CSV</th>
      <th align="center">Split column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🇨🇳 CMMD classification</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/OOD_CSV/CMMD/merged_final_cmmd.csv"><code>merged_final_cmmd.csv</code></a></td>
      <td align="center"><code>fold</code></td>
    </tr>
    <tr>
      <td>🇨🇦 NL-Breast classification</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/OOD_CSV/NL-Breast/nlbs_image_labels_v2.csv"><code>nlbs_image_labels_v2.csv</code></a></td>
      <td align="center"><code>fold</code></td>
    </tr>
    <tr>
      <td>🇹🇷 LUMINA classification</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/OOD_CSV/LUMINA/lumina_metadata.csv"><code>lumina_metadata.csv</code></a></td>
      <td align="center"><code>fold</code></td>
    </tr>
    <tr>
      <td>🇺🇸 RSNA classification</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/OOD_CSV/RSNA/train_folds.csv"><code>train_folds.csv</code></a></td>
      <td align="center"><code>fold</code></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr finding classification and detection</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/OOD_CSV/VinDr/vindr_detection_v1_folds.csv"><code>vindr_detection_v1_folds.csv</code></a></td>
      <td align="center"><code>split</code></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr abnormal classification</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/OOD_CSV/VinDr/vindr_detection_v1_folds_abnormal.csv"><code>vindr_detection_v1_folds_abnormal.csv</code></a></td>
      <td align="center"><code>split</code></td>
    </tr>
  </tbody>
</table>

**Notes for 🇻🇳 VinDr:**
- `vindr_detection_v1_folds.csv` also contains a legacy `fold` column. Use `split` as the authoritative column.
- `vindr_detection_v1_folds_abnormal.csv` has no `fold` column. Do not use the legacy `split_new` column.

If you download these preprocessed PNG datasets, skip the conversion commands below.

## Pre-processing images

If you want to download the original DICOM images and preprocess them yourself, obtain them from the dataset's official source and use the unified converter below. The default output size is **1520x912** pixels (height x width).

### Unified DICOM-to-PNG preprocessing

[`src/preprocessing/preprocess_images.py`](src/preprocessing/preprocess_images.py) works with any directory of single-frame grayscale DICOM images. It has no dataset names, fixed filesystem paths, label columns, or fold requirements. It runs on CPU using NumPy, OpenCV, and pydicom; CUDA, DALI, PyTorch, and pandas are not required.

For one image in Python, use [`mf.preprocess(img)`](docs/python-package.md#standalone-dicom-preprocessing).
It uses the same default pixel-processing pipeline and returns an image without writing a file.

```bash
python src/preprocessing/preprocess_images.py \
  --input-root /path/to/dicom_images \
  --output-root /path/to/preprocessed_png \
  --height 1520 --width 912 \
  --workers 4
```

The script recursively finds `.dcm` and `.dicom` files, preserving relative directories when writing PNGs. For other filename extensions, supply `--pattern`, for example `--pattern '*'` for extensionless DICOM files in a dedicated image directory. To select files from a CSV instead, add `--csv /path/to/images.csv --path-column image_path`; the column name is configurable, and paths may be absolute or relative to `--input-root` but must remain within that root. Duplicate CSV paths are processed once.

The default pipeline is: remove a five-pixel border, correct `MONOCHROME1`, apply min-max intensity scaling, convert to uint8, apply the existing NumPy breast-extraction heuristic, and resize with OpenCV `INTER_AREA`. Use `--border 0` or `--no-breast-crop` to disable those respective steps. This preserves the legacy CPU convention, not the old GPU implementation's rounding or decoding behavior.

Intensity processing is explicit rather than inferred from a dataset name:

- `--intensity minmax` (default): raw-pixel min-max scaling, as in the original RSNA/VinDr CPU converters.
- `--intensity window --fallback percentile`: the original Canada converter's linear window-center/window-width mapping after rescale slope/intercept, with 1st/99th-percentile fallback if window metadata is absent. The fallback can be changed to `minmax`, and the quantiles with `--p-lo`/`--p-hi`. This legacy mapping is not a complete implementation of DICOM VOI LUT/window-function semantics.

Compressed DICOMs require an appropriate pydicom decoding plugin for their transfer syntax; JPEG 2000 may require compatible `pylibjpeg` and `pylibjpeg-openjpeg` installations. Decode failures, missing files, and unsupported color/multiframe inputs are recorded in a timestamped CSV under the output directory and cause a nonzero exit status. Use `--report` to choose a different new report file. Existing PNGs are not overwritten by default: `--existing overwrite` explicitly replaces them, while `--existing skip` leaves them unchanged without revalidating their preprocessing settings.

This converter does not generate labels, change train/validation/test folds, transform detection bounding boxes, or anonymize images and filenames. Keep the local processing reports private. The dataset-specific converters below remain available for reference; matching historical evaluation requires choosing the same intensity settings and input pipeline.

### Convert to PNG: RSNA

```bash
python ./src/preprocessing/preprocess_image_to_png_kaggle.py \
  --phase="train" \
  --height=1520 \
  --width=912 \
  --base_folder="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/RSNA_Cancer_Detection"
```

### Convert to PNG: VinDr

```bash
python ./src/preprocessing/preprocess_image_to_png_vindr.py \
  --height=1520 \
  --width=912 \
  --base_folder="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset/External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
```

The legacy RSNA converter writes to `<base_folder>/mammo_clip/train_images_png`, and the legacy VinDr converter writes to `<base_folder>/mammo_clip/images_png`. The unified converter writes to your explicit `--output-root`. Point `IMG_DIR` to the generated PNG directory when running an evaluation. Alternatively, use the preprocessed dataset downloads above.

## Pretraining Mammo-FM

The pretraining workflow uses the EfficientNet-B5 detector image encoder and a ModernBERT text encoder with a maximum
sequence length of 2,048 tokens. The configuration combines two image-text datasets and one image-label dataset. These
neutral dataset labels describe the type of supervision expected by the loader and do not identify an institution.

> **Text-encoder initialization:** Before starting pretraining, download the
> [ModernBERT checkpoint](https://huggingface.co/batmanLab/Mammo-FM/tree/main/text_encoders/modernbert_mlm_5epochs)
> used to initialize the Mammo-FM text encoder. This domain-adapted checkpoint was fine-tuned on in-house radiology
> reports. The reports themselves are not distributed with the checkpoint.

Keep checkpoints and other run artifacts outside the Mammo-FM source repository. For example, download the text
encoder into `Mammo-FM-Test-check`:

```bash
MAMMO_FM_ASSET_DIR="/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check"

huggingface-cli download batmanLab/Mammo-FM \
  --include "text_encoders/modernbert_mlm_5epochs/*" \
  --local-dir "${MAMMO_FM_ASSET_DIR}"

export MAMMO_FM_MODERNBERT_DIR="${MAMMO_FM_ASSET_DIR}/text_encoders/modernbert_mlm_5epochs"
```

Set the training-data locations and an external output directory. The image-label CSV is expected to contain the image
paths used by its records, so it does not require a separate image-root variable.

```bash
export MAMMO_FM_IMAGE_TEXT_1_IMAGE_DIR="/path/to/image-text-dataset-1/images"
export MAMMO_FM_IMAGE_TEXT_1_CSV="/path/to/image-text-dataset-1/training.csv"
export MAMMO_FM_IMAGE_LABEL_CSV="/path/to/image-label-dataset/training.csv"
export MAMMO_FM_IMAGE_TEXT_2_IMAGE_DIR="/path/to/image-text-dataset-2/images"
export MAMMO_FM_IMAGE_TEXT_2_CSV="/path/to/image-text-dataset-2/training.csv"
export MAMMO_FM_OUTPUT_DIR="${MAMMO_FM_ASSET_DIR}/pretraining/b5_modernbert_2048"
```

From the repository root and with the Mammo-FM environment activated, submit the two-GPU SGE job using the
[`b5_det_2048.sh`](src/scripts/breast-clip-train/mammo_fm/b5_det_2048.sh) launcher:

```bash
mkdir -p "${MAMMO_FM_OUTPUT_DIR}/logs"
qsub -V -o "${MAMMO_FM_OUTPUT_DIR}/logs" \
  src/scripts/breast-clip-train/mammo_fm/b5_det_2048.sh
```

On an interactive node where two GPUs have already been allocated, run the same launcher directly:

```bash
bash src/scripts/breast-clip-train/mammo_fm/b5_det_2048.sh
```

## Mammo-FM (image-text alignment) Contrastive Checkpoints

Following are the pre-training checkpoints of Mammo-FM:

<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th align="left">Model architecture</th>
      <th align="left">Checkpoints (Hugging Face)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Best Checkpoint — Mammo-FM (4 institutions, 140k patients, 821k mammograms)</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-FM_BatmanlabTrained_CLIP.tar">Mammo-FM_BatmanlabTrained_CLIP.tar</a></td>
    </tr>
    <tr>
      <td>Last Epoch checkpoint — Mammo-FM image encoder used by Mammo-GRG</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-FM_BatmanlabTrained_CLIP_last.tar">Mammo-FM_BatmanlabTrained_CLIP_last.tar</a></td>
    </tr>
    <tr>
      <td>Checkpoint — Mammo-FM (Mayo only, 77k patients, 465k mammograms)</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-FM_ASU_Trained_CLIP.tar">Mammo-FM_ASU_Trained_CLIP.tar</a></td>
    </tr>
  </tbody>
</table>

## Evaluation (Diagnosis)

### Zero-shot evaluation of Mammo-FM

The B5 zero-shot workflow evaluates the Mammo-FM checkpoint on CMMD, NL-Breast, 🇹🇷 LUMINA, RSNA, and the VinDr
finding and abnormality tasks. It uses the same prompt ensembles as the official evaluation and writes per-image scores
and JSON metrics outside the source repository.

First download both the [Mammo-FM checkpoint](#mammo-fm-image-text-alignment-contrastive-checkpoints) and the
[domain-adapted ModernBERT checkpoint](#pretraining-mammo-fm) into `Mammo-FM-Test-check`. Then set the public-dataset
roots. Each root must contain the preprocessed PNG directory and corresponding evaluation CSV described in
[Data Download](#data-download).

```bash
export MAMMO_FM_REPO_ROOT="$(pwd)"
export MAMMO_FM_ASSET_DIR="/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check"
export MAMMO_FM_CLIP_CKPT="${MAMMO_FM_ASSET_DIR}/Mammo-FM_BatmanlabTrained_CLIP.tar"
export MAMMO_FM_MODERNBERT_DIR="${MAMMO_FM_ASSET_DIR}/text_encoders/modernbert_mlm_5epochs"
export MAMMO_FM_ZERO_SHOT_DIR="${MAMMO_FM_ASSET_DIR}/zero_shot/b5"

export MAMMO_FM_CMMD_DATA_DIR="/path/to/ChineseMammoDataset"
export MAMMO_FM_NL_BREAST_DATA_DIR="/path/to/NL-Breast-Screen-data"
export MAMMO_FM_LUMINA_DATA_DIR="/path/to/LUMINA"
export MAMMO_FM_RSNA_DATA_DIR="/path/to/RSNA_Breast_Imaging/Dataset"
```

From the repository root, with the `Mammo-FM` environment activated, run the local
[`b5.sh`](src/scripts/zero_shot/mammo_fm/b5.sh) launcher:

```bash
bash src/scripts/zero_shot/mammo_fm/b5.sh
```

The evaluation uses fold 0 for CMMD, NL-Breast, and RSNA; the official `test` split for both VinDr tasks; and all
🇹🇷 LUMINA images, matching the original zero-shot workflow. Outputs are written to `${MAMMO_FM_ZERO_SHOT_DIR}` and
include `df_zs_<dataset>.csv`, `results-<dataset>.json`, the resolved configuration, and the evaluation log. No model
weights or evaluation outputs are stored inside the Mammo-FM repository.

### Linear probe/Full finetune scripts

Dataset-specific bash entrypoints are available in [`src/scripts`](src/scripts). Set the shared environment variables
described below, then run the linked launcher for the required experiment. Completed checkpoints, SCC logs, prediction
tables, and release metadata are available in the [released downstream classification checkpoints](#released-downstream-classification-checkpoints).

#### Released downstream classification checkpoints

The completed data-fraction-1.0 classification runs are released on Hugging Face under
[linear probing](https://huggingface.co/batmanLab/Mammo-FM/tree/main/linear_probe_mammo_FM_image_encoder)
and [full fine-tuning](https://huggingface.co/batmanLab/Mammo-FM/tree/main/fullfinetune_mammo_FM_image_encoder).
All runs use seed 10 and fold 0; each checkpoint was selected by validation AUROC. RSNA/VinDr density,
RSNA invasive subtype, and the CMMD checkpoints linked below come from completed **15-epoch** runs under
`portrait_15epochs`; the other linked classifiers come from nine-epoch runs. Linear probing freezes the
Mammo-FM image encoder, whereas full fine-tuning updates the encoder and classification head.

<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th align="left">Dataset</th>
      <th align="left">Task</th>
      <th align="left">Linear-probe checkpoint</th>
      <th align="left">Full-finetuning checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🇺🇸 RSNA</td>
      <td>Cancer</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/rsna/cancer_classification/checkpoint/mammo_fm_rsna_cancer_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/rsna/cancer_classification/checkpoint/mammo_fm_rsna_cancer_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇺🇸 RSNA</td>
      <td>Density A/B/C/D</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/rsna/density_classification/portrait_15epochs/checkpoint/mammo_fm_rsna_density_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/rsna/density_classification/portrait_15epochs/checkpoint/mammo_fm_rsna_density_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇺🇸 RSNA</td>
      <td>Invasive vs non-invasive (cancer-positive cases)</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/rsna/invasive_cancer_classification/portrait_15epochs/checkpoint/mammo_fm_rsna_invasive_cancer_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/rsna/invasive_cancer_classification/portrait_15epochs/checkpoint/mammo_fm_rsna_invasive_cancer_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr</td>
      <td>Density A/B/C/D</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/vindr/density_classification/portrait_15epochs/checkpoint/mammo_fm_vindr_density_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/vindr/density_classification/portrait_15epochs/checkpoint/mammo_fm_vindr_density_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr</td>
      <td>Abnormality</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/vindr/abnormality_classification/checkpoint/mammo_fm_vindr_abnormality_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/vindr/abnormality_classification/checkpoint/mammo_fm_vindr_abnormality_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr</td>
      <td>Mass</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/vindr/mass_classification/checkpoint/mammo_fm_vindr_mass_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/vindr/mass_classification/checkpoint/mammo_fm_vindr_mass_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr</td>
      <td>Suspicious calcification</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/vindr/suspicious_calcification_classification/checkpoint/mammo_fm_vindr_suspicious_calcification_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/vindr/suspicious_calcification_classification/checkpoint/mammo_fm_vindr_suspicious_calcification_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇨🇳 CMMD</td>
      <td>Cancer</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/cmmd/cancer_classification/portrait_15epochs/checkpoint/mammo_fm_cmmd_cancer_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/cmmd/cancer_classification/portrait_15epochs/checkpoint/mammo_fm_cmmd_cancer_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇨🇳 CMMD</td>
      <td>Mass</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/cmmd/mass_classification/portrait_15epochs/checkpoint/mammo_fm_cmmd_mass_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/cmmd/mass_classification/portrait_15epochs/checkpoint/mammo_fm_cmmd_mass_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇨🇳 CMMD</td>
      <td>Calcification</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/cmmd/calcification_classification/portrait_15epochs/checkpoint/mammo_fm_cmmd_calcification_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/cmmd/calcification_classification/portrait_15epochs/checkpoint/mammo_fm_cmmd_calcification_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇨🇦 NL-Breast</td>
      <td>Cancer</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/nl_breast/cancer_classification/checkpoint/mammo_fm_nl_breast_cancer_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/nl_breast/cancer_classification/checkpoint/mammo_fm_nl_breast_cancer_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
    <tr>
      <td>🇹🇷 LUMINA</td>
      <td>Cancer</td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/linear_probe_mammo_FM_image_encoder/lumina/cancer_classification/checkpoint/mammo_fm_lumina_cancer_linear_probe_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
      <td><a href="https://huggingface.co/batmanLab/Mammo-FM/blob/main/fullfinetune_mammo_FM_image_encoder/lumina/cancer_classification/checkpoint/mammo_fm_lumina_cancer_full_finetune_seed10_fold0_best_validation_auroc.pth">Checkpoint</a></td>
    </tr>
  </tbody>
</table>


Download both released classifier collections to the external asset directory with:

```bash
MAMMO_FM_ASSET_DIR="/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check"

huggingface-cli download batmanLab/Mammo-FM \
  --include "linear_probe_mammo_FM_image_encoder/**" \
            "fullfinetune_mammo_FM_image_encoder/**" \
  --local-dir "${MAMMO_FM_ASSET_DIR}"
```

Download the primary Mammo-FM checkpoint into the external test directory:

```bash
MAMMO_FM_ASSET_DIR="/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check"

huggingface-cli download batmanLab/Mammo-FM \
  Mammo-FM_BatmanlabTrained_CLIP.tar \
  --local-dir "${MAMMO_FM_ASSET_DIR}"

export MAMMO_FM_CLIP_CKPT="${MAMMO_FM_ASSET_DIR}/Mammo-FM_BatmanlabTrained_CLIP.tar"
```

Both linear-probe and full-finetuning jobs request `-pe omp 4`. Linear probes request `-l mem_per_core=4G` (16 GB total host memory); full fine-tuning requests `-l mem_per_core=8G` (32 GB total).

All diagnosis classification inputs use **1520 pixels high × 912 pixels wide**, for every dataset and for
training, validation, testing, and zero-shot inference. 


#### Density and invasive versus non-invasive cancer

The [density/subtype launcher](src/scripts/submit_density_invasive.sh) trains both linear probes and fully fine-tuned
models from **`Mammo-FM_BatmanlabTrained_CLIP.tar`**, with 15 epochs,
seed 10, and all available training data. SGE requests `gpu_c=7.0` for linear probes and `gpu_c=8.0` for full fine-tuning;
each job requests one GPU, four CPU slots, and 24 hours. Linear probes request 4 GB RAM per slot (16 GB total), while full fine-tuning requests 8 GB per slot (32 GB total). 


```bash
export MAMMO_FM_EVAL_DIR="/path/to/Mammo-FM-Test-check/downstream_eval/portrait_density_invasive"
export LUMINA_DENSITY_METADATA="${MAMMO_FM_EVAL_DIR}/metadata/lumina_density.csv"
python src/scripts/prepare_lumina_density.py \
  --images-csv /path/to/LUMINA/lumina_metadata.csv \
  --benign-xlsx /path/to/LUMINA/Benign_Cases.xlsx \
  --malign-xlsx /path/to/LUMINA/Malign_Cases.xlsx \
  --output "${LUMINA_DENSITY_METADATA}"

export MAMMO_FM_CLIP_CKPT="/path/to/Mammo-FM-Test-check/Mammo-FM_BatmanlabTrained_CLIP.tar"
export MAMMO_FM_CONDA_ENV="/path/to/your/classification-conda-env"
bash src/scripts/submit_density_invasive.sh
```

Set the dataset roots in the launcher for your installation. It also submits new CMMD cancer/mass/calcification
LP/FT runs using the corrected portrait geometry; set `RERUN_CMMD=0` to omit those reruns. Use `DRY_RUN=1` to preview
jobs. Checkpoints, four-class probability CSVs, logs, and completion metadata stay in the external output directory.
These newly trained endpoints are not automatically added to the published Python package or Hugging Face release.

#### Downstream launcher table

<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th align="left">Dataset</th>
      <th align="left">Task and default targets</th>
      <th align="left">Launcher</th>
      <th align="left">Run command</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🇺🇸 RSNA</td>
      <td>Classification: cancer</td>
      <td><a href="src/scripts/rsna_classification_run.sh"><code>rsna_classification_run.sh</code></a></td>
      <td><code>bash src/scripts/rsna_classification_run.sh</code></td>
    </tr>
    <tr>
      <td>🇨🇳 CMMD</td>
      <td>Classification: cancer, mass, calcification</td>
      <td><a href="src/scripts/cmmd_classification_run.sh"><code>cmmd_classification_run.sh</code></a></td>
      <td><code>bash src/scripts/cmmd_classification_run.sh</code></td>
    </tr>
    <tr>
      <td>🇨🇦 NL-Breast</td>
      <td>Classification: cancer</td>
      <td><a href="src/scripts/nl_breast_classification_run.sh"><code>nl_breast_classification_run.sh</code></a></td>
      <td><code>bash src/scripts/nl_breast_classification_run.sh</code></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr</td>
      <td>Classification: abnormality</td>
      <td><a href="src/scripts/vindr_abnormal_classification_run.sh"><code>vindr_abnormal_classification_run.sh</code></a></td>
      <td><code>bash src/scripts/vindr_abnormal_classification_run.sh</code></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr</td>
      <td>Finding classification: mass, suspicious calcification, asymmetry</td>
      <td><a href="src/scripts/vindr_classification_run.sh"><code>vindr_classification_run.sh</code></a></td>
      <td><code>bash src/scripts/vindr_classification_run.sh</code></td>
    </tr>
    <tr>
      <td>🇻🇳 VinDr</td>
      <td>Detection: mass, suspicious calcification</td>
      <td><a href="src/scripts/vindr_detection_run.sh"><code>vindr_detection_run.sh</code></a></td>
      <td><code>bash src/scripts/vindr_detection_run.sh</code></td>
    </tr>
  </tbody>
</table>

The blocks below are single-run examples that mirror those scripts. Update paths for your environment and pick the
checkpoint you want to evaluate. `_lp` and `_ft` denotes linear probing and full finetuning respectively for `--arch`
parameter. Run these commands from the Mammo-FM repository root. All Python entrypoints, configs, and qsub templates
used by the launchers are resolved from this repository; no separate downstream codebase is required.

Keep downloaded checkpoints and evaluation artifacts outside the source repository:

```bash
export MAMMO_FM_REPO_ROOT="$(pwd)"
export MAMMO_FM_ASSET_DIR="/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-FM-Test-check"
export MAMMO_FM_CLIP_CKPT="${MAMMO_FM_ASSET_DIR}/Mammo-FM_BatmanlabTrained_CLIP.tar"
export MAMMO_FM_EVAL_DIR="${MAMMO_FM_ASSET_DIR}/evaluation"
export MAMMO_FM_CONDA_ENV="Mammo-FM"
```

#### RSNA classification

```bash
DATASET_NAME="RSNA"
CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
IMG_DIR="RSNA_Cancer_Detection/train_images_png"
CSV_FILE="RSNA_Cancer_Detection/train_folds.csv"
EXTRA_ARGS="--label 'cancer' --n_folds 1"
LABEL="cancer"

python "${MAMMO_FM_REPO_ROOT}/src/codebase/train_classifier.py" \
  --tensorboard-path "${MAMMO_FM_EVAL_DIR}/tensorboard" \
  --checkpoints "${MAMMO_FM_EVAL_DIR}/checkpoints" \
  --output_path "${MAMMO_FM_EVAL_DIR}/outputs" \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --data_frac "1.0" \
  --dataset "$DATASET_NAME" \
  --arch "breast_clip_det_b5_period_n_lp" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --epochs 15 \
  --batch-size 2 \
  --num-workers 2 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --print-freq 500 \
  --log-freq 500 \
  --running-interactive 'y' \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --label "$LABEL" \
  --n_folds 1
```

#### VinDr classification (abnormal)

```bash
DATASET_NAME="VINDR"
CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
IMG_DIR="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png"
CSV_FILE="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds_abnormal.csv"
TASK_TYPE="classification"
WEIGHTED_BCE="n"
LABEL="abnormal"
EXTRA_ARGS="--label 'abnormal' --n_folds 1"

python "${MAMMO_FM_REPO_ROOT}/src/codebase/train_classifier.py" \
  --tensorboard-path "${MAMMO_FM_EVAL_DIR}/tensorboard" \
  --checkpoints "${MAMMO_FM_EVAL_DIR}/checkpoints" \
  --output_path "${MAMMO_FM_EVAL_DIR}/outputs" \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --data_frac "1.0" \
  --dataset "$DATASET_NAME" \
  --arch "breast_clip_det_b5_period_n_lp" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --epochs 15 \
  --batch-size 2 \
  --num-workers 2 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --print-freq 500 \
  --log-freq 500 \
  --running-interactive 'n' \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --label "$LABEL" \
  --n_folds 1
```

#### CMMD classification (cancer)

```bash
DATASET_NAME="CMMD"
CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"
DATA_DIR="/restricted/projectnb/batmanlab/shawn24/Additional_Breast_data/ChineseMammoDataset"
IMG_DIR="cmmd_png"
CSV_FILE="cmmd_png/merged_final_cmmd.csv"
TASK_TYPE="classification"
WEIGHTED_BCE="n"
LABEL="cancer"
EXTRA_ARGS="--label 'cancer' --n_folds 1"

python "${MAMMO_FM_REPO_ROOT}/src/codebase/train_classifier.py" \
  --tensorboard-path "${MAMMO_FM_EVAL_DIR}/tensorboard" \
  --checkpoints "${MAMMO_FM_EVAL_DIR}/checkpoints" \
  --output_path "${MAMMO_FM_EVAL_DIR}/outputs" \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --data_frac "1.0" \
  --dataset "$DATASET_NAME" \
  --inference_mode "y" \
  --arch "breast_clip_det_b5_period_n_ft" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --epochs 15 \
  --batch-size 8 \
  --num-workers 2 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --print-freq 500 \
  --log-freq 500 \
  --running-interactive 'n' \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --label "$LABEL" \
  --n_folds 1
```

#### NL-Breast classification (cancer)

```bash
DATASET_NAME="NLBreast"
CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/NL-Breast-Screen-data"
IMG_DIR="NLBS_Data_png_v1"
CSV_FILE="nlbs_image_labels_v2.csv"
TASK_TYPE="classification"
WEIGHTED_BCE="n"
LABEL="cancer"
EXTRA_ARGS="--label 'cancer' --n_folds 1"

python "${MAMMO_FM_REPO_ROOT}/src/codebase/train_classifier.py" \
  --tensorboard-path "${MAMMO_FM_EVAL_DIR}/tensorboard" \
  --checkpoints "${MAMMO_FM_EVAL_DIR}/checkpoints" \
  --output_path "${MAMMO_FM_EVAL_DIR}/outputs" \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --data_frac "1.0" \
  --dataset "$DATASET_NAME" \
  --arch "breast_clip_det_b5_period_n_ft" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --epochs 15 \
  --batch-size 8 \
  --num-workers 2 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --warmup-epochs 1 \
  --print-freq 500 \
  --log-freq 500 \
  --running-interactive 'y' \
  --weighted-BCE 'y' \
  --balanced-dataloader 'n' \
  --label "$LABEL" \
  --n_folds 1
```

#### VinDr detection (Mass)

```bash
DATASET_NAME="ViNDr"
DATA_DIR="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset"
IMG_DIR="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images_png"
CSV_FILE="External/Vindr/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/vindr_detection_v1_folds.csv"
TASK_TYPE="Detection"


CLIP_CKPT="${MAMMO_FM_CLIP_CKPT}"
LABEL="Mass"
ARCH="breast_clip_det_b5"

python "${MAMMO_FM_REPO_ROOT}/src/codebase/train_detector.py" \
  --tensorboard-path "${MAMMO_FM_EVAL_DIR}/tensorboard" \
  --checkpoints "${MAMMO_FM_EVAL_DIR}/checkpoints" \
  --output_path "${MAMMO_FM_EVAL_DIR}/outputs" \
  --data-dir "$DATA_DIR" \
  --img-dir "$IMG_DIR" \
  --csv-file "$CSV_FILE" \
  --dataset "$DATASET_NAME" \
  --arch "$ARCH" \
  --epochs 120 \
  --batch-size 7 \
  --freeze_backbone "n" \
  --data_frac 1.0 \
  --concepts "$LABEL" \
  --clip_chk_pt_path "$CLIP_CKPT" \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive 'n' \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2
```

## Risk Prediction (Prognosis)

The prognostic component is available in [`src/codebase/risk`](src/codebase/risk). It includes Stage 1 knowledge
distillation from five MIRAI risk probabilities, Stage 2 supervised five-year risk training, SAE training, image
activation maps and mammogram overlays, per-exam risk intervention/Shapley analysis, and sentence-level text
attribution. BU and BUMC references are retained where they identify the original cohort or a required source field.

Use the [risk README](src/codebase/risk/README.md) for the complete data contract and runnable commands. The released
[BU risk heads](https://huggingface.co/batmanLab/Mammo-FM/tree/main/risk_checkpoints/BU_S1_KD_S2)
are the newly trained **S1 MIRAI KD → S2 BU outcomes** model and its **S2-only** comparison. The S1→S2 head is the
package and CLI default. Download links and exact release/source hashes are listed in the
[checkpoint documentation](src/codebase/risk/docs/HF_CHECKPOINTS.md).
Downloaded weights, images, reports, feature arrays, patient metadata, SAE checkpoints, and run outputs must remain
outside the source repository.

The heads were trained on 7,425 BU pretraining exams, selected on 2,475 validation exams, and evaluated on the exact
exam complement of 3,794 exams from 3,380 patients. This evaluation is exam-disjoint but **not patient-disjoint**:
every test patient has another pretraining exam. Each head produces five direct, independent sigmoid probabilities.
No fitted calibrator or cross-horizon monotonic correction is applied, so these research scores must not be described
as clinically validated absolute risks. The validation split was also used for checkpoint selection, and 853
validation patients recur in 942 test exams.

Both releases are inference-only risk heads. The package separately downloads and verifies the frozen
`Mammo-FM_BatmanlabTrained_CLIP.tar` encoder, then strictly loads the selected head. The cleaned training runner froze the
encoder and its BatchNorm state while training the risk aggregator and all five horizon heads.

## Radiology Report Generation (Mammo-GRG)

The public four-view report-generation component is available in
[`src/codebase/Mammo-GRG`](src/codebase/Mammo-GRG). It includes the exact Mammo-FM feature-extraction path,
the code-only [instruction-data generators](src/codebase/Mammo-GRG/create_instruction_tuning_data), Stage-1 projector
training, Stage-2 LoRA instruction tuning, inference, one demonstration notebook, and local report evaluation with an
LLM parser, BLEU, ROUGE, and Mammo-GREEN.

Use the [Mammo-GRG README](src/codebase/Mammo-GRG/README.md) for the complete workflow. The released Stage-1, Stage-2,
and grounding checkpoints are stored in the
[`Mammo-GRG` Hugging Face directory](https://huggingface.co/batmanLab/Mammo-FM/tree/main/Mammo-GRG). Keep downloaded
weights, feature maps, training outputs, and generated reports outside this source repository.

Create the dedicated report-generation environment from
[`src/codebase/Mammo-GRG/environment.yml`](src/codebase/Mammo-GRG/environment.yml) before running the Mammo-GRG
training, inference, demo, or local evaluation commands.

## Citation

**Mammo-FM**

```bibtex
@article{ghosh2025mammo,
  title={Mammo-FM: Breast-specific foundational model for Integrated Mammographic Diagnosis, Prognosis, and Reporting},
  author={Ghosh, Shantanu and Joshi, Vedant Parthesh and Syed, Rayan and Kassem, Aya and Varshney, Abhishek and Basak, Payel and Dai, Weicheng and Gichoya, Judy Wawira and Trivedi, Hari M and Banerjee, Imon and others},
  journal={arXiv preprint arXiv:2512.00198},
  year={2025}
}
```

**Mammo-CLIP (MICCAI 2024)**

```bibtex
@InProceedings{10.1007/978-3-031-72390-2_59,
author="Ghosh, Shantanu
and Poynton, Clare B.
and Visweswaran, Shyam
and Batmanghelich, Kayhan",
editor="Linguraru, Marius George
and Dou, Qi
and Feragen, Aasa
and Giannarou, Stamatia
and Glocker, Ben
and Lekadir, Karim
and Schnabel, Julia A.",
title="Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="632--642",
isbn="978-3-031-72390-2"
}
```

## License and copyright

The source code is Licensed under [Apache License 2.0](LICENSE.txt).
Model weights is Licensed
under [Custom Academic License for Model Weights](LICENSE_WEIGHTS.txt).

Copyright © [Batman Lab](https://www.batman-lab.com/), 2026

## Contact

For any queries, contact [Shantanu Ghosh](https://shantanu-ai.github.io/) (email: **shawn24@bu.edu**)

## Contributing

Did you try Mammo-FM on other datasets containing 2D-Mammograms and want to report the results? Feel free to send
a [pull request](https://github.com/batmanlab/Mammo-FM/pulls).

## Acknowledgement

We thank the [CT-CHAT](https://github.com/ibrahimethemhamamci/CT-CHAT) repository for its contribution to the
report-generation component of Mammo-FM.
