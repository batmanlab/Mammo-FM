# Mammo-GRG: four-view mammography report generation

Mammo-GRG generates a screening-mammography report from four precomputed Mammo-FM feature maps in the fixed order
`LMLO`, `LCC`, `RMLO`, and `RCC`. This directory contains the dataset-neutral feature extractor, stage-1 and stage-2
training code, inference code, one demonstration notebook, and local evaluation utilities. Model weights and generated
artifacts must be stored outside the source repository.

## Table of Contents

1. [Generate the instruction-tuning data](#1-generate-the-instruction-tuning-data)
2. [Extract image embeddings](#2-extract-image-embeddings)
3. [Training RRG](#3-training-rrg)
   1. [Stage 1](#stage-1)
   2. [Stage 2](#stage-2)
   3. [Grounding and fact-checking classifiers](#grounding-and-fact-checking-classifiers)
4. [Inference](#4-inference)
   1. [Stage 2: preliminary report generation](#stage-2-preliminary-report-generation)
   2. [Stage 3: classifier-grounded report refinement](#stage-3-classifier-grounded-report-refinement)
5. [Pretraining RRG checkpoints](#5-pretraining-rrg-checkpoints)
6. [Demo](#6-demo)
7. [Evaluation](#7-evaluation)
   1. [LLM as parser](#llm-as-parser)
   2. [BLEU and ROUGE](#bleu-and-rouge)
   3. [Mammo-GREEN](#mammo-green)

Create the dedicated Mammo-GRG environment from the included [`environment.yml`](environment.yml). It reproduces the
Python 3.10, PyTorch 2.1.2, CUDA 11.8, DeepSpeed, PEFT, and FlashAttention stack used by the report generator. From the
Mammo-FM repository root, run:

```bash
cd src/codebase/Mammo-GRG
conda env create -f environment.yml
conda activate Mammo-GRG
python -m pip install flash-attn==2.7.3 --no-build-isolation
```

FlashAttention is installed separately because its required `--no-build-isolation` option cannot be represented
reliably as a package entry in a Conda environment file. If it must be compiled locally, ensure that a CUDA 11.8
toolkit is available. The stage-2 model is based on `meta-llama/Meta-Llama-3.1-8B-Instruct`; users must obtain access
to that base model separately.

Set an external asset root and download the released files:

```bash
export MAMMO_FM_ASSET_ROOT="/path/to/Mammo-FM-Test-check"
export MAMMO_GRG_ASSET_DIR="${MAMMO_FM_ASSET_ROOT}/Mammo-GRG"

huggingface-cli download batmanLab/Mammo-FM \
  Mammo-FM_BatmanlabTrained_CLIP_last.tar \
  --local-dir "${MAMMO_FM_ASSET_ROOT}"

huggingface-cli download batmanLab/Mammo-FM \
  --include "Mammo-GRG/*" \
  --local-dir "${MAMMO_FM_ASSET_ROOT}"
```

The remaining commands assume the current directory is `src/codebase/Mammo-GRG`.

## 1. Generate the instruction-tuning data

The code-only instruction-data workflow is available in
[`create_instruction_tuning_data`](create_instruction_tuning_data). Source reports, retrieved articles, and generated
instruction JSON files are intentionally excluded from both this repository and the Hugging Face release. Write every
generated artifact to an external, access-controlled directory:

```bash
export MAMMO_GRG_DATA_DIR="/path/to/access-controlled/Mammo-GRG-instruction-data"
mkdir -p "${MAMMO_GRG_DATA_DIR}"
```

The primary deterministic generator converts one report per exam into Stage-1 report-generation instructions:

```bash
python create_instruction_tuning_data/generate_report_questions.py \
  --input-file /path/to/report_manifest.csv \
  --output "${MAMMO_GRG_DATA_DIR}/report_generation.json" \
  --seed 0
```

The report manifest must contain `patient_exam_id`, `patient_id`, `exam_id`, `file_path`, `Report`, `dataset`, `fold`,
`mass_present`, `calc_present`, `arch_distortion_present`, and `asymmetry_present`. Each `file_path` value must identify
the four views in `LMLO`, `LCC`, `RMLO`, `RCC` order. The resulting `image` field is resolved against
`--embedding-root` by the training loader.

Additional generators preserve the report-derived instruction tasks used for Stage 2:

| Script | Generated instruction type |
|---|---|
| [`conversation_data_generate.py`](create_instruction_tuning_data/conversation_data_generate.py) | Three long-answer VQA conversations for any report length |
| [`conversation_data_generate_1set.py`](create_instruction_tuning_data/conversation_data_generate_1set.py) | One long-answer conversation; default report-length bucket is below 150 characters |
| [`conversation_data_generate_2sets.py`](create_instruction_tuning_data/conversation_data_generate_2sets.py) | Two long-answer conversations; default bucket is 150-250 characters |
| [`conversation_data_generate_3sets.py`](create_instruction_tuning_data/conversation_data_generate_3sets.py) | Three long-answer conversations; default bucket is above 350 characters |
| [`generate_freeresponse_description.py`](create_instruction_tuning_data/generate_freeresponse_description.py) | Free-response, description, and multiple-choice VQA |
| [`generate_short_answers.py`](create_instruction_tuning_data/generate_short_answers.py) | Finding-specific short-answer VQA |
| [`retrieve_pubmed.py`](create_instruction_tuning_data/retrieve_pubmed.py) | Optional retrieval of public breast-imaging articles from PubMed Central |
| [`generate_questions.py`](create_instruction_tuning_data/generate_questions.py) | Text-only article-derived questions |
| [`combine_json.py`](create_instruction_tuning_data/combine_json.py) | Ordered merge of generated JSON lists |

The language-model-backed generators read authentication from the `OPENAI_API_KEY` environment variable; no credential
is accepted as a command-line argument or stored in the source. Use them only with report text that is de-identified and
authorized for transmission to the selected external service. For example:

```bash
python create_instruction_tuning_data/conversation_data_generate.py \
  --input-file /path/to/deidentified_report_manifest.csv \
  --output "${MAMMO_GRG_DATA_DIR}/conversations_chunk_0.json" \
  --chunk-index 0 \
  --chunk-size 100
```

Merge the reviewed task files into the Stage-2 instruction list:

```bash
python create_instruction_tuning_data/combine_json.py \
  --input-files \
    "${MAMMO_GRG_DATA_DIR}/report_generation.json" \
    "${MAMMO_GRG_DATA_DIR}/conversations_chunk_0.json" \
  --output-file "${MAMMO_GRG_DATA_DIR}/stage2_instructions.json"
```

The training code accepts a JSON list with four embedding paths and alternating `human`/`gpt` messages. Paths should be
relative to the directory passed through `--embedding-root`:

```json
[
  {
    "id": "exam-001",
    "embeddings": [
      "exam-001/LMLO.pt",
      "exam-001/LCC.pt",
      "exam-001/RMLO.pt",
      "exam-001/RCC.pt"
    ],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nGenerate the screening-mammography report.<report_generation>"
      },
      {
        "from": "gpt",
        "value": "Findings: [reference findings]\nImpression: [reference impression]"
      }
    ]
  }
]
```

Stage 1 requires one `human`/`gpt` pair per sample. Stage 2 also supports additional alternating question-answer turns.
Do not place protected health information or generated instruction data in this source repository or its Hugging Face
checkpoint release.

## 2. Extract image embeddings

The report generator was trained with the
[Mammo-FM image encoder](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-FM_BatmanlabTrained_CLIP_last.tar).
The extractor reproduces the training preprocessing and saves one pre-pooling feature tensor with shape
`[1, 2048, 48, 29]` for each 1520x912 PNG.

Prepare a CSV with a `file_path` column, then run:

```bash
export MAMMO_FM_CHECKPOINT="${MAMMO_FM_ASSET_ROOT}/Mammo-FM_BatmanlabTrained_CLIP_last.tar"

bash scripts/extract_image_embeddings.sh \
  --manifest /path/to/images.csv \
  --image-column file_path \
  --image-root /path/to/png-images \
  --output-root /path/to/image-embeddings \
  --device cuda
```

The output tree mirrors paths relative to `--image-root`, replacing each image suffix with `.pt`.

## 3. Training RRG

The launchers use DeepSpeed and retain the released training configuration: a 512-query attentional projector followed
by LoRA instruction tuning of Llama 3.1 8B. Keep all outputs outside this repository.

### Stage 1

Stage 1 freezes the language model and trains the visual projector for one epoch:

```bash
bash scripts/train_stage1.sh \
  --deepspeed configs/zero3.json \
  --data-path /path/to/stage1_instructions.json \
  --embedding-root /path/to/image-embeddings \
  --output-dir /path/to/Mammo-FM-Test-check/Mammo-GRG-runs/stage1
```

### Stage 2

Stage 2 initializes the projector from Stage 1 and performs two epochs of LoRA instruction tuning:

```bash
bash scripts/train_stage2.sh \
  --deepspeed configs/zero3.json \
  --data-path /path/to/stage2_instructions.json \
  --embedding-root /path/to/image-embeddings \
  --stage1-checkpoint /path/to/Mammo-FM-Test-check/Mammo-GRG-runs/stage1/mm_projector.bin \
  --output-dir /path/to/Mammo-FM-Test-check/Mammo-GRG-runs/stage2
```

### Grounding and fact-checking classifiers

Mass, calcification, and asymmetry classifiers provide structured image-level evidence for report grounding and
fact-checking. To train new classifiers, initialize them with the Mammo-FM image encoder and use either the
[linear-probe or full-finetune workflows](../../../README.md#linear-probefull-finetune-scripts). Linear probing freezes
the image encoder; full fine-tuning updates it. The released grounding files are the best-validation checkpoints
from the pooled-data full-finetuning runs. They are image-level classifiers and do not provide pixel-level or bounding-box localization.

For new pooled-data grounding experiments, use the [data preparation script](../../scripts/prepare_grounding_data.py)
and [full-finetuning launcher](../../scripts/submit_grounding_ft.sh). 
Create a private source-path configuration outside the repository:

```json
{
  "vindr": {"csv": "/data/vindr/vindr_detection_v1_folds.csv", "image_root": "/data/vindr/images_png"},
  "embed": {"csv": "/data/embed/Embed_classification_diagnostic_data_wo_dup_folds.csv"},
  "reports": {
    "train_csv": "/data/reports/train_report_gen_stage1_image_level_with_density.csv",
    "test_csv": "/data/reports/val_report_gen_stage1_image_level_with_density.csv"
  }
}
```

The report-image CSV historically named `val` supplies the **held-out test** cohort. Its patients are excluded from
the training pool before making a deterministic patient-level training/validation split. Existing EMBED folds and
VinDr's official held-out test partition are preserved. VinDr's `patient_id` is a study identifier, so disjointness can
only be verified at study level for that source. Duplicate VinDr annotations are combined using a logical OR.
The preparation script checks all PNG paths and patient/study split overlaps before writing private manifests.
It does not copy report text. Keep these manifests and all training artifacts outside the public repository.

From the Mammo-FM repository root, with the classification environment active:

```bash
export MAMMO_FM_EVAL_DIR=/path/to/Mammo-FM-Test-check/grounding_ft/my_run
export MAMMO_FM_CLIP_CKPT=/path/to/Mammo-FM_BatmanlabTrained_CLIP.tar
export MAMMO_FM_CONDA_ENV=/path/to/classification_conda_env
python src/scripts/prepare_grounding_data.py \
  --config /private/path/grounding_sources.json \
  --output-dir "$MAMMO_FM_EVAL_DIR/metadata"
DRY_RUN=1 bash src/scripts/submit_grounding_ft.sh
bash src/scripts/submit_grounding_ft.sh
```

The SCC launcher requests one GPU (`gpu_c=8.0`), 4 CPU slots, 8 GB RAM per slot (32 GB total), and 24 hours for each task.
Defaults are 15 epochs, seed 10, batch size 8, AdamW learning rate `5e-5`, and full encoder fine-tuning.
Checkpoint selection uses validation exam/breast AUROC.
Calcification retains each source's native target: VinDr uses suspicious calcification, while the other sources
record calcification presence. It is therefore not a uniformly suspicious-only endpoint. Released files contain
model weights only; later training checkpoints do not automatically replace a published snapshot.

## 4. Inference

### Stage 2: preliminary report generation

Pass exactly four `.pt` files in `LMLO`, `LCC`, `RMLO`, `RCC` order:

```bash
bash scripts/infer.sh \
  --model-path "${MAMMO_GRG_ASSET_DIR}/stage2/checkpoint-5500" \
  --model-base meta-llama/Meta-Llama-3.1-8B-Instruct \
  --embeddings \
    /path/to/LMLO.pt \
    /path/to/LCC.pt \
    /path/to/RMLO.pt \
    /path/to/RCC.pt \
  --output /path/to/generated_report.json
```

Add `--load-4bit` when supported by the local GPU environment to reduce memory use.

### Stage 3: classifier-grounded report refinement

Stage 3 combines the Stage-2 preliminary report with structured image-level predictions from the Mammo-FM mass,
calcification, and asymmetry classifiers. The terms zero-shot, linear probe, and full finetune identify the source of
those structured predictions; Stage 3 itself uses the gated `meta-llama/Llama-3.1-8B-Instruct` model and has no separate
learned Stage-3 checkpoint.

Prepare a CSV containing `patient_id`, `exam_id`, `generated_report`, and `zero_shot_per_image`. The
`zero_shot_per_image` cell must be a serialized list with one dictionary per image. Each dictionary must provide
`image_laterality`, `image_view_type`, `mass_pred`, `suspicious calcification_pred`, and `asymmetry_pred`. Then run:

```bash
python scripts/final_stage.py \
  --input-csv /path/to/stage3_ft_input.csv \
  --output-csv /path/to/stage3_ft_reports.csv \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --max-new-tokens 320 \
  --temperature 0.3 \
  --top-p 0.9
```

The output CSV preserves the input columns and adds `final_generated_report_zs`. The released Stage-3 implementation
uses stochastic sampling and does not set a random seed, so an independently regenerated report is not guaranteed to
be byte-identical to an earlier run. Architectural-distortion predictions are not consumed by this released refiner.

## 5. Pretraining RRG checkpoints

The released checkpoint set is available in the
[Mammo-GRG Hugging Face directory](https://huggingface.co/batmanLab/Mammo-FM/tree/main/Mammo-GRG). The
[manifest](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-GRG/manifest.json) records byte sizes, SHA-256
hashes, architecture metadata, and source roles for every file.

| Component | Released artifact | Purpose |
|---|---|---|
| Mammo-FM image encoder | [`Mammo-FM_BatmanlabTrained_CLIP_last.tar`](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-FM_BatmanlabTrained_CLIP_last.tar) | Exact epoch-3 encoder used to create RRG feature maps |
| Stage 1 | [`stage1/mm_projector.bin`](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-GRG/stage1/mm_projector.bin) | 512-query attentional visual projector |
| Stage 2 | [`stage2/checkpoint-5500`](https://huggingface.co/batmanLab/Mammo-FM/tree/main/Mammo-GRG/stage2/checkpoint-5500) | LoRA adapter, non-LoRA trainables, tokenizer, and model configuration |
| Grounding: mass | [`grounding/mass_best.pth`](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-GRG/grounding/mass_best.pth) | Image-level mass classifier |
| Grounding: calcification | [`grounding/calcification_best.pth`](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-GRG/grounding/calcification_best.pth) | Image-level calcification classifier |
| Grounding: asymmetry | [`grounding/asymmetry_best.pth`](https://huggingface.co/batmanLab/Mammo-FM/blob/main/Mammo-GRG/grounding/asymmetry_best.pth) | Image-level asymmetry classifier |

The Llama base-model weights are not duplicated in this checkpoint release.

## 6. Demo

Use [`demo/four_view_report_generation.ipynb`](demo/four_view_report_generation.ipynb) for a single four-view example.
Set the external Stage-2 checkpoint directory and four embedding paths in the configuration cell before running it.

## 7. Evaluation

All three evaluators run locally. They do not require a hosted inference service or a service credential.

### LLM as parser

Parse generated reports into finding, laterality, view, and BI-RADS fields with a local instruction model:

```bash
python evaluation/llm_parser.py \
  --input-csv /path/to/reports.csv \
  --output-csv /path/to/parsed_reports.csv \
  --report-col generated_report \
  --model /path/to/local-instruction-model
```

### BLEU and ROUGE

Compute corpus BLEU-1 through BLEU-4 and mean ROUGE-L:

```bash
python evaluation/text_metrics.py \
  --input-csv /path/to/reports.csv \
  --reference-col ground_truth_report \
  --hypothesis-col generated_report \
  --output-json /path/to/text_metrics.json
```

### Mammo-GREEN

Run the mammography-specific clinical error judge with a local instruction model:

```bash
python evaluation/mammo_green.py \
  --input-csv /path/to/reports.csv \
  --output-csv /path/to/mammo_green_results.csv \
  --reference-col ground_truth_report \
  --hypothesis-col generated_report \
  --model /path/to/local-instruction-model
```