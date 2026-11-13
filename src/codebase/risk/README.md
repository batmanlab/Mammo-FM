# Mammo-FM risk prediction and interpretation

This directory contains the portable Mammo-FM prognostic pipeline: Stage 1
knowledge distillation from five MIRAI risk probabilities, Stage 2 supervised
five-year risk training, sparse-autoencoder (SAE) training, image activation
maps and mammogram overlays, per-exam risk intervention/Shapley analysis, and
sentence-level text attribution.

BU and BUMC names are retained where they describe the original cohort or an
established data-field contract. No images, reports, patient-level tables,
feature caches, predictions, or SAE weights are included.

## Table of Contents

1. [Directory layout](#directory-layout)
2. [Environment](#environment)
3. [Risk checkpoints](#risk-checkpoints)
4. [Stage 1: MIRAI knowledge distillation](#stage-1-mirai-knowledge-distillation)
5. [Stage 2: supervised five-year risk training](#stage-2-supervised-five-year-risk-training)
6. [SAE training](#sae-training)
7. [Image activation maps and mammogram overlays](#image-activation-maps-and-mammogram-overlays)
8. [Risk intervention and Shapley attribution](#risk-intervention-and-shapley-attribution)
9. [Text attribution heatmaps](#text-attribution-heatmaps)
10. [Validation](#validation)

## Directory layout

```text
scripts/legacy/                         Stage 1 -> Stage 2 launcher
scripts/run_sae.sh                      SAE command-line entry point
src/codebase/mayo_mammofm_risk/         five-horizon risk model and losses
src/codebase/AsymMirai-master/           preserved Mayo DICOM data loader
src/codebase/mammo_sae/                  SAE and interpretation workflows
config/                                  SAE model/data examples
docs/DATA_CONTRACT.md                    required manifest fields
docs/HF_CHECKPOINTS.md                   released checkpoint provenance
docs/SOURCE_MANIFEST.md                  copied scope and intentional changes
tests/                                   synthetic, data-free checks
```

The risk code reuses the `breastclip` image/text model implementation already
present in the parent Mammo-FM repository. The AsymMirai and Breast-CLIP source
licenses are retained under `licenses/`.

## Environment

From this directory, install the known-working dependencies:

```bash
python -m pip install -r requirements.txt
```

Text attribution uses a newer Transformers release and should be installed in
a separate environment after the base requirements:

```bash
python -m pip install -r requirements-sae-text.txt
```

Set these paths once for the examples below:

```bash
export MAMMO_FM_ROOT=/path/to/Mammo-FM
export RISK_ROOT="$MAMMO_FM_ROOT/src/codebase/risk"
export PYTHONPATH="$RISK_ROOT/src/codebase:$MAMMO_FM_ROOT/src/codebase${PYTHONPATH:+:$PYTHONPATH}"
```

## Risk checkpoints

The public release contains two newly trained BU inference heads:

| Protocol | Checkpoint |
| --- | --- |
| S1 MIRAI KD → S2 BU outcomes (default) | [Download](https://huggingface.co/batmanLab/Mammo-FM/resolve/7605810fab99c90eb9c9ff23206c237bafa3f500/risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S1_KD_S2_head.pt) |
| S2 BU outcomes only | [Download](https://huggingface.co/batmanLab/Mammo-FM/resolve/7605810fab99c90eb9c9ff23206c237bafa3f500/risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S2_only_head.pt) |

The five old fold checkpoints and old release manifest have been removed. Both
new files contain only the 5.84-million-parameter four-view risk head and use
the same frozen
[`Mammo-FM_BatmanlabTrained_CLIP.tar`](https://huggingface.co/batmanLab/Mammo-FM/blob/7605810fab99c90eb9c9ff23206c237bafa3f500/Mammo-FM_BatmanlabTrained_CLIP.tar)
encoder. The `mammofm` package downloads and hash-verifies both artifacts and
strictly combines them for inference. S1→S2 is the default; use
`--bu-model s2-only` for the comparison.

```bash
hf download batmanLab/Mammo-FM \
  --revision 7605810fab99c90eb9c9ff23206c237bafa3f500 \
  --include 'risk_checkpoints/BU_S1_KD_S2/*' \
  --local-dir /path/to/checkpoints
```

The training/validation/test sizes were 7,425/2,475/3,794 examinations. The
test complement is exact-exam-disjoint but not patient-disjoint: all 3,380 test
patients have another examination in pretraining. The five head outputs are
converted directly from five independent logits to sigmoid probabilities. No
fitted calibrator or cross-horizon monotonic correction is applied. These are
research model probabilities, not clinically validated absolute risks.
Validation was also used for checkpoint selection, and 853 validation patients
contribute 942 test exams. See
[docs/HF_CHECKPOINTS.md](docs/HF_CHECKPOINTS.md) for checkpoint downloads and
provenance.

## Stage 1: MIRAI knowledge distillation

Stage 1 trains five risk heads against MIRAI probabilities supplied in the
manifest columns `1_year_risk` through `5_year_risk`. It does not invoke or
ship a MIRAI model. The data loader keeps the existing patient-disjoint split
values and complete four-view exam selection; it does not create new folds.

## Stage 2: supervised five-year risk training

In an S1→S2 run, Stage 2 initializes from the matching best Stage 1 checkpoint
and trains the five-horizon transformer/risk heads with censor-masked,
class-weighted BCE.
In this cleaned implementation, the Mammo-FM image encoder and its BatchNorm
state remain frozen in both stages. Thus “full fine-tuning on risk” refers to
full supervised training of the risk aggregator and all five risk heads, not
fine-tuning the Mammo-FM image encoder.

Run both stages and then evaluate an existing held-out split:

```bash
cd "$RISK_ROOT"
export CSV=/path/to/image_level_five_year_risk_manifest.csv
export MAMMO_CHECKPOINT=/path/to/Mammo-FM_BatmanlabTrained_CLIP.tar
export OUT=/path/to/risk_outputs
export PYTHON_BIN=python
export NPROC=1
export FOLDS="1"
export SPLIT_COL='split_fold_{fold}'
export TRAIN_SPLIT='train_fold_{fold}'
export VAL_SPLIT='val_fold_{fold}'
export PREDICT_SPLIT='test_fold_{fold}'
bash scripts/legacy/mayo_mammofm_stage1_stage2_inference.sh
```

For a single split column containing `train`, `valid`, and `test`, set
`SPLIT_COL=split`, `TRAIN_SPLIT=train`, `VAL_SPLIT=valid`, and
`PREDICT_SPLIT=test`. The full schema is documented in
[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md).

The generic `mayo_mammofm_risk.run` entrypoint uses the preserved **image-level
DICOM** training loader and remains separate from the new public four-view
inference path. End-user inference is available through `mammofm exam`, which
accepts named LMLO/LCC/RMLO/RCC PNG or DICOM inputs, loads the frozen Batmanlab
encoder plus the selected BU head, and defaults to S1→S2.

## SAE training

SAEs must be trained on spatial features extracted from the same encoder and
preprocessing contract that will be interpreted. The existing
`mammo_sae.risk_spatial` adapter expects a complete legacy
encoder-plus-risk-model checkpoint; it does **not** accept either new head-only
BU release directly. For that legacy adapter, the configuration has this form:

```json
{
  "checkpoint": "/path/to/complete_encoder_and_risk_model.pt",
  "device": "cuda"
}
```

Then extract, split by patient, and train:

```bash
cd "$RISK_ROOT"
bash scripts/run_sae.sh prepare-manifest \
  --input /path/to/image_manifest.csv \
  --columns config/sae_manifest_columns.json \
  --output /path/to/sae/manifest.csv

bash scripts/run_sae.sh extract \
  --manifest /path/to/sae/manifest.csv \
  --factory mammo_sae.risk_spatial:risk_checkpoint_spatial \
  --adapter-config /path/to/risk_spatial_adapter.json \
  --output /path/to/sae/features

bash scripts/run_sae.sh split \
  --features /path/to/sae/features/features.npy \
  --metadata /path/to/sae/features/metadata.csv \
  --group patient_id --fraction 0.15 --seed 42 \
  --output /path/to/sae/split

bash scripts/run_sae.sh train-original \
  --train /path/to/sae/split/train.npy \
  --val /path/to/sae/split/val.npy \
  --model-config config/sae_matryoshka.json \
  --epochs 30 --batch-size 4096 --device cuda \
  --output /path/to/sae/training
```

Keep SAE training, neuron selection, and final evaluation patient-disjoint.
The example finding configuration is a schema placeholder, not a validated
finding-to-neuron map.

## Image activation maps and mammogram overlays

`heatmaps` saves per-neuron activation grids (`.npy`), standalone hot-color
previews, and a manifest. It does not overlay activations on a mammogram:

```bash
bash scripts/run_sae.sh heatmaps \
  --features /path/to/heldout_features.npy \
  --metadata /path/to/heldout_metadata.csv \
  --checkpoint /path/to/sae/training/last.pt \
  --neurons 10 20 --device cuda \
  --output /path/to/sae/image_activation_maps
```

`localize` is the image-overlay route. It requires `image_path` plus complete
`patch_h`, `patch_w`, `grid_h`, and `grid_w` metadata:

```bash
bash scripts/run_sae.sh localize \
  --features /path/to/heldout_features.npy \
  --metadata /path/to/heldout_metadata.csv \
  --checkpoint /path/to/sae/training/last.pt \
  --finding-config /path/to/validated_findings.json \
  --device cuda --output /path/to/sae/mammogram_overlays
```

## Risk intervention and Shapley attribution

The existing `mammo_sae.adapters:legacy_risk` factory also requires a complete
legacy encoder-plus-risk-model checkpoint and does not load the new public
head-only files. Its configuration has this form:

```json
{
  "checkpoint": "/path/to/complete_encoder_and_risk_model.pt",
  "device": "cuda"
}
```

Intervene on selected latents for one complete four-view examination:

```bash
bash scripts/run_sae.sh intervene \
  --features /path/to/heldout_features.npy \
  --metadata /path/to/heldout_metadata.csv \
  --checkpoint /path/to/sae/training/last.pt \
  --neurons 10 20 \
  --factory mammo_sae.adapters:legacy_risk \
  --adapter-config /path/to/risk_predictor.json \
  --patient-id PATIENT_ID --exam-id EXAM_ID \
  --device cuda --output /path/to/sae/intervention.json
```

Estimate Shapley values over either the selected neurons or all SAE latents:

```bash
bash scripts/run_sae.sh shapley \
  --features /path/to/heldout_features.npy \
  --metadata /path/to/heldout_metadata.csv \
  --checkpoint /path/to/sae/training/last.pt \
  --neurons 10 20 --background selected --samples 100 --seed 42 \
  --factory mammo_sae.adapters:legacy_risk \
  --adapter-config /path/to/risk_predictor.json \
  --patient-id PATIENT_ID --exam-id EXAM_ID \
  --device cuda --output /path/to/sae/shapley.json
```

These are per-exam model effects. They are not cohort-level causal effects.

## Text attribution heatmaps

Text attribution suppresses selected image latents and measures the change in
cosine similarity to each supplied report sentence. It requires a matching
multimodal Mammo-FM checkpoint containing the trained text encoder and both
projection heads; the risk checkpoint alone is insufficient. A text adapter
configuration has this form:

```json
{
  "checkpoint": "/path/to/multimodal_mammo_fm_checkpoint.tar",
  "text_model_directory": "/path/to/local_modernbert",
  "device": "cuda"
}
```

Run one image and one or more selected neurons:

```bash
bash scripts/run_sae.sh text \
  --features /path/to/one_image_features.npy \
  --checkpoint /path/to/sae/training/last.pt \
  --neurons 10 \
  --factory mammo_sae.adapters:breastclip_text \
  --adapter-config /path/to/text_adapter.json \
  --sentences /path/to/report_sentences.json \
  --device cuda --output /path/to/sae/text_attribution.json
```

The command writes JSON and a companion HTML heatmap. Values are raw
image-sentence similarity changes, not cancer probabilities or risk changes.

## Validation

The copied workflows have data-free synthetic tests:

```bash
cd "$RISK_ROOT"
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$RISK_ROOT/src/codebase:$MAMMO_FM_ROOT/src/codebase" \
python -m unittest discover -s tests -v
```

All command outputs and downloaded checkpoints should remain outside this
source directory.

The exact copied scope, exclusions, and small compatibility changes are listed
in [docs/SOURCE_MANIFEST.md](docs/SOURCE_MANIFEST.md).
