# mammofm Python package

Local inference with Mammo-FM: single-image `mammofm.diagnosis` and
`mammofm.features`, four-view `mammofm.prognosis` and `mammofm.report_gen`, and a four-view `mammofm
exam` command that produces a Stage-3 grounded report plus separate Mayo- and
BU-trained five-horizon risk JSON objects. Diagnosis supports RSNA, VinDr,
CMMD, NL-Breast and LUMINA.

**Research only.** Diagnosis and grounding outputs are uncalibrated model
scores. Mayo- and BU-trained risk outputs are the models' direct sigmoid
probabilities; no fitted calibrator or cross-horizon monotonic correction is
applied. They are not validated clinical patient-level absolute risks.
Generated reports require expert review. Read the repository's
`LICENSE.txt` and `LICENSE_WEIGHTS.txt`; packaging does not change their terms.
Images stay on your machine. The package has no report-generation API key,
telemetry, image upload, or remote inference service.

## Contents

- [Installation](#installation)
- [Diagnosis](#diagnosis)
- [Supported tasks and missing models](#supported-tasks-and-missing-models)
- [Image features](#image-features)
- [Custom zero-shot prompts](#custom-zero-shot-prompts)
- [Prognosis](#prognosis)
- [Grounded report generation](#grounded-report-generation)
- [DICOM and PNG preprocessing](#dicom-and-png-preprocessing)
  - [Standalone DICOM preprocessing](#standalone-dicom-preprocessing)
- [Checkpoints, caching, and offline use](#checkpoints-caching-and-offline-use)
- [API reference](#api-reference)
- [Testing and publishing](#testing-and-publishing)
- [Four-view report and risk CLI](#four-view-report-and-risk-cli)

## Installation

Python 3.10+ and PyTorch 2.6+ are required. CPU inference is supported; a compatible
CUDA-enabled PyTorch installation is recommended for repeated inference. Install
the appropriate PyTorch build for your machine before installing this package.

From the repository root:

```bash
python -m pip install .
# Include decoders for compressed mammography DICOMs:
python -m pip install '.[dicom]'
```

The published distribution name is `mammofm`:

```bash
python -m pip install --upgrade 'mammofm[dicom]>=0.4.1'
# For preprocessed PNGs and uncompressed DICOMs only:
python -m pip install --upgrade 'mammofm>=0.4.1'
```

The `mf.prognosis` API does not require the report extra or language-model
weights. To use the full exam CLI or report API, install the optional dependencies:

```bash
python -m pip install --upgrade 'mammofm[report,dicom]>=0.4.1'
```

No model weights or patient data are bundled in the wheel or source distribution.

On a shared CPU host with a low process/thread limit, set these before starting
Python to bound numerical and tokenizer worker pools:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1
```

You can additionally set `cv2.setNumThreads(2)` in your application. The package
does not change process-wide thread settings. The latter PyTorch setting avoids
an import-time compiler worker pool in some Transformers versions even though
Mammo-FM uses eager, noncompiled inference.

## Diagnosis

```python
import json
import mammofm as mf

output = mf.diagnosis('/data/mammogram.dcm', 'vindr')
print(json.dumps(output, indent=2, allow_nan=False))

# A preprocessed PNG uses the same API.
output = mf.diagnosis('/data/mammogram.png', 'nlbreast')

# LUMINA cancer: zero-shot, linear-probe and full-finetuning probabilities.
output = mf.diagnosis('/data/mammogram.dcm', 'lumina')

# CMMD cancer, mass and calcification, using the same 1520H x 912W geometry.
output = mf.diagnosis('/data/mammogram.dcm', 'cmmd')

# RSNA cancer, conditional invasive subtype and four-class density.
# Select LP/FT because density and invasive subtype have no packaged zero-shot endpoint.
output = mf.diagnosis('/data/mammogram.png', 'rsna',
                      methods=['linear_probe', 'full_finetune'])
print(output['full_finetune']['density'])  # {'A': ..., 'B': ..., 'C': ..., 'D': ...}
```

`output` is a Python dictionary that can be directly serialized to JSON. Its
top-level prediction groups are `zero_shot`, `linear_probe`, and `full_finetune`.
Each maps binary task names to positive-class probabilities (Python floats).
For RSNA and VinDr, `density` maps to a dictionary of A/B/C/D class probabilities
that sum to one. A missing endpoint returns `None` (`null` in JSON) with a reason.

```python
output = mf.diagnosis('/data/mammogram.png', 'vindr')
output['zero_shot']['mass']
output['linear_probe']['mass']
output['full_finetune']['mass']
output['unavailable']  # Reasons for unavailable fields; never substitute zeros.
output['metadata']     # Input processing, pinned checkpoint identities, semantics.
```

No hard labels or clinical thresholds are returned. Binary classifiers use sigmoid;
density classifiers use four-class softmax. Binary zero-shot preserves the official evaluator:
normalize individual projected prompt embeddings, average within negative and
positive classes, compute cosine similarity to each mean, and apply a two-class
softmax. **The checkpoint's learned logit scale is not multiplied into these
scores**, matching the original evaluation.

To avoid unnecessary models or downloads:

```python
output = mf.diagnosis('/data/mammogram.png', 'rsna', methods=['linear_probe'])
print(mf.availability('rsna'))  # Does not load or download any weights.
```

Missing published models produce a warning and explicit null fields. Use
`strict=True` to raise `MissingModelError` before any model download if any
requested task/method is unavailable. Download failures, invalid inputs,
checkpoint mismatches, and actual inference failures always raise errors rather
than masquerading as a missing model.

## Supported tasks and missing models

Assets are pinned individually to immutable HF revisions and SHA-256 hashes;
the returned checkpoint metadata records the revision used for each method.

| Dataset argument | Task key | Zero-shot | Linear probe | Full finetune |
| --- | --- | --- | --- | --- |
| `rsna` | `cancer` | Available | Available | Available |
| `rsna` | `invasive_cancer` | -- | Available, conditional on cancer | Available, conditional on cancer |
| `rsna` | `density` | -- | Available, A/B/C/D | Available, A/B/C/D |
| `vindr` | `abnormality` | Available, historical surrogate | Available | Available |
| `vindr` | `mass` | Available | Available | Available |
| `vindr` | `suspicious_calcification` | Available | Available | Available |
| `vindr` | `density` | -- | Available, A/B/C/D | Available, A/B/C/D |
| `cmmd` | `cancer` | Available | Available | Available |
| `cmmd` | `mass` | Available | Available | Available |
| `cmmd` | `calcification` | Available, canonical calcification surrogate | Available | Available |
| `nlbreast` | `cancer` | Available | Available | Available |
| `lumina` | `cancer` | Available | Available | Available |

NL-Breast aliases such as `nl_breast` and `NL-Breast` are accepted. Unknown
diagnosis datasets are rejected. RSNA cancer is **not** an invasive-cancer
substitute. The `invasive_cancer` classifiers were trained on cancer-positive
cases with known subtype: 0 = non-invasive, 1 = invasive. Their outputs are
**conditional on cancer**, not estimates of population invasive-cancer risk.
The package does not multiply or otherwise combine these with the cancer score.

RSNA/VinDr density, RSNA invasive subtype and CMMD LP/FT use validation-selected
checkpoints from completed 15-epoch runs, stored under each task's
`portrait_15epochs` folder in the public classifier collections. All use
**1520 pixels high × 912 pixels wide**. CMMD zero-shot calcification preserves
the original evaluator's mapping to the existing suspicious-calcification prompts.
Density zero-shot is explicitly unavailable in the package; existing research
prompts are not substituted for a validated packaged endpoint.

LUMINA uses the released cancer classifiers for
[linear probing](https://huggingface.co/batmanLab/Mammo-FM/tree/main/linear_probe_mammo_FM_image_encoder/lumina/cancer_classification)
and [full finetuning](https://huggingface.co/batmanLab/Mammo-FM/tree/main/fullfinetune_mammo_FM_image_encoder/lumina/cancer_classification).
Its zero-shot branch uses the Batmanlab CLIP checkpoint and the canonical cancer prompts,
as do the other supported cancer endpoints. All three outputs are single-image scores.

VinDr abnormality means **BI-RADS 4/5 versus 1/2**, with BI-RADS 3 excluded in
evaluation. It does not mean any finding or pathology-confirmed cancer. Its
historical zero-shot surrogate uses the original `cancer` / `no cancer` prompts;
this limitation is also recorded in the output metadata.

## Image features

```python
import numpy as np
import mammofm as mf

output = mf.features('/data/mammogram.dcm', 'rsna')
spatial = output['before_pooling']
pooled = output['after_pooling']
print(spatial.shape)  # (1, 2048, 48, 29) at the standard image size
print(pooled.shape)   # (1, 2048)
np.savez('/outputs/image_features.npz', before_pooling=spatial, after_pooling=pooled)

# An unknown or omitted dataset uses general min-max DICOM preprocessing.
output = mf.features('/data/mammogram.png', 'my_dataset')
output = mf.features('/data/mammogram.dcm')
```

The arrays are full float32 CPU NumPy arrays, retaining the batch dimension.
They are the final image-encoder spatial map and its global average-pooled
vector, **before the CLIP projection and L2 normalization**. Feature arrays are
not automatically converted to enormous JSON lists. `metadata` identifies the
checkpoint and preprocessing. Use `.tolist()` explicitly if JSON is required.

Both feature extraction and zero-shot use the requested
[`Mammo-FM_BatmanlabTrained_CLIP.tar`](https://huggingface.co/batmanLab/Mammo-FM/blob/5e07ca4c4e88cd118b0c2cfaacadddfe69f30026/Mammo-FM_BatmanlabTrained_CLIP.tar).
The classifier paths load their own complete released encoder+head weights;
fine-tuned encoders are not replaced with the Batmanlab encoder.

## Custom zero-shot prompts

```python
import mammofm as mf

Pos_label = ["There is a mass in the breast.", "A breast mass is present."]
Neg_label = ["There is no mass in the breast.", "No breast mass is present."]
positive = mf.text_embeddings(Pos_label)
negative = mf.text_embeddings(Neg_label)
image = mf.image_embeddings('/data/mammogram.dcm', dataset='rsna')
output = mf.zero_shot(image, positive, negative)
print(output['positive_cosine'])
print(output['negative_cosine'])
print(output['score'])
```

Both embedding functions use the joint
[`Mammo-FM_BatmanlabTrained_CLIP.tar`](https://huggingface.co/batmanLab/Mammo-FM/blob/5e07ca4c4e88cd118b0c2cfaacadddfe69f30026/Mammo-FM_BatmanlabTrained_CLIP.tar)
checkpoint, including its learned image and text projections. Separate text assets
provide only tokenizer/configuration, not replacement text weights.

- `text_embeddings(prompts)` accepts a string or a nonempty sequence of nonblank
  strings and returns a float32 CPU NumPy array `[number_of_prompts, D]`. Rows retain
  prompt order and are L2-normalized. Tokenization uses the checkpoint's maximum
  text length; longer prompts are truncated.
- `image_embeddings(img, dataset=None)` accepts one DICOM or preprocessed PNG and
  returns a float32 CPU NumPy array `[1, D]` in the same projected CLIP space.
  Dataset-specific DICOM conversion and standard image normalization match the
  zero-shot diagnosis branch; unknown/omitted datasets use general preprocessing.
  Do not substitute `mf.features()` outputs: those are **unprojected** encoder features.
- `zero_shot(image_embeddings, positive_embeddings, negative_embeddings)` accepts
  these arrays (or real numeric lists/PyTorch tensors). It normalizes each prompt
  row, averages within each class, and normalizes each class average. It then
  computes cosine similarity to both class averages and their difference.

The JSON-serializable output has `positive_cosine`, `negative_cosine`, and `score`
lists, with one entry per image, plus `metadata.scoring`. Here
`score = positive_cosine - negative_cosine`; a higher score favors the supplied
positive prompts. These are **raw similarities, not probabilities**: no sigmoid,
softmax, learned temperature, threshold, or calibration is applied. This helper
does not change the canonical `mf.diagnosis()` endpoint or its scoring.

Embedding inference selects CUDA when available, otherwise CPU. Pass `device='cpu'`
or `device='cuda:0'` explicitly, together with the usual `cache_dir`, `model_dir`,
and `local_files_only` options when needed. A reusable client avoids reloading the
joint checkpoint and lets you set prompt batch size:

```python
model = mf.MammoFM(device='cuda', prompt_batch_size=16)
positive = model.text_embeddings(Pos_label)
negative = model.text_embeddings(Neg_label)
first = model.image_embeddings('/data/first.png', 'rsna')
second = model.image_embeddings('/data/second.png', 'rsna')

import numpy as np
output = model.zero_shot(np.concatenate([first, second]), positive, negative)
```

Reuse text embeddings across images; no models or downloads are needed for the
scoring helper itself. Inputs must use the same checkpoint and projected space.
Empty, nonfinite, zero-length, or dimension-mismatched vectors are rejected, as is
a class whose prompt vectors cancel to a zero average. Prompt text and images stay
local and are not saved or transmitted by these APIs. Custom prompts do not imply
a validated classifier or clinical performance for the requested finding.

## Prognosis

```python
import json
import mammofm as mf

img = {
    'lmlo': '/data/left_mlo.dcm',
    'lcc': '/data/left_cc.png',
    'rmlo': '/data/right_mlo.dcm',
    'rcc': '/data/right_cc.png',
}
output = mf.prognosis(img)
print(json.dumps(output, indent=2, allow_nan=False))

# A reusable client accepts the same four-view mapping.
model = mf.MammoFM(device='cpu', model_dir='/models/Mammo-FM', local_files_only=True)
output = model.prognosis(img)
```

The result is a JSON-serializable dictionary with exactly five keys:
`1_year`, `2_year`, `3_year`, `4_year`, and `5_year`. Each value is a Python
`float` between 0 and 1, expressed as a probability, not a percentage. No image
features, report, raw scores, or metadata are returned by this API.

Supply all four correctly labeled views from one examination. Keys are
case-insensitive (`lmlo` or `LMLO`, for example), and insertion order does not
matter: inference uses **LMLO → LCC → RMLO → RCC**. A single image cannot
substitute for the four views. DICOMs use the unified general preprocessing
converter; preprocessed PNGs bypass DICOM conversion. Model inputs use
**1520H × 912W** geometry and the standard image normalization. DICOM-derived
PNGs remain in a private temporary directory and are removed after inference.

The default is the BU **Stage 1 MIRAI knowledge distillation → Stage 2 risk
finetuning** model. Its uncalibrated v1 inference head is
`risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S1_KD_S2_head.pt`, paired with
`Mammo-FM_BatmanlabTrained_CLIP.tar`. The head contains the four-view transformer
and five risk outputs, not the image encoder; both assets are downloaded and
verified separately. For offline use, place both files under `model_dir` with
these relative paths.

Outputs are the head's direct, independent sigmoid probabilities. No fitted
calibrator or cross-horizon monotonicity correction is applied, so values can
decrease from one year to the next. Validation and test contain overlapping
patients; these research outputs are not independently validated clinical
patient-level absolute risks.

## Grounded report generation

```python
import mammofm as mf

img = {
    'lcc': '/data/left_cc.png',
    'lmlo': '/data/left_mlo.dcm',
    'rcc': '/data/right_cc.png',
    'rmlo': '/data/right_mlo.dcm',
}
output = mf.report_gen(img, device='cuda')
print(output['grounded_report'])
findings = output['findings']
print(findings['views']['lmlo']['mass'])  # probability and positive boolean
print(findings['left'])                 # breast-level positive booleans
print(findings['right'])
print(findings['bilateral'])            # positive on both breasts
```

The returned JSON-serializable dictionary has exactly two top-level keys:
`grounded_report` and `findings`. It does **not** return image features or the
intermediate preliminary report. Findings include per-view probabilities and
positive flags for `mass`, `suspicious_calcification`, and `asymmetry`, plus
left/right, exam-level and bilateral flags, and model provenance.
These are image-level findings, not spatial lesion localization or clinical
diagnoses. Generated text can still contain errors and requires expert review.

All four views must belong to the same examination. Keys are case-insensitive;
dictionary insertion order does not matter. The internal order is always
**LMLO → LCC → RMLO → RCC**. Missing/duplicate view names, unknown names and
constant images raise errors. A single image cannot be substituted for four
views. The caller is responsible for correct anatomical labels and exam grouping.

DICOMs always use the **unified general min-max preprocessing** here, regardless
of dataset, producing the 1520H × 912W PNG-equivalent pixels in memory. The
`report_gen` Python API does not write a temporary image; the `mammofm exam` CLI
intentionally materializes a private temporary PNG so every report/risk branch
consumes the same converted pixels, then deletes it. Preprocessed PNGs bypass DICOM conversion;
standard model resizing and normalization still apply.

The visual encoder is the RRG training checkpoint
[`Mammo-FM_BatmanlabTrained_CLIP_last.tar`](https://huggingface.co/batmanLab/Mammo-FM/blob/4b8766c3e00fc1f5b1bcbb606d78394a774a7ed4/Mammo-FM_BatmanlabTrained_CLIP_last.tar),
**not** the Batmanlab checkpoint used by `mf.features`. Stages 1 and 2 are training
stages: inference loads the final Stage-2 projector and LoRA adapter, which already
incorporate Stage-1 training. It does not overwrite them with the Stage-1 projector.
The three released pooled-data FT classifiers supply findings to a separate, unadapted
Llama-3.1-8B-Instruct model for Stage-3 grounding.

Finding decisions preserve the released grounding workflow and are not
prospectively validated clinical diagnoses. A breast is positive when either
of its views is positive; bilateral means both breasts are positive.

Report generation requires approved access to
[`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
and an existing Hugging Face login, or a complete local base-model directory:

```python
output = mf.report_gen(
    img, device='cuda',
    model_dir='/models/Mammo-FM',
    base_model_path='/models/Llama-3.1-8B-Instruct',
    local_files_only=True,
)
```

Under `model_dir`, retain the public HF paths for the epoch-3 image checkpoint,
`Mammo-GRG/stage2/checkpoint-5500/` and `Mammo-GRG/grounding/`. No separate Stage-1
projector download is required. Public Mammo-FM assets are SHA-256 verified; the
base model is pinned to revision `0e9e39f249a16976918f6564b8830bc894c89659` when
downloaded. A local base override is checked for architecture and completeness,
not claimed to be cryptographically identical to that revision.

All report generation runs locally: there is no OpenAI key or remote inference
call. The Llama base weights alone are about 16 GB. A CUDA GPU with sufficient
memory is strongly recommended; CPU inference uses float32 and is slow and
memory-intensive. Image models and the two language models are loaded
sequentially, and report generation clears this client's cached diagnosis models
to keep them from occupying GPU memory. It does not delete any disk cache.

Preliminary decoding is greedy (up to 512 tokens). Grounding preserves the
historical sampling settings, temperature 0.3 and top-p 0.9 (up to 320 tokens).
`seed=0` is the default. Hardware, precision and library versions can affect
outputs; exact historical text equality is not promised. In particular, this
implementation uses unquantized language-model weights; the historical Stage-3
launcher used NF4 quantization and unseeded sampling. The report extra uses
the tested Transformers 4.57.x and PEFT 0.14.x compatibility range, without a
FlashAttention, DeepSpeed or quantization requirement.

## DICOM and PNG preprocessing

### Standalone DICOM preprocessing

Use `mf.preprocess(img)` to convert one local DICOM path (`str` or `pathlib.Path`)
with the unified converter's default settings.

```python
import mammofm as mf

image = mf.preprocess('/data/mammogram.dcm')
image.save('/outputs/mammogram.png')  # Optional; the parent directory must exist.
```

The result is a Pillow `Image.Image` in grayscale mode `L`, **1520 pixels high ×
912 pixels wide** (`image.size == (912, 1520)`). `numpy.asarray(image)` gives its
8-bit pixel array. It uses unified min-max preprocessing for every input, without
a dataset argument or model-input normalization. It runs locally on CPU, without
loading models, downloading weights, or writing files unless you explicitly save
the result. DICOM metadata is not copied; burned-in text is not anonymized.
Only single-frame grayscale DICOMs are accepted, not PNGs. Compressed DICOMs
require a compatible decoder from the `dicom` extra.

### Inference image inputs

For `diagnosis` and `features`, `img` is a local image path; `prognosis` and
`report_gen` take a dictionary of four named image paths. Each image is a
single-frame grayscale DICOM or an already preprocessed 8-bit PNG. RGB PNGs
are also accepted. Color/multiframe DICOMs,
16-bit PNGs, and transparent PNGs are rejected rather than silently altered.

| Dataset | DICOM intensity route |
| --- | --- |
| RSNA, VinDr | Original CPU-style min-max scaling |
| NL-Breast | DICOM rescale + window center/width; 1st/99th-percentile fallback |
| CMMD, LUMINA | General min-max preprocessing |
| Unknown/omitted dataset in `features` | General min-max preprocessing |
| All `prognosis` and `report_gen` inputs | General min-max preprocessing |

DICOMs use the unified converter's five-pixel border removal, MONOCHROME1
correction, original breast-cropping heuristic, and `INTER_AREA` resize to
1520H x 912W. Public single-image APIs convert in memory. The `prognosis` API
and exam CLI write only short-lived metadata-free PNGs in a private temporary
directory and remove them after the run. PNGs skip this contrast/cropping pipeline. PNG input should already
have the intended breast preprocessing; being a PNG does not establish that.

Inference inputs then receive the model's RGB conversion, branch-specific resize
when needed (`INTER_LINEAR`), per-image min-max normalization and
`(x - 0.3089279) / 0.25053555408335154`. Constant images are handled without NaNs.

All enabled branches use **1520H x 912W**. For arbitrary PNG sizes, this resize
is a package standardization step; the original non-CMMD loaders expected
already-preprocessed PNG dimensions and did not apply it. Image preprocessing
does not update detection boxes, perform anatomical segmentation or anonymize
burned-in text. No patient identifiers or source paths are included in results.

The shared image operations preserve the original CPU conventions, not a
guarantee of byte-identical behavior for every legacy GPU/decoder path.

## Checkpoints, caching, and offline use

The first relevant inference downloads weights from the public
[`batmanLab/Mammo-FM`](https://huggingface.co/batmanLab/Mammo-FM) repository. The
Batmanlab joint checkpoint is about 2.15 GB; each classifier is downloaded separately
only when requested. No HF token is required for this pinned public release.

```python
model = mf.MammoFM(device='cuda', cache_dir='/models/huggingface')
result = model.diagnosis('/data/image.png', 'vindr')
features = model.features('/data/image.png', 'vindr')
model.clear_cache()  # Release this client's model references; keep disk cache.
```

For offline use, first download the published assets, preserving their HF
relative paths. Then use:

```python
model = mf.MammoFM(device='cpu', model_dir='/models/Mammo-FM', local_files_only=True)
output = model.diagnosis('/data/image.png', 'nlbreast')
```

The base checkpoint sits directly under `model_dir`; classifiers remain under
`linear_probe_mammo_FM_image_encoder/...` and `fullfinetune_mammo_FM_image_encoder/...`.
Zero-shot also requires `config.json`, `tokenizer.json`, `tokenizer_config.json`,
and `special_tokens_map.json` under `text_encoders/modernbert_mlm_5epochs/`.
It does **not** require that directory's separate MLM initialization weights:
fine-tuned text weights are already in the joint CLIP checkpoint.

`MAMMOFM_MODEL_DIR` and `MAMMOFM_CACHE_DIR` provide optional environment defaults.
Otherwise, downloads use the standard Hugging Face cache, not the codebase.
Every asset is pinned to a revision and SHA-256 checked before loading. Model
loading is strict; there is no silent random-weight or ImageNet-weight fallback.

## API reference

```python
mf.preprocess(img)
mf.diagnosis(img, dataset, *, methods=None, strict=False, device=None,
             cache_dir=None, model_dir=None, local_files_only=False)
mf.features(img, dataset=None, *, device=None, cache_dir=None,
            model_dir=None, local_files_only=False)
mf.text_embeddings(prompts, *, device=None, cache_dir=None,
                   model_dir=None, local_files_only=False)
mf.image_embeddings(img, dataset=None, *, device=None, cache_dir=None,
                    model_dir=None, local_files_only=False)
mf.zero_shot(image_embeddings, positive_embeddings, negative_embeddings)
mf.prognosis(img, *, device=None, cache_dir=None, model_dir=None,
             local_files_only=False)
mf.report_gen(img, *, device=None, cache_dir=None, model_dir=None,
              local_files_only=False, base_model_path=None, seed=0,
              max_new_tokens=512, grounding_max_new_tokens=320)
mf.run_exam(view_paths, output_dir, *, bu_model='s1s2', device=None,
            cache_dir=None, model_dir=None, local_files_only=False,
            base_model_path=None, report_seed=0,
            preliminary_max_new_tokens=512,
            grounding_max_new_tokens=320, only='all', overwrite=False)
mf.availability(dataset=None)
mf.clear_cache()

mf.MammoFM(*, device=None, cache_dir=None, model_dir=None,
           local_files_only=False, prompt_batch_size=32)
```

`mf.preprocess(img)` is a standalone CPU utility and takes no device or cache
arguments. For model inference, `device=None` selects CUDA when available,
otherwise CPU. `methods` selects
`zero_shot`, `linear_probe`, and/or `full_finetune`. All three are requested by
default. A reusable client caches its encoder, CLIP model, prognosis model,
prompt prototypes, and at most two classifier models in memory. For parallel workers, construct
one client per worker; clients are not promised to be thread-safe.
`MammoFM.report_gen` accepts the same report-specific keywords as the convenience
function; device and cache settings come from the client constructor.
`MammoFM.prognosis(img)` likewise uses the client's device and cache settings.
`MammoFM.text_embeddings` and `MammoFM.image_embeddings` share the client's CLIP
model and device settings. `mf.zero_shot` and `MammoFM.zero_shot` score already
computed embeddings on CPU without loading any model.

## Testing and publishing

Synthetic tests do not need private data, downloaded checkpoints or GPUs:

```bash
python -m unittest discover -s tests/package -p 'test_*.py'
```

Build and check release artifacts from a clean checkout:

```bash
python -m pip install '.[dev]'
python -m build
python -m twine check dist/*
```

Before publishing, inspect both archives: they must contain only the inference
package, public prompts, documentation, packaging metadata and license notices.
Do not include weights, images, private CSVs, test-inference artifacts, secrets,
editor/assistant files, or the original training code tree. Run an installed-wheel
smoke test outside the repository so accidental checkout imports cannot pass.

PyPI publication is a separate maintainer action requiring project ownership
and an authenticated upload or trusted-publishing configuration. Do not embed
upload tokens in the repository. Only after release review and publication:

```bash
python -m twine upload dist/*
```

## Four-view report and risk CLI

Supply exactly one image for each view. Each input may be an 8-bit PNG or a
single-frame grayscale DICOM:

```bash
mammofm exam \
  --lmlo /data/exam/LMLO.dcm \
  --lcc  /data/exam/LCC.png \
  --rmlo /data/exam/RMLO.dcm \
  --rcc  /data/exam/RCC.png \
  --output-dir /outputs/exam-001 \
  --device cuda
```

The default run writes exactly three public result files after all branches
succeed:

- `report.json` contains only the final Stage-3 classifier-grounded report. The
  internal Stage-2 preliminary report is never printed, returned, or saved.
- `risk_mayo.json` contains only the direct 1-, 2-, 3-, 4-, and 5-year sigmoid
  probabilities plus the Mayo model comment.
- `risk_bu.json` contains only the same five horizons plus the BMC model comment.
  It uses the Stage 1 MIRAI KD → Stage 2 BU checkpoint by default.

The same three objects are emitted as one JSON object on stdout; progress goes
to stderr. For `--only risk`, the complete public output is:

```json
{
  "mayo_risk": {
    "1yr": 0.1,
    "2yr": 0.2,
    "3yr": 0.3,
    "4yr": 0.4,
    "5yr": 0.5,
    "comment": "Image encoder is Mammo-FM multi-institution checkpoint. Risk heads are finetuned with Mayo risks. Risk scores are not calibrated. These are probabilities."
  },
  "bu_risk": {
    "1yr": 0.1,
    "2yr": 0.2,
    "3yr": 0.3,
    "4yr": 0.4,
    "5yr": 0.5,
    "comment": "Image encoder is Mammo-FM multi-institution checkpoint. Risk heads are finetuned with BMC risks. Risk scores are not calibrated. These are probabilities."
  }
}
```

The example numbers above are placeholders. Actual values are the model's
direct probabilities. `risk_mayo.json` stores the value of `mayo_risk`, while
`risk_bu.json` stores the value of `bu_risk`. Input paths, DICOM identifiers,
model/checkpoint metadata, preprocessing metadata, internal risk logits,
percentages, and calibration fields are excluded. Use `--quiet` to suppress
progress.

The canonical order is always **LMLO → LCC → RMLO → RCC**, independent of file
names or argument order. PNG pixels bypass DICOM conversion. Each DICOM goes
through the unified converter and is materialized as a metadata-free temporary
PNG; all downstream branches use that PNG, and it is deleted when the command
finishes. The four inputs must be correctly labeled views from one examination.

Useful modes and safeguards:

```bash
# Run both risk models without loading report language models.
mammofm exam ... --only risk

# Generate only the Stage-3 report.
mammofm exam ... --only report

# Use the optional Stage-2-only BU comparison head instead of the default S1→S2 head.
mammofm exam ... --bu-model s2-only

# Offline models and gated Llama base.
mammofm exam ... \
  --model-dir /models/Mammo-FM \
  --base-model-path /models/Llama-3.1-8B-Instruct \
  --local-files-only
```

The S1→S2 head is the CLI default for BU. Existing JSON files are not
overwritten unless `--overwrite` is supplied. You can also call
`mammofm.run_exam(...)` from Python with the same view mapping and options.

The five output keys are `1yr` through `5yr`. For both Mayo and BU, these are
the model heads' direct sigmoid probabilities on the 0–1 scale. The command
applies no fitted calibrator and no cumulative-maximum or other cross-horizon
monotonicity correction. Accordingly, the five values can decrease from one
horizon to the next and must not be interpreted as clinically validated
absolute risks.

For the BU experiment, the validation split was also used for checkpoint
selection and is not patient-disjoint from test. The selected protocol, exact head,
encoder, and revision provenance are documented by the release rather than
repeated in each compact risk JSON.
