"""Public preprocessing, diagnosis, image-feature, prognosis and report APIs."""

from collections import OrderedDict
from functools import lru_cache
from importlib import resources
import json
import warnings

from ._assets import AssetStore, provenance
from ._errors import MissingModelError
from ._similarity import zero_shot
from ._registry import (
    CLASSIFIERS, ENCODER, PROGNOSIS_ENCODER, REPO_ID, REVISION,
    RISK_CHECKPOINTS, TASKS, TASK_DEFINITIONS, TEXT_ASSETS, UNAVAILABLE_REASON,
)

METHODS = ("zero_shot", "linear_probe", "full_finetune")
DENSITY_CLASSES = ("A", "B", "C", "D")


@lru_cache(maxsize=1)
def _prompts():
    return json.loads(resources.files("mammofm").joinpath("data/diagnosis_prompts.json").read_text(encoding="utf-8"))


def _prompt_task(task):
    # The original evaluator maps its suspicious-calcification prompts to CMMD's calc label.
    return "suspicious_calcification" if task == "calcification" else task


def _dataset(dataset, required=True):
    from ._inputs import normalize_dataset
    name = normalize_dataset(dataset)
    if required and name not in TASKS:
        raise ValueError("Unknown diagnosis dataset {!r}; choose {}".format(dataset, ", ".join(TASKS)))
    return name


def _missing_reason(dataset, task, method):
    if method == "zero_shot":
        reason = UNAVAILABLE_REASON.get((dataset, task, method))
        if reason is not None:
            return reason
        if _prompt_task(task) not in _prompts():
            return "No released canonical zero-shot prompts for {}; other task prompts are not substituted.".format(task)
    elif (dataset, task, method) not in CLASSIFIERS:
        return "No {} checkpoint for {}/{} in the pinned public release.".format(method, dataset, task)
    return None


def _methods(methods):
    if methods is None:
        return METHODS
    if isinstance(methods, str):
        methods = (methods,)
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("methods must be a nonempty, unique selection of {}".format(METHODS))
    return methods


def availability(dataset=None):
    """Return task/model availability without downloading weights or reading images."""
    datasets = (_dataset(dataset),) if dataset is not None else tuple(TASKS)
    return {
        name: {
            method: {
                task: {"available": _missing_reason(name, task, method) is None,
                       "reason": _missing_reason(name, task, method)}
                for task in TASKS[name]
            } for method in METHODS
        } for name in datasets
    }


class MammoFM:
    """Reusable, lazily loaded inference client.

    Parameters are keyword-only. ``model_dir`` mirrors the HF repository layout;
    ``cache_dir`` is a Hugging Face cache. ``local_files_only=True`` prevents
    network access. Images are always processed locally. No weights are shipped
    inside the Python distribution.
    """

    zero_shot = staticmethod(zero_shot)

    def __init__(self, *, device=None, cache_dir=None, model_dir=None,
                 local_files_only=False, prompt_batch_size=32):
        if not isinstance(prompt_batch_size, int) or isinstance(prompt_batch_size, bool) or prompt_batch_size < 1:
            raise ValueError("prompt_batch_size must be a positive integer")
        self._requested_device = device
        self._device = None
        self.store = AssetStore(cache_dir, model_dir, local_files_only)
        self.prompt_batch_size = prompt_batch_size
        self._encoder = None
        self._risk_model = None
        self._clip = None
        self._tokenizer = None
        self._prototypes = {}
        self._classifiers = OrderedDict()

    @property
    def device(self):
        if self._device is None:
            import torch
            self._device = torch.device(self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
            if self._device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable; use device='cpu'")
        return self._device

    def clear_cache(self):
        """Release model/prototype references held by this client; keep disk assets."""
        self._encoder = self._clip = self._tokenizer = self._risk_model = None
        self._prototypes.clear()
        self._classifiers.clear()

    def _get_encoder(self):
        if self._encoder is None:
            from ._model import load_encoder
            self._encoder = load_encoder(self.store.resolve(ENCODER), device=self.device)
        return self._encoder

    def _get_risk_model(self):
        if self._risk_model is None:
            from ._risk import load_risk_model
            self._risk_model = load_risk_model(
                self.store.resolve(RISK_CHECKPOINTS["bu"]["s1s2"]),
                device=self.device,
                encoder_checkpoint=self.store.resolve(PROGNOSIS_ENCODER),
            )
        return self._risk_model

    def _get_clip(self):
        if self._clip is None:
            from pathlib import Path
            from shutil import copyfile
            from tempfile import TemporaryDirectory
            from transformers import AutoTokenizer
            from ._model import load_clip
            paths = [self.store.resolve(asset) for asset in TEXT_ASSETS.values()]
            checkpoint = self.store.resolve(ENCODER)
            # A partially populated model_dir can resolve remaining files from
            # the HF cache. Co-locate only verified public text assets so neither
            # library sees an incomplete directory or unrelated local configs.
            with TemporaryDirectory(prefix="mammofm-text-") as temporary:
                directory = Path(temporary)
                for path in paths:
                    copyfile(path, directory / path.name)
                tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
                model = load_clip(checkpoint, text_config_dir=directory, device=self.device)
            self._tokenizer, self._clip = tokenizer, model
        return self._clip

    def _get_classifier(self, dataset, task, method):
        key = (dataset, task, method)
        if key not in self._classifiers:
            from ._model import load_classifier
            asset = CLASSIFIERS[key]
            num_classes = 4 if task == "density" else 1
            if asset.get("num_classes", num_classes) != num_classes:
                raise ValueError("Registered classifier output count does not match its task")
            # Keep at most two classifier models resident; all weights stay cached on disk.
            if len(self._classifiers) >= 2:
                self._classifiers.popitem(last=False)
            self._classifiers[key] = load_classifier(
                self.store.resolve(asset), device=self.device, num_classes=num_classes)
        self._classifiers.move_to_end(key)
        return self._classifiers[key]

    def _prototype(self, task):
        import torch
        import numpy as np
        task = _prompt_task(task)
        if task not in self._prototypes:
            model = self._get_clip()
            means = []
            for label in ("negative", "positive"):
                texts = _prompts()[task][label]
                encoded = []
                for start in range(0, len(texts), self.prompt_batch_size):
                    tokens = self._tokenizer(
                        texts[start:start + self.prompt_batch_size], padding="longest",
                        truncation=True, return_tensors="pt", max_length=model.text_max_length,
                    )
                    tokens = {name: value.to(self.device) for name, value in tokens.items()}
                    with torch.inference_mode():
                        encoded.append(model.encode_text_normalized(tokens).detach().float().cpu().numpy())
                mean = np.concatenate(encoded, axis=0).mean(axis=0)
                if not np.isfinite(mean).all() or np.linalg.norm(mean) == 0:
                    raise RuntimeError("Invalid zero-shot text prototype")
                means.append(mean)
            prototypes = torch.from_numpy(np.stack(means)).to(self.device)
            self._prototypes[task] = torch.nn.functional.normalize(prototypes, dim=1)
        return self._prototypes[task]

    def text_embeddings(self, prompts):
        """Return projected, unit-normalized float32 CPU arrays, one row per prompt."""
        from collections.abc import Sequence
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, Sequence) and not isinstance(prompts, (bytes, bytearray)):
            prompts = list(prompts)
        else:
            raise TypeError("prompts must be a string or a nonempty sequence of strings")
        if not prompts:
            raise ValueError("prompts must not be empty")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("every prompt must be a string")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must not contain empty or whitespace-only strings")

        import numpy as np
        import torch
        model = self._get_clip()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(prompts), self.prompt_batch_size):
                batch = prompts[start:start + self.prompt_batch_size]
                tokens = self._tokenizer(
                    batch, padding="longest", truncation=True, return_tensors="pt",
                    max_length=model.text_max_length,
                )
                tokens = {name: value.to(self.device) for name, value in tokens.items()}
                batches.append(_embedding_array(model.encode_text_normalized(tokens), len(batch)))
        if len({batch.shape[1] for batch in batches}) != 1:
            raise RuntimeError("Text encoder produced inconsistent embedding dimensions")
        return np.concatenate(batches, axis=0)

    def image_embeddings(self, img, dataset=None):
        """Return a projected, unit-normalized float32 CPU CLIP image array [1, D]."""
        import torch
        from ._inputs import inference_tensor, load_image
        name = _dataset(dataset, required=False)
        image, _ = load_image(img, name)
        tensor = inference_tensor(image, branch="clip", dataset=name).to(self.device)
        with torch.inference_mode():
            values = self._get_clip().encode_image_normalized(tensor)
        return _embedding_array(values, 1)

    def features(self, img, dataset=None):
        """Return unprojected spatial and average-pooled encoder features as NumPy arrays.

        Shapes include the batch dimension: before_pooling [1,2048,H',W'] and
        after_pooling [1,2048]. Unknown datasets use general min-max preprocessing.
        """
        import torch
        from ._inputs import inference_spec, inference_tensor, load_image
        name = _dataset(dataset, required=False)
        image, info = load_image(img, name)
        tensor = inference_tensor(image, branch="features", dataset=name).to(self.device)
        with torch.inference_mode():
            result = self._get_encoder().features(tensor)
        arrays = {}
        for key in ("before_pooling", "after_pooling"):
            values = result[key].detach().float().cpu()
            if not torch.isfinite(values).all():
                raise RuntimeError("Encoder produced nonfinite features")
            arrays[key] = values.numpy()
        arrays["metadata"] = {
            "dataset": name, "input": info,
            "inference": inference_spec("features", name),
            "checkpoint": provenance(ENCODER),
            "feature_space": "image_encoder_before_CLIP_projection_and_L2_normalization",
        }
        return arrays

    def prognosis(self, img):
        """Return BU S1 KD + S2 probabilities keyed ``1_year`` through ``5_year``.

        ``img`` must map LMLO, LCC, RMLO and RCC (case-insensitive) to four
        local DICOM/8-bit PNG paths from one exam. DICOMs use the unified
        converter; PNGs bypass conversion. All inputs use 1520-high x 912-wide
        inference tensors. Values are the head's direct sigmoid probabilities:
        no calibration or cross-horizon monotonic correction is applied. They
        are research-only, independently estimated horizons, not validated
        clinical absolute risks. No output files, reports or Mayo model are
        produced or loaded.
        """
        from ._risk import predict_risks
        from .exam import _prepared_views, _view_paths

        paths = _view_paths(img)
        with _prepared_views(paths) as (images, _, _):
            model = self._get_risk_model()
            probabilities = predict_risks(
                model, images)[0].detach().float().cpu().tolist()
        if len(probabilities) != 5:
            raise RuntimeError("Risk model did not produce exactly five probabilities")
        return {f"{year}_year": _probability(value)
                for year, value in enumerate(probabilities, start=1)}

    def report_gen(self, img, *, base_model_path=None, seed=0,
                   max_new_tokens=512, grounding_max_new_tokens=320):
        """Generate a grounded report from explicitly named four-view image paths.

        ``img`` maps LMLO, LCC, RMLO, RCC (case-insensitive) to DICOM or
        preprocessed PNG paths. Returns only ``grounded_report`` and ``findings``.
        Requires the optional ``report`` dependencies and approved Llama access
        or a local base-model directory. All inference stays on this machine.
        """
        from ._reports import report_gen
        return report_gen(self, img, base_model_path=base_model_path, seed=seed,
                          max_new_tokens=max_new_tokens,
                          grounding_max_new_tokens=grounding_max_new_tokens)

    def diagnosis(self, img, dataset, *, methods=None, strict=False):
        """Return JSON-serializable, uncalibrated per-image model probabilities.

        Unreleased tasks are null with a reason in ``unavailable``. ``strict=True``
        raises before any model download when a requested task is unavailable.
        Actual inference/download errors always raise; they are never hidden as null.
        Binary tasks return floats; density returns an A/B/C/D probability dict.
        RSNA invasive cancer is conditional on cancer-positive cases, not a
        population invasive-cancer risk estimate; see metadata task definitions.
        """
        name = _dataset(dataset)
        selected = _methods(methods)
        missing = {
            method: {task: _missing_reason(name, task, method) for task in TASKS[name]
                     if _missing_reason(name, task, method) is not None}
            for method in selected
        }
        missing = {method: entries for method, entries in missing.items() if entries}
        if missing and strict:
            raise MissingModelError(json.dumps(missing, sort_keys=True))
        if missing:
            warnings.warn("Some requested models are not released; inspect output['unavailable']. "
                          "Missing probabilities are None, not zero.", UserWarning, stacklevel=2)
        import torch
        from ._inputs import inference_spec, inference_tensor, load_image
        image, info = load_image(img, name)
        result = {"dataset": name, **{method: {task: None for task in TASKS[name]} for method in selected}}
        checkpoints = {method: {} for method in selected}
        with torch.inference_mode():
            zero_shot_tasks = [task for task in TASKS[name]
                               if task not in missing.get("zero_shot", {})]
            if "zero_shot" in selected and zero_shot_tasks:
                model = self._get_clip()
                tensor = inference_tensor(image, branch="clip", dataset=name).to(self.device)
                encoded = model.encode_image_normalized(tensor)
                encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
                for task in zero_shot_tasks:
                    # Official evaluator: cosine -> binary softmax, no learned temperature.
                    logits = encoded @ self._prototype(task).T
                    result["zero_shot"][task] = _probability(logits.softmax(dim=1)[0, 1])
                    checkpoints["zero_shot"][task] = provenance(ENCODER)
            for method in (m for m in selected if m != "zero_shot"):
                available = [task for task in TASKS[name] if task not in missing.get(method, {})]
                if not available:
                    continue
                tensor = inference_tensor(image, branch="classifier", dataset=name).to(self.device)
                for task in available:
                    probability = self._get_classifier(name, task, method).probabilities(tensor)
                    if task == "density":
                        result[method][task] = _density_probabilities(probability)
                    else:
                        if probability.numel() != 1:
                            raise RuntimeError("Expected one probability for a single-image classifier")
                        result[method][task] = _probability(probability.reshape(-1)[0])
                    checkpoints[method][task] = provenance(CLASSIFIERS[(name, task, method)])
        result["unavailable"] = missing
        result["metadata"] = {
            "schema_version": "1.0", "research_only": True,
            "probabilities_calibrated": False, "prediction_level": "single_image",
            "input": info, "repo_id": REPO_ID, "revision": REVISION,
            "zero_shot_scoring": "softmax(cosine(image, mean_normalized_prompts)); temperature=1",
            "inference": {method: inference_spec("clip" if method == "zero_shot" else "classifier", name)
                          for method in selected},
            "checkpoints": checkpoints,
            "task_definitions": {task: TASK_DEFINITIONS[(name, task)] for task in TASKS[name]},
        }
        if "density" in TASKS[name]:
            result["metadata"]["density_class_order"] = list(DENSITY_CLASSES)
            result["metadata"]["density_scoring"] = "softmax over mutually exclusive A/B/C/D classes"
        if name == "vindr":
            result["metadata"]["abnormality_definition"] = (
                "BI-RADS 4/5 versus 1/2; BI-RADS 3 excluded in evaluation. "
                "The historical zero-shot surrogate uses cancer/no cancer prompts, not a broad any-finding prompt."
            )
        return result


def _embedding_array(values, expected_rows):
    import torch
    if (not isinstance(values, torch.Tensor) or values.ndim != 2
            or values.shape[0] != expected_rows or values.shape[1] == 0
            or values.is_complex() or values.dtype == torch.bool):
        raise RuntimeError("Encoder produced an invalid embedding shape or dtype")
    values = values.detach().float().cpu()
    lengths = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if (not torch.isfinite(values).all() or not torch.isfinite(lengths).all()
            or bool((lengths <= 0).any())):
        raise RuntimeError("Encoder produced nonfinite or zero embeddings")
    return (values / lengths).numpy()


def _probability(value):
    import math
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RuntimeError("Model produced an invalid probability")
    return value


def _density_probabilities(values):
    import math
    import torch
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != (1, len(DENSITY_CLASSES)):
        raise RuntimeError("Expected [1, 4] A/B/C/D probabilities for a single-image density classifier")
    probabilities = [_probability(value) for value in values[0]]
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("Density probabilities must sum to one")
    return dict(zip(DENSITY_CLASSES, probabilities))


@lru_cache(maxsize=2)
def _client(device=None, cache_dir=None, model_dir=None, local_files_only=False):
    return MammoFM(device=device, cache_dir=cache_dir, model_dir=model_dir, local_files_only=local_files_only)


def preprocess(img):
    """Return one DICOM as an in-memory, preprocessed grayscale Pillow image.

    ``img`` is a local DICOM path (string or path-like). Uses the unified
    converter's defaults: five-pixel border removal, MONOCHROME1 correction,
    min-max scaling, breast-cropping heuristic, and INTER_AREA resizing to
    1520 pixels high by 912 wide. The result is 8-bit grayscale (mode ``L``),
    without model-input normalization or DICOM metadata. Burned-in text is
    not anonymized. PNG, color and multiframe inputs are rejected.

    Runs on CPU without loading models, downloading weights or writing files.
    Call ``image.save('preprocessed.png')`` on the result to save it explicitly.
    """
    from ._inputs import load_image

    image, details = load_image(img)
    if details['input_format'] != 'dicom':
        raise ValueError('preprocess expects a DICOM source, not an already-preprocessed PNG')
    return image.convert('L')


def diagnosis(img, dataset, *, methods=None, strict=False, device=None, cache_dir=None,
              model_dir=None, local_files_only=False):
    """Convenience wrapper for :meth:`MammoFM.diagnosis`. Images are processed locally."""
    return _client(device, cache_dir, model_dir, local_files_only).diagnosis(img, dataset, methods=methods, strict=strict)


def features(img, dataset=None, *, device=None, cache_dir=None, model_dir=None, local_files_only=False):
    """Convenience wrapper for :meth:`MammoFM.features`; returns full NumPy arrays."""
    return _client(device, cache_dir, model_dir, local_files_only).features(img, dataset)


def text_embeddings(prompts, *, device=None, cache_dir=None, model_dir=None, local_files_only=False):
    """Encode custom prompts in the joint CLIP space; returns a float32 NumPy array."""
    return _client(device, cache_dir, model_dir, local_files_only).text_embeddings(prompts)


def image_embeddings(img, dataset=None, *, device=None, cache_dir=None, model_dir=None,
                     local_files_only=False):
    """Encode one DICOM/PNG in the joint CLIP space; returns a float32 NumPy array."""
    return _client(device, cache_dir, model_dir, local_files_only).image_embeddings(img, dataset)


def prognosis(img, *, device=None, cache_dir=None, model_dir=None,
              local_files_only=False):
    """Return five direct, uncalibrated risk probabilities for a four-view exam.

    See :meth:`MammoFM.prognosis` for input requirements and probability limits.
    CUDA is selected automatically when available; pass ``device='cpu'`` to
    override. Images stay local, and verified model downloads are cached.
    """
    return _client(device, cache_dir, model_dir, local_files_only).prognosis(img)


def report_gen(img, *, device=None, cache_dir=None, model_dir=None,
               local_files_only=False, base_model_path=None, seed=0,
               max_new_tokens=512, grounding_max_new_tokens=320):
    """Return a grounded report and findings for a named four-view DICOM/PNG dict."""
    return _client(device, cache_dir, model_dir, local_files_only).report_gen(
        img, base_model_path=base_model_path, seed=seed,
        max_new_tokens=max_new_tokens, grounding_max_new_tokens=grounding_max_new_tokens,
    )


def clear_cache():
    """Release convenience-client references; downloaded model files are not deleted."""
    _client.cache_clear()
