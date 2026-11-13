"""Local four-view report generation with the released image-level grounding models."""

from collections.abc import Mapping
from contextlib import contextmanager
import gc
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
from shutil import copyfile
from tempfile import TemporaryDirectory

from ._assets import provenance
from ._errors import CheckpointError
from ._report_registry import (
    BASE_LLM_REPO,
    BASE_LLM_REVISION,
    GROUNDING_MODELS,
    REPORT_ENCODER,
    REPORT_STAGE2_FILES,
)

VIEW_ORDER = ("lmlo", "lcc", "rmlo", "rcc")
_FEATURE_SHAPE = (1, 2048, 48, 29)
_PREDICTION_KEYS = {
    "mass": "mass_pred",
    "suspicious_calcification": "suspicious calcification_pred",
    "asymmetry": "asymmetry_pred",
}


def _view_paths(img):
    if not isinstance(img, Mapping):
        raise TypeError("img must be a dictionary with exactly lmlo, lcc, rmlo and rcc image paths")
    normalized = {}
    for name, path in img.items():
        if not isinstance(name, str) or name.lower() not in VIEW_ORDER:
            raise ValueError("Image view names must be exactly lmlo, lcc, rmlo and rcc (case-insensitive)")
        name = name.lower()
        if name in normalized:
            raise ValueError("Duplicate image view: {}".format(name))
        normalized[name] = path
    if set(normalized) != set(VIEW_ORDER):
        raise ValueError("Report generation requires all four views: lmlo, lcc, rmlo and rcc")
    return {name: normalized[name] for name in VIEW_ORDER}


def _load_views(img):
    import numpy as np
    import torch
    from ._inputs import inference_tensor, load_image

    tensors = {}
    for view, path in _view_paths(img).items():
        # None selects the unified general DICOM route, never a dataset window.
        # Existing PNG contrast/crop is preserved by load_image.
        image, _ = load_image(path, None)
        pixels = np.asarray(image)
        if not pixels.size or not np.isfinite(pixels).all() or pixels.max() == pixels.min():
            raise ValueError("{} image is empty, nonfinite or constant".format(view))
        tensor = inference_tensor(image, branch="features", dataset=None)
        if not torch.isfinite(tensor).all() or tensor.max() == tensor.min():
            raise ValueError("{} image has no usable intensity variation after preprocessing".format(view))
        tensors[view] = tensor
    return tensors


def _require_dependencies():
    missing = [name for name in ("peft", "accelerate", "einops")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError("Report generation requires {}. Install mammofm[report].".format(
            ", ".join(missing)))


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 4096:
        raise ValueError("{} must be an integer between 1 and 4096".format(name))


def _validate_base_directory(directory):
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise CheckpointError("base_model_path must be a local Llama-3.1-8B-Instruct model directory")
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CheckpointError("The base language model needs a readable config.json") from error
    expected = {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
                "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256}
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected.items()):
        raise CheckpointError("The released report adapter requires the original Llama-3.1-8B-Instruct architecture")
    if config.get("auto_map") or config.get("quantization_config"):
        raise CheckpointError("Use the original non-quantized base model without custom remote code")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (directory / name).is_file():
            raise CheckpointError("The base language model is missing {}".format(name))
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CheckpointError("Invalid base-model safetensors index") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError("Empty base-model safetensors index")
        for filename in weight_map.values():
            if not isinstance(filename, str):
                raise CheckpointError("Unsafe base-model shard filename")
            relative = PurePosixPath(filename)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in filename
                    or relative.suffix != ".safetensors"):
                raise CheckpointError("Unsafe base-model shard filename")
            if not (directory / filename).is_file():
                raise CheckpointError("The base language model has missing safetensors shards")
    elif not (directory / "model.safetensors").is_file():
        raise CheckpointError("The base language model needs its complete safetensors weights")
    return directory


def _resolve_base_model(store, base_model_path):
    if base_model_path is not None:
        return _validate_base_directory(base_model_path), "local_override"
    from huggingface_hub import snapshot_download
    try:
        directory = snapshot_download(
            repo_id=BASE_LLM_REPO, revision=BASE_LLM_REVISION,
            cache_dir=store.cache_dir, local_files_only=store.local_files_only,
            allow_patterns=["config.json", "generation_config.json", "tokenizer.json",
                            "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                            "model.safetensors", "model.safetensors.index.json", "model-*.safetensors"],
        )
    except Exception as error:
        raise CheckpointError(
            "Could not obtain the gated Llama-3.1-8B-Instruct base model. Accept its license "
            "on Hugging Face and log in with the HF CLI, or provide base_model_path to a "
            "complete local model. local_files_only=True disables downloads."
        ) from error
    return _validate_base_directory(directory), "pinned_huggingface_snapshot"


@contextmanager
def _stage2_directory(paths):
    # A model_dir may be partially populated. Stage only the verified public
    # assets, with links where possible, rather than copying image data or keys.
    with TemporaryDirectory(prefix="mammofm-report-") as temporary:
        directory = Path(temporary)
        for name, source in paths.items():
            if not isinstance(name, str) or Path(name).name != name or "\\" in name:
                raise CheckpointError("Unsafe Stage-2 asset filename")
            source = Path(source).resolve()
            destination = directory / name
            try:
                destination.symlink_to(source)
            except (NotImplementedError, OSError):
                try:
                    os.link(source, destination)
                except OSError:
                    copyfile(source, destination)
        yield directory


def _release_memory(device):
    import torch
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _visual_embeddings(tensors, checkpoint, device):
    import torch
    from ._model import load_encoder
    model = None
    try:
        model = load_encoder(checkpoint, device=device)
        features = []
        with torch.inference_mode():
            for view in VIEW_ORDER:
                result = model.features(tensors[view].to(device))
                raw = result["before_pooling"]
                if tuple(raw.shape) != _FEATURE_SHAPE or not torch.isfinite(raw).all():
                    raise RuntimeError("Invalid report encoder spatial features")
                features.append(raw.detach().to(device="cpu", dtype=torch.float32))
                del raw, result
        return torch.stack(features, dim=1).unsqueeze(2)
    finally:
        del model
        _release_memory(device)


def _grounding_findings(tensors, checkpoints, device):
    import torch
    from ._model import load_classifier

    views = {view: {} for view in VIEW_ORDER}
    for task, asset in GROUNDING_MODELS.items():
        model = None
        try:
            model = load_classifier(checkpoints[task], device=device)
            with torch.inference_mode():
                for view in VIEW_ORDER:
                    output = model.probabilities(tensors[view].to(device))
                    if output.numel() != 1:
                        raise RuntimeError("Expected a single grounding probability per image")
                    probability = float(output.reshape(-1)[0])
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise RuntimeError("Grounding classifier produced an invalid probability")
                    views[view][task] = {"probability": probability,
                                         "positive": probability >= asset["threshold"]}
                    del output
        finally:
            del model
            _release_memory(device)
    side = {
        name: {task: any(views[view][task]["positive"] for view in VIEW_ORDER if view.startswith(prefix))
               for task in GROUNDING_MODELS}
        for name, prefix in (("left", "l"), ("right", "r"))
    }
    findings = {
        "views": views, **side,
        "exam": {task: side["left"][task] or side["right"][task] for task in GROUNDING_MODELS},
        "bilateral": {task: side["left"][task] and side["right"][task] for task in GROUNDING_MODELS},
    }
    records = [
        {"image_laterality": view[0].upper(), "image_view_type": view[1:].upper(),
         **{_PREDICTION_KEYS[task]: int(views[view][task]["positive"]) for task in GROUNDING_MODELS}}
        for view in VIEW_ORDER
    ]
    return findings, records


def report_gen(client, img, *, base_model_path=None, seed=0,
               max_new_tokens=512, grounding_max_new_tokens=320):
    """Return only the grounded report and classifier findings for four named views.

    Stages 1/2 are training stages: inference uses the final Stage-2 projector
    and LoRA weights, then the separate Stage-3 grounding refiner. The epoch-3
    image encoder is used internally; neither features nor the intermediate
    preliminary report are returned. All image/report processing stays local.
    """
    _positive_integer(max_new_tokens, "max_new_tokens")
    _positive_integer(grounding_max_new_tokens, "grounding_max_new_tokens")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    tensors = _load_views(img)
    _require_dependencies()
    from ._report_model import generate_grounded, generate_preliminary

    device = client.device
    encoder = client.store.resolve(REPORT_ENCODER)
    stage2 = {name: client.store.resolve(asset) for name, asset in REPORT_STAGE2_FILES.items()}
    grounding = {task: client.store.resolve(asset) for task, asset in GROUNDING_MODELS.items()}
    base, base_source = _resolve_base_model(client.store, base_model_path)
    client.clear_cache()
    _release_memory(device)
    embeddings = _visual_embeddings(tensors, encoder, device)
    findings, records = _grounding_findings(tensors, grounding, device)
    del tensors
    with _stage2_directory(stage2) as checkpoint_dir:
        preliminary = generate_preliminary(
            embeddings, checkpoint_dir, base, device,
            max_new_tokens=max_new_tokens, seed=seed,
        )
    del embeddings
    grounded = generate_grounded(
        preliminary, records, base, device,
        max_new_tokens=grounding_max_new_tokens, seed=seed,
    )
    del preliminary
    if not isinstance(grounded, str) or not grounded.strip():
        raise RuntimeError("Report refinement returned an empty grounded report")
    findings["metadata"] = {
        "research_only": True, "probabilities_calibrated": False,
        "prediction_level": "image", "view_order": list(VIEW_ORDER),
        "checkpoints": {task: provenance(asset) for task, asset in GROUNDING_MODELS.items()},
        "visual_encoder": provenance(REPORT_ENCODER),
        "stage2": {name: provenance(asset) for name, asset in REPORT_STAGE2_FILES.items()},
        "base_language_model": {"repo_id": BASE_LLM_REPO, "revision": BASE_LLM_REVISION,
                                "source": base_source, "local_override_weights_verified": False},
        "generation": {"seed": seed, "preliminary_max_new_tokens": max_new_tokens,
                       "grounding_max_new_tokens": grounding_max_new_tokens,
                       "preliminary_do_sample": False, "grounding_do_sample": True,
                       "grounding_temperature": 0.3, "grounding_top_p": 0.9},
    }
    return {"grounded_report": grounded.strip(), "findings": findings}
