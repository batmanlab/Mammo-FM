"""Strict, offline inference backends for the released Mammo-FM models.

Checkpoint resolution and integrity verification belong to the public API.
This module never downloads weights, imports the research checkout, or falls
back to unrestricted pickle loading.
"""

from __future__ import annotations

import _codecs
import collections
from collections.abc import Mapping
from pathlib import Path
import re
import typing

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._vendor.efficientnet import EfficientNet


class _ConfigMetadata:
    """Inert replacement for historical OmegaConf objects during safe loading."""

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise ValueError("Invalid serialized configuration metadata")
        self.__dict__.update(state)


def _plain_config(value):
    if isinstance(value, _ConfigMetadata):
        if "_content" in value.__dict__:
            return _plain_config(value.__dict__["_content"])
        if "_val" in value.__dict__:
            return _plain_config(value.__dict__["_val"])
        return None
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def _safe_globals():
    # Only the concrete metadata types present in the historical releases are
    # accepted. OmegaConf is not imported and no resolvers are executed.
    metadata_names = (
        "omegaconf.base.ContainerMetadata", "omegaconf.base.Metadata",
        "omegaconf.dictconfig.DictConfig", "omegaconf.listconfig.ListConfig",
        "omegaconf.nodes.AnyNode",
    )
    allowed = [(_ConfigMetadata, name) for name in metadata_names]
    allowed += [dict, list, int, collections.defaultdict, typing.Any,
                np.ndarray, np.dtype, _codecs.encode]
    core = np._core if hasattr(np, "_core") else np.core
    for namespace in ("numpy.core.multiarray", "numpy._core.multiarray"):
        allowed.extend([
            (core.multiarray._reconstruct, namespace + "._reconstruct"),
            (core.multiarray.scalar, namespace + ".scalar"),
        ])
    allowed.extend(type(np.dtype(kind)) for kind in
                   ("float64", "float32", "int64", "int32"))
    return allowed


def _read_checkpoint(path):
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < (2, 6):
        raise RuntimeError("Mammo-FM checkpoint loading requires PyTorch >= 2.6")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with torch.serialization.safe_globals(_safe_globals()):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a model state dictionary")
    raw = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Checkpoint has no model tensors")
    state = {}
    for name, tensor in raw.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Model state must map string parameter names to tensors")
        while name.startswith(("module.", "_orig_mod.")):
            name = name.split(".", 1)[1]
        if name in state:
            raise ValueError(f"Duplicate parameter after prefix removal: {name}")
        state[name] = tensor
    config = _plain_config(checkpoint.get("config", {}))
    if not isinstance(config, dict):
        raise ValueError("Checkpoint configuration must be a mapping")
    return state, config


def _strict_load(model, state):
    # PyTorch can silently synthesize absent batch-norm counters; explicitly
    # compare keys first so missing encoder or classifier tensors never pass.
    expected, actual = set(model.state_dict()), set(state)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Checkpoint architecture mismatch: missing={missing}; unexpected={extra}")
    model.load_state_dict(state, strict=True)


def _check_encoder_config(config):
    if not config:
        return
    model = config.get("model", {})
    image = model.get("image_encoder", {})
    if (image.get("source") != "cnn"
            or image.get("name") != "tf_efficientnet_b5_ns-detect"
            or image.get("model_type") != "cnn"):
        raise ValueError("Only the released Mammo-FM EfficientNet-B5 image encoder is supported")


def _image_encoder():
    # Preserve the original static SAME padding based on B5's nominal 456px
    # architecture. Replacing this with dynamic padding changes model outputs.
    model = EfficientNet.from_name("efficientnet-b5", num_classes=1)
    model.out_dim = 2048
    return model


class Encoder(nn.Module):
    """2048-channel spatial and unprojected pooled image representations."""

    def __init__(self):
        super().__init__()
        self.image_encoder = _image_encoder()

    def _input(self, image):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected a normalized B x 3 x H x W image tensor")
        parameter = next(self.image_encoder.parameters())
        return image.to(device=parameter.device, dtype=parameter.dtype)

    @torch.inference_mode()
    def features(self, image):
        raw = self.image_encoder.extract_features(self._input(image))
        pooled = self.image_encoder._avg_pooling(raw).flatten(start_dim=1)
        return {"before_pooling": raw, "after_pooling": pooled}

    @torch.inference_mode()
    def forward(self, image):
        return self.features(image)["after_pooling"]


class _LinearClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.classification_head = nn.Linear(2048, num_classes)

    def forward(self, features):
        return self.classification_head(features)


class Classifier(Encoder):
    def __init__(self, num_classes=1):
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes not in (1, 4):
            raise ValueError("Classifier must have one binary output or four density outputs")
        super().__init__()
        self.num_classes = num_classes
        self.classifier = _LinearClassifier(num_classes)

    @torch.inference_mode()
    def forward(self, image):
        return self.classifier(self.features(image)["after_pooling"])

    @torch.inference_mode()
    def probabilities(self, image):
        logits = self(image)
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise RuntimeError("Classifier produced an unexpected logit shape")
        if self.num_classes == 4:
            return logits.softmax(dim=1)
        return logits.sigmoid().squeeze(-1)


class _LinearProjection(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, inputs):
        return self.projection(inputs)


class _MLPProjection(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, inputs):
        projected = self.projection(inputs)
        return self.layer_norm(self.dropout(self.fc(self.gelu(projected))) + projected)


def _projection(embedding_dim, config):
    if config["name"] == "linear":
        return _LinearProjection(embedding_dim, int(config["proj_dim"]))
    if config["name"] == "mlp":
        return _MLPProjection(embedding_dim, int(config["proj_dim"]), float(config["dropout"]))
    raise ValueError("Unsupported CLIP projection head")


class CLIP(Encoder):
    def __init__(self, config, text_config_dir):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        model_config = config["model"]
        self.text_pooling = model_config["text_encoder"]["pooling"]
        if self.text_pooling not in {"eos", "bos", "mean"}:
            raise ValueError("Unsupported text pooling")
        self.text_max_length = int(config["base"]["text_max_length"])
        text_config = AutoConfig.from_pretrained(
            str(text_config_dir), local_files_only=True, trust_remote_code=False)
        if text_config.model_type != "modernbert":
            raise ValueError("The released Mammo-FM text encoder requires ModernBERT configuration")
        # Disable compilation/optional FlashAttention dependencies for portable
        # inference; eager attention preserves the model's attention equations.
        text_config.reference_compile = False
        self.text_encoder = AutoModel.from_config(
            text_config, trust_remote_code=False, attn_implementation="eager")
        self.image_projection = _projection(2048, model_config["projection_head"])
        self.text_projection = _projection(text_config.hidden_size, model_config["projection_head"])
        self.logit_scale = nn.Parameter(torch.zeros(()))

    @torch.inference_mode()
    def encode_image_normalized(self, image):
        pooled = self.features(image)["after_pooling"]
        return F.normalize(self.image_projection(pooled), dim=1)

    @torch.inference_mode()
    def encode_text_normalized(self, tokens):
        device = next(self.text_encoder.parameters()).device
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        if ids.ndim != 2 or mask.shape != ids.shape or not (mask.sum(dim=1) > 0).all():
            raise ValueError("Expected nonempty B x sequence token IDs and attention masks")
        output = self.text_encoder(
            input_ids=ids, attention_mask=mask, output_hidden_states=True, return_dict=True)
        # The training/evaluation implementation selects hidden_states[-1],
        # which ModernBERT exposes BEFORE its final layer normalization. Using
        # last_hidden_state here would silently change zero-shot predictions.
        hidden = output.hidden_states[-1]
        if self.text_pooling == "eos":
            features = hidden[torch.arange(hidden.shape[0], device=device), mask.sum(dim=-1) - 1]
        elif self.text_pooling == "bos":
            features = hidden[:, 0]
        else:
            expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
            features = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)
        return F.normalize(self.text_projection(features), dim=1)


def load_encoder(checkpoint, device="cpu"):
    state, config = _read_checkpoint(checkpoint)
    _check_encoder_config(config)
    if any(name.startswith("image_encoder.") for name in state):
        state = {name[len("image_encoder."):]: value for name, value in state.items()
                 if name.startswith("image_encoder.")}
    model = Encoder()
    _strict_load(model.image_encoder, state)
    return model.to(device).eval()


def load_classifier(checkpoint, device="cpu", *, num_classes=None):
    state, config = _read_checkpoint(checkpoint)
    _check_encoder_config(config)
    weight = state.get("classifier.classification_head.weight")
    bias = state.get("classifier.classification_head.bias")
    if (weight is None or weight.ndim != 2 or weight.shape[1] != 2048
            or weight.shape[0] not in (1, 4) or bias is None
            or tuple(bias.shape) != (weight.shape[0],)):
        raise ValueError("Checkpoint classifier head must have shape [1, 2048] or [4, 2048] with a matching bias")
    inferred_classes = int(weight.shape[0])
    if num_classes is not None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes not in (1, 4):
            raise ValueError("Expected classifier output count must be 1 or 4")
        if inferred_classes != num_classes:
            raise ValueError(f"Checkpoint has {inferred_classes} classifier outputs; expected {num_classes}")
    model = Classifier(inferred_classes)
    _strict_load(model, state)
    return model.to(device).eval()


def load_clip(checkpoint, text_config_dir, device="cpu"):
    state, config = _read_checkpoint(checkpoint)
    _check_encoder_config(config)
    model = CLIP(config, text_config_dir)
    _strict_load(model, state)
    return model.to(device).eval()
