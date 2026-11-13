"""Strict four-view Mammo-FM five-horizon risk inference."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._model import _image_encoder, _read_checkpoint, _strict_load, load_encoder
from ._registry import BU_RISK_ENCODER


VIEW_ORDER = ("LMLO", "LCC", "RMLO", "RCC")
TIME_SEQUENCE = (0, 0, 0, 0)
VIEW_SEQUENCE = (0, 1, 0, 1)  # MLO=0, CC=1
SIDE_SEQUENCE = (0, 0, 1, 1)  # left=0, right=1


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.25):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.dim_per_head = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, values):
        batch, length, hidden = values.shape
        query = self.query(values).view(batch, length, self.num_heads, self.dim_per_head)
        key = self.key(values).view(batch, length, self.num_heads, self.dim_per_head)
        value = self.value(values).view(batch, length, self.num_heads, self.dim_per_head)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) /
                            math.sqrt(self.dim_per_head), dim=-1)
        context = torch.matmul(self.dropout(weights), value)
        context = context.transpose(1, 2).contiguous().view(batch, length, hidden)
        return self.out_proj(context)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.25):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layernorm2 = nn.LayerNorm(hidden_dim)

    def forward(self, values):
        values = self.layernorm1(values + self.multihead_attention(values))
        return self.layernorm2(values + self.fc(values))


class MetadataTransformer(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=3, num_heads=8, dropout=0.25):
        super().__init__()
        self.time_embed = nn.Embedding(11, 32, padding_idx=0)
        self.view_embed = nn.Embedding(3, 32, padding_idx=0)
        self.side_embed = nn.Embedding(3, 32, padding_idx=0)
        self.embed_fc = nn.Linear(96, hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, values, time_seq, view_seq, side_seq):
        metadata = torch.cat((self.time_embed(time_seq), self.view_embed(view_seq),
                              self.side_embed(side_seq)), dim=-1)
        values = values + self.embed_fc(metadata)
        for layer in self.layers:
            values = layer(values)
        return values


class FourViewTransformer(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=3, num_heads=8,
                 dropout=0.25):
        super().__init__()
        self.projection_layer = nn.Linear(input_dim, hidden_dim)
        self.mask_embedding = nn.Embedding(2, input_dim, padding_idx=1)
        self.kept_images_vec = nn.Parameter(torch.ones(1, 4, 1), requires_grad=False)
        self.transformer = MetadataTransformer(hidden_dim, num_layers, num_heads, dropout)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)

    def mask_input(self, image_features, view_seq):
        batch, images, _ = image_features.shape
        is_masked = torch.bernoulli(
            self.kept_images_vec.expand(batch, images, 1) * 0.0).to(image_features.device)
        is_masked = is_masked * (view_seq < 2).unsqueeze(-1).float()
        is_kept = 1.0 - is_masked
        return image_features * is_kept + self.mask_embedding(is_kept.squeeze(-1).long())

    def forward(self, image_features, time_seq, view_seq, side_seq):
        values = self.projection_layer(self.mask_input(image_features, view_seq))
        values = self.transformer(values, time_seq, view_seq, side_seq).mean(dim=1)
        return self.fc(self.dropout(self.relu(values)))


class FiveYearRiskModel(nn.Module):
    """Full EfficientNet-B5 encoder plus the historical four-view risk head."""

    def __init__(self):
        super().__init__()
        self.image_encoder = _image_encoder()
        self.aggregator = FourViewTransformer(input_dim=2048)

    def forward(self, images, time_seq, view_seq, side_seq):
        if not isinstance(images, torch.Tensor) or images.ndim != 5 or images.shape[1:3] != (4, 3):
            raise ValueError("Expected normalized images shaped B x 4 x 3 x H x W")
        batch = images.shape[0]
        flat = images.reshape(batch * 4, *images.shape[2:])
        pooled, _ = self.image_encoder({"image": flat, "breast_clip_train_mode": True})
        features = pooled.flatten(start_dim=1).reshape(batch, 4, -1)
        return self.aggregator(features, time_seq.long(), view_seq.long(), side_seq.long())


def _compatible_state(state):
    """Verify and remove the historical duplicate encoder registration."""
    state = dict(state)
    prefix = "aggregator.image_encoder."
    for duplicate_name in [name for name in state if name.startswith(prefix)]:
        canonical_name = "image_encoder." + duplicate_name[len(prefix):]
        if canonical_name not in state:
            raise ValueError(f"Risk checkpoint is missing canonical tensor {canonical_name!r}")
        duplicate, canonical = state[duplicate_name], state[canonical_name]
        if duplicate.shape != canonical.shape or not torch.equal(duplicate, canonical):
            raise ValueError(f"Risk checkpoint has conflicting tensor {duplicate_name!r}")
        del state[duplicate_name]
    return state


def load_risk_model(checkpoint, device="cpu", *, encoder_checkpoint=None):
    """Strictly load either a complete legacy model or a released head."""
    state, config = _read_checkpoint(checkpoint)
    model = FiveYearRiskModel()
    if encoder_checkpoint is None:
        _strict_load(model, _compatible_state(state))
        return model.to(device).eval()

    if config.get("release_format") != "inference_only_head":
        raise ValueError("BU risk head has an unsupported release format")
    if config.get("encoder_frozen") is not True:
        raise ValueError("BU risk head must identify its encoder as frozen")
    if config.get("encoder_source_sha256") != BU_RISK_ENCODER["sha256"]:
        raise ValueError("BU risk head requires an unexpected image encoder")
    if config.get("view_order") != list(VIEW_ORDER):
        raise ValueError("BU risk head has an incompatible view order")
    for key, expected in (
        ("time_seq", list(TIME_SEQUENCE)),
        ("view_seq", list(VIEW_SEQUENCE)),
        ("side_seq", list(SIDE_SEQUENCE)),
    ):
        if config.get(key) != expected:
            raise ValueError(f"BU risk head has incompatible {key}")
    if config.get("prediction_calibrated") is not False:
        raise ValueError("BU risk head must identify its outputs as uncalibrated")

    encoder = load_encoder(encoder_checkpoint, device="cpu")
    model.image_encoder = encoder.image_encoder
    _strict_load(model.aggregator, state)
    return model.to(device).eval()


@torch.inference_mode()
def predict_risk_logits(model, images):
    parameter = next(model.parameters())
    images = images.to(device=parameter.device, dtype=parameter.dtype)
    batch = images.shape[0]
    sequences = [torch.tensor(values, device=parameter.device).expand(batch, -1)
                 for values in (TIME_SEQUENCE, VIEW_SEQUENCE, SIDE_SEQUENCE)]
    values = model(images, *sequences)
    if values.shape != (batch, 5) or not torch.isfinite(values).all():
        raise RuntimeError("Risk model produced invalid five-horizon output")
    return values


@torch.inference_mode()
def predict_risks(model, images):
    values = predict_risk_logits(model, images).sigmoid()
    if bool(((values < 0) | (values > 1)).any()):
        raise RuntimeError("Risk model produced a probability outside [0, 1]")
    return values


__all__ = [
    "FiveYearRiskModel",
    "VIEW_ORDER",
    "load_risk_model",
    "predict_risk_logits",
    "predict_risks",
]
