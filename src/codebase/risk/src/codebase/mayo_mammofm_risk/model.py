"""Mammo-FM image encoder plus the five-head transformer used in the main code."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from breastclip.model.modules import EfficientNet, load_image_encoder


EMBEDDING_DIM = 96
MAX_TIME = 10
MAX_VIEWS = 2
MAX_SIDES = 2


def compatible_model_state(state):
    """Normalize historical teacher/student checkpoints to the shared model layout.

    The original risk wrapper registered the same image encoder twice: once as
    ``image_encoder`` and again as ``aggregator.image_encoder``.  PyTorch
    therefore serialized a duplicate set of tensors.  The cleaned model keeps
    one encoder registration.  Remove only that verified duplicate namespace;
    all remaining keys are still loaded strictly by the caller.
    """
    normalized = dict(state)
    duplicate_prefix = "aggregator.image_encoder."
    duplicate_keys = [key for key in normalized if key.startswith(duplicate_prefix)]
    for duplicate_key in duplicate_keys:
        suffix = duplicate_key[len(duplicate_prefix):]
        canonical_key = "image_encoder." + suffix
        if canonical_key not in normalized:
            raise ValueError(
                f"Historical checkpoint is missing canonical tensor {canonical_key!r}"
            )
        duplicate = normalized[duplicate_key]
        canonical = normalized[canonical_key]
        if duplicate.shape != canonical.shape or not torch.equal(duplicate, canonical):
            raise ValueError(
                f"Historical checkpoint contains conflicting encoder tensor {duplicate_key!r}"
            )
        del normalized[duplicate_key]
    return normalized


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

    def forward(self, x):
        batch_size, seq_length, hidden_dim = x.shape
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_per_head)
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_dim)
        return self.out_proj(context)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.25):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layernorm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.layernorm1(x + self.multihead_attention(x))
        return self.layernorm2(x + self.fc(x))


class MetadataTransformer(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=3, num_heads=8, dropout=0.25):
        super().__init__()
        embedding_dim_per_type = EMBEDDING_DIM // 3
        self.time_embed = nn.Embedding(MAX_TIME + 1, embedding_dim_per_type, padding_idx=0)
        self.view_embed = nn.Embedding(MAX_VIEWS + 1, embedding_dim_per_type, padding_idx=0)
        self.side_embed = nn.Embedding(MAX_SIDES + 1, embedding_dim_per_type, padding_idx=0)
        self.embed_fc = nn.Linear(EMBEDDING_DIM, hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, time_seq, view_seq, side_seq):
        metadata_embedding = torch.cat(
            [self.time_embed(time_seq), self.view_embed(view_seq), self.side_embed(side_seq)],
            dim=-1,
        )
        x = x + self.embed_fc(metadata_embedding)
        for layer in self.layers:
            x = layer(x)
        return x


class FourViewTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, num_heads=8, dropout=0.25):
        super().__init__()
        self.projection_layer = nn.Linear(input_dim, hidden_dim)
        # These two members are retained from the main implementation. With
        # mask_probability=0 they are an identity operation, but preserving the
        # layout keeps the aggregator architecture/checkpoints recognizable.
        self.mask_embedding = nn.Embedding(2, input_dim, padding_idx=1)
        self.kept_images_vec = nn.Parameter(torch.ones(1, 4, 1), requires_grad=False)
        self.transformer = MetadataTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)

    def mask_input(self, image_features, view_seq):
        batch_size, num_images, _ = image_features.shape
        mask_probability = 0.0
        is_masked = torch.bernoulli(
            self.kept_images_vec.expand(batch_size, num_images, 1) * mask_probability
        ).to(image_features.device)
        is_masked = is_masked * (view_seq < MAX_VIEWS).unsqueeze(-1).float()
        is_kept = 1.0 - is_masked
        output = image_features * is_kept
        output = output + self.mask_embedding(is_kept.squeeze(-1).long())
        return output

    def forward(self, image_features, time_seq, view_seq, side_seq):
        x = self.mask_input(image_features, view_seq)
        x = self.projection_layer(x)
        x = self.transformer(x, time_seq, view_seq, side_seq)
        pooled = x.mean(dim=1)
        return self.fc(self.dropout(self.relu(pooled)))


class MammoFMFiveYearRiskModel(nn.Module):
    """Frozen Mayo Mammo-FM encoder with a trainable transformer and five risk heads."""

    def __init__(self, mammo_checkpoint, freeze_encoder=True):
        super().__init__()
        try:
            checkpoint = torch.load(mammo_checkpoint, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only was added
            checkpoint = torch.load(mammo_checkpoint, map_location="cpu")

        encoder_config = checkpoint["config"]["model"]["image_encoder"]
        # The checkpoint immediately supplies every encoder parameter.  The
        # Breast-CLIP detector factory ignores its ``pretrained`` flag and
        # calls ``from_pretrained`` unconditionally, which causes an
        # unnecessary network/cache access.  Construct those two supported
        # detector architectures without weights, then load the checkpoint
        # strictly below.
        encoder_name = encoder_config.get("name", "").lower()
        if encoder_config.get("source", "").lower() == "cnn" and encoder_name in (
            "tf_efficientnetv2-detect",
            "tf_efficientnet_b5_ns-detect",
        ):
            efficientnet_name, out_dim = (
                ("efficientnet-b2", 1408)
                if encoder_name == "tf_efficientnetv2-detect"
                else ("efficientnet-b5", 2048)
            )
            self.image_encoder = EfficientNet.from_name(efficientnet_name, num_classes=1)
            self.image_encoder.out_dim = out_dim
        else:
            encoder_init_config = dict(encoder_config)
            encoder_init_config["pretrained"] = False
            self.image_encoder = load_image_encoder(encoder_init_config)
        encoder_weights = {
            key[len("image_encoder."):]: value
            for key, value in checkpoint["model"].items()
            if key.startswith("image_encoder.")
        }
        self.image_encoder.load_state_dict(encoder_weights, strict=True)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False

        self.aggregator = FourViewTransformer(input_dim=self.image_encoder.out_dim)

    def train(self, mode=True):
        super().train(mode)
        # Frozen means parameters and BatchNorm/dropout state stay frozen.
        if self.freeze_encoder:
            self.image_encoder.eval()
        return self

    def encode_images(self, images):
        batch_size, num_images, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * num_images, channels, height, width)
        input_dict = {"image": flat_images, "breast_clip_train_mode": True}
        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with context:
            encoder_output = self.image_encoder(input_dict)
        if isinstance(encoder_output, (tuple, list)):
            image_features = encoder_output[0]
        elif isinstance(encoder_output, dict):
            image_features = encoder_output.get("pooler_output", encoder_output.get("image_features"))
            if image_features is None:
                raise ValueError("Unsupported dictionary output from the Mammo-FM image encoder")
        else:
            image_features = encoder_output
        if image_features.ndim > 2:
            image_features = image_features.flatten(start_dim=1)
        return image_features.reshape(batch_size, num_images, -1)

    def forward(self, images, time_seq, view_seq, side_seq):
        features = self.encode_images(images)
        return self.aggregator(features, time_seq.long(), view_seq.long(), side_seq.long())
