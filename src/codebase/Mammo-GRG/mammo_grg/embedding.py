"""Exact Mammo-FM epoch-3 feature extraction used by the report generator.

This dataset-neutral extractor preserves the released training pipeline's
min-max/mean-std preprocessing and saves the pre-pooling EfficientNet-B5 map
with shape ``[1, 2048, 48, 29]`` per image.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


class MammoFMFeatureExtractor(torch.nn.Module):
    def __init__(self, checkpoint: str | Path):
        super().__init__()
        # The implementation is already part of this public Mammo-FM codebase
        # and is behaviorally identical to the archived extractor backbone.
        codebase_root = Path(__file__).resolve().parents[2]
        if str(codebase_root) not in sys.path:
            sys.path.insert(0, str(codebase_root))
        from breastclip.model.modules.efficientnet_custom import EfficientNet

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Mammo-FM checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" not in state:
            raise KeyError(f"Checkpoint does not contain a 'model' state dict: {checkpoint}")

        # ``from_name`` avoids an unnecessary ImageNet download; every encoder
        # parameter is then replaced using the strict Mammo-FM state dict.
        self.image_encoder = EfficientNet.from_name("efficientnet-b5", num_classes=1)
        weights = {
            key.removeprefix("image_encoder."): value
            for key, value in state["model"].items()
            if key.startswith("image_encoder.")
        }
        if not weights:
            raise KeyError("No 'image_encoder.' weights were found in the Mammo-FM checkpoint")
        self.image_encoder.load_state_dict(weights, strict=True)
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, raw_features = self.image_encoder(
            {"image": images, "breast_clip_train_mode": True}
        )
        return raw_features


def preprocess_png(path: str | Path, mean: float, std: float) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    image -= image.min()
    maximum = image.max()
    if maximum <= 0:
        raise ValueError(f"Cannot min-max normalize a constant image: {path}")
    image /= maximum
    image = (image - mean) / std
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()
