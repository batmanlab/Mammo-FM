"""Optional model connections. Core SAE workflows do not import these models."""

import numpy as np
import torch
from PIL import Image

from .core import load_checkpoint


class FourViewPredictor:
    def __init__(self, head, view_order):
        self.head = head
        self.view_order = view_order

    def bind(self, metadata):
        if len(metadata[["patient_id", "exam_id"]].drop_duplicates()) != 1:
            raise ValueError("Bind a predictor to exactly one examination")
        if set(metadata.view) != set(self.view_order):
            raise ValueError("Expected exactly these views: " + str(self.view_order))
        if "image_id" in metadata:
            if (metadata.groupby("view").image_id.nunique() != 1).any():
                raise ValueError("Select exactly one image for each view before prediction")
        indices = [np.flatnonzero(metadata.view.to_numpy() == v) for v in self.view_order]
        def predict(patches):
            if len(patches) != len(metadata):
                raise ValueError("Prediction patches do not match bound metadata")
            pooled = torch.stack([patches[i].mean(0) for i in indices]).unsqueeze(0)
            return self.head(pooled).squeeze(0)
        return predict


def legacy_risk(checkpoint, device="cpu"):
    """Load the legacy aggregator from cleaned or historical checkpoints."""
    from mayo_mammofm_risk.model import FourViewTransformer, compatible_model_state
    saved = load_checkpoint(checkpoint)
    state = compatible_model_state(saved["model"])
    weights = {k[len("aggregator."):]: v for k, v in state.items() if k.startswith("aggregator.")}
    if not weights:
        raise ValueError("Expected a cleaned legacy checkpoint with aggregator.* weights")
    model = FourViewTransformer(input_dim=weights["projection_layer.weight"].shape[1]).to(device).eval()
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    def head(x):
        x = x.to(device)
        time = torch.zeros((1, 4), device=device, dtype=torch.long)
        # Match mayo_mammofm_risk.data, not the historical teacher/student wrapper.
        view = torch.tensor([[1, 1, 0, 0]], device=device)
        side = torch.tensor([[0, 1, 0, 1]], device=device)
        return model(x, time, view, side).sigmoid()
    return FourViewPredictor(head, ["L_CC", "R_CC", "L_MLO", "R_MLO"])


def breastclip_spatial(checkpoint, device="cpu", mean=0.3089279, std=0.25053555408335154):
    """BU PNG preprocessing, without resize/crop, and a detector spatial encoder.

    This adapter is explicitly for the released EfficientNet-B5 detector. Other
    preprocessing/encoders should provide their own extractor factory.
    """
    from breastclip.model.modules import EfficientNet
    saved = load_checkpoint(checkpoint)
    config = saved["config"]["model"]["image_encoder"]
    if config["name"].lower() != "tf_efficientnet_b5_ns-detect":
        raise ValueError("This spatial adapter supports tf_efficientnet_b5_ns-detect only")
    if std <= 0:
        raise ValueError("Image standard deviation must be positive")
    encoder = EfficientNet.from_name("efficientnet-b5", num_classes=1)
    weights = {k[len("image_encoder."):]: v for k, v in saved["model"].items()
               if k.startswith("image_encoder.")}
    encoder.load_state_dict(weights, strict=True)
    encoder.to(device).eval().requires_grad_(False)
    def extract(row):
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.float32)
        image -= image.min()
        maximum = image.max()
        if maximum > 0:
            image /= maximum
        tensor = torch.from_numpy((image - mean) / std).permute(2, 0, 1).unsqueeze(0).to(device)
        # The detector's extract_features returns the pre-pooling spatial map.
        return encoder.extract_features(tensor)[0]
    return extract


class BreastClipText:
    """Restore trained text weights and full projection heads from one checkpoint."""
    def __init__(self, checkpoint, text_model_directory, device="cpu", max_length=256):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from breastclip.model.modules import load_projection_head
        saved = load_checkpoint(checkpoint)
        config, state = saved["config"]["model"], saved["model"]
        self.device = device
        self.max_length = max_length
        self.pooling = config["text_encoder"]["pooling"]
        text_config = AutoConfig.from_pretrained(text_model_directory, local_files_only=True)
        self.encoder = AutoModel.from_config(text_config)
        weights = {k[len("text_encoder."):]: v for k, v in state.items() if k.startswith("text_encoder.")}
        if weights and all(k.startswith("text_encoder.") for k in weights):
            weights = {k[len("text_encoder."):]: v for k, v in weights.items()}
        self.encoder.load_state_dict(weights, strict=True)
        self.encoder.to(device).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_directory, local_files_only=True)
        self.projections = {}
        for name in ["image_projection", "text_projection"]:
            prefix = name + "."
            weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            projection = load_projection_head(weights["projection.weight"].shape[1], config["projection_head"])
            projection.load_state_dict(weights, strict=True)
            self.projections[name] = projection.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, sentences):
        tokens = self.tokenizer(sentences, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt").to(self.device)
        hidden = self.encoder(**tokens).last_hidden_state
        if self.pooling == "mean":
            mask = tokens.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        elif self.pooling == "bos":
            pooled = hidden[:, 0]
        elif self.pooling == "eos":
            pooled = hidden[torch.arange(len(hidden), device=hidden.device), tokens.attention_mask.sum(1)-1]
        else:
            raise ValueError("Unsupported text pooling: " + self.pooling)
        return self.projections["text_projection"](pooled)

    def project_image(self, pooled):
        return self.projections["image_projection"](pooled.to(self.device))


def breastclip_text(**kwargs):
    return BreastClipText(**kwargs)
