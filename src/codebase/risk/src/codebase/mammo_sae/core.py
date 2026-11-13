"""Checkpoint-aware SAE inference in the encoder's original feature units."""
import json
from pathlib import Path

import torch
from torch import nn

from .impl.sae import MatryoshkaAutoencoder


def load_checkpoint(path):
    # Research checkpoints can contain NumPy scalars as well as tensors.
    return torch.load(path, map_location="cpu", weights_only=False)


class SpatialSAE(nn.Module):
    def __init__(self, model, mean, scale):
        super().__init__()
        if model.normalize:
            raise ValueError("Per-vector normalized SAEs require decode context; unsupported here")
        self.model = model
        mean = torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if mean.shape != (model.n_inputs,) or not torch.isfinite(mean).all():
            raise ValueError("Invalid checkpoint feature mean")
        if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
            raise ValueError("Checkpoint scaling_factor must be a finite positive scalar")
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    @classmethod
    def load(cls, path, config=None, device="cpu", preprocessing="checkpoint"):
        checkpoint = load_checkpoint(path)
        saved = checkpoint.get("model_config")
        if config is None:
            config = saved
        elif isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        if config is None:
            raise ValueError("Historical checkpoint requires an explicit model configuration JSON")
        if saved is not None and config != saved:
            raise ValueError("Supplied configuration differs from checkpoint model_config")
        model = MatryoshkaAutoencoder(**config)
        model.load_state_dict(checkpoint["model"], strict=True)
        if preprocessing == "checkpoint":
            mean, scale = checkpoint["mean_center"], checkpoint["scaling_factor"]
        elif preprocessing == "identity":
            mean, scale = torch.zeros(model.n_inputs), 1.0
        else:
            raise ValueError("preprocessing must be checkpoint or identity")
        return cls(model, mean, scale).to(device).eval()

    def encode(self, features, level=-1):
        features = torch.as_tensor(features, device=self.mean.device, dtype=self.mean.dtype)
        if features.ndim != 2 or features.shape[1] != self.model.n_inputs:
            raise ValueError("Expected patch features [N, input_dim]")
        if not torch.isfinite(features).all():
            raise ValueError("Features contain nonfinite values")
        # Inference needs one level, not nine simultaneous dense latent arrays.
        normalized = (features - self.mean) * self.scale
        pre_activation = self.model.encode_pre_act(normalized)
        return self.model.latent_soft_cap(self.model.activation[level](pre_activation))

    def decode(self, activations):
        return self.model.decode([activations])[0] / self.scale + self.mean

    def forward(self, features, level=-1):
        return self.decode(self.encode(features, level))


def validate_neurons(neurons, n_latents):
    if len(neurons) != len(set(neurons)) or any(n < 0 or n >= n_latents for n in neurons):
        raise ValueError("Neuron IDs must be unique and within the SAE latent dimension")


@torch.no_grad()
def intervene(sae, features, predictor, neurons, level=-1):
    """Report original, reconstructed, and ablated predictions separately.

    predictor accepts [N,D] patches and returns probabilities or scores. Its
    closure owns view ordering, pooling, metadata and the frozen prediction head.
    """
    validate_neurons(neurons, sae.model.n_latents)
    x = torch.as_tensor(features, dtype=sae.mean.dtype, device=sae.mean.device)
    z = sae.encode(x, level)
    original = predictor(x)
    baseline = predictor(sae.decode(z))
    changed = z.clone()
    changed[:, neurons] = 0
    ablated = predictor(sae.decode(changed))
    return {"original": original, "reconstructed": baseline, "ablated": ablated,
            "reconstruction_delta": baseline - original, "ablation_delta": ablated - baseline}


@torch.no_grad()
def shapley(sae, features, predictor, neurons, samples=100, seed=42, level=-1):
    """Permutation Shapley over selected neurons; unselected neurons stay active.

    Every permutation uses the same examination and reconstruction level.
    All selected neurons absent defines the baseline coalition. The sum of
    contributions equals the full-minus-empty score for each permutation.
    """
    validate_neurons(neurons, sae.model.n_latents)
    if samples < 1 or not neurons:
        raise ValueError("Provide neurons and a positive number of permutations")
    z = sae.encode(features, level)
    generator = torch.Generator().manual_seed(seed)
    sums = {}
    for _ in range(samples):
        coalition = z.clone()
        coalition[:, neurons] = 0
        previous = predictor(sae.decode(coalition))
        for index in torch.randperm(len(neurons), generator=generator).tolist():
            neuron = neurons[index]
            coalition[:, neuron] = z[:, neuron]
            current = predictor(sae.decode(coalition))
            sums[neuron] = sums.get(neuron, torch.zeros_like(current)) + current - previous
            previous = current
    return {n: sums[n] / samples for n in neurons}


@torch.no_grad()
def shapley_all_background(sae, features, predictor, neurons, samples=100, seed=42, level=-1):
    """Target-neuron Shapley values with every latent eligible for each coalition.

    Unlike selected-neuron Shapley, unselected neurons are not held active.
    Estimates are for this examination; a cohort aggregate must be explicit.
    """
    validate_neurons(neurons, sae.model.n_latents)
    if samples < 1 or not neurons:
        raise ValueError("Provide neurons and a positive number of permutations")
    z = sae.encode(features, level)
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for neuron in neurons:
        total = None
        for _ in range(samples):
            order = torch.randperm(sae.model.n_latents, generator=generator)
            position = int((order == neuron).nonzero()[0])
            coalition = torch.zeros_like(z)
            predecessors = order[:position].to(z.device)
            coalition[:, predecessors] = z[:, predecessors]
            before = predictor(sae.decode(coalition))
            coalition[:, neuron] = z[:, neuron]
            delta = predictor(sae.decode(coalition)) - before
            total = delta if total is None else total + delta
        result[neuron] = total / samples
    return result


class ReconstructedEncoder(nn.Module):
    """Generalizability adapter: encoder [B,C,H,W] -> SAE -> mean pool [B,C]."""
    def __init__(self, spatial_encoder, sae, level=-1):
        super().__init__()
        self.spatial_encoder, self.sae, self.level = spatial_encoder, sae, level
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, images):
        with torch.no_grad():
            spatial = self.spatial_encoder(images)
            if spatial.ndim != 4:
                raise ValueError("Spatial encoder must return [B,C,H,W]")
            b, c, h, w = spatial.shape
            flat = spatial.permute(0, 2, 3, 1).reshape(-1, c)
            return self.sae(flat, self.level).reshape(b, h * w, c).mean(1)
