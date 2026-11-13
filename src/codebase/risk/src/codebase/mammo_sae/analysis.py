"""Neuron summaries, spatial maps and sentence ablation in shared CLIP space."""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .core import validate_neurons


@torch.no_grad()
def neuron_summary(sae, features, metadata, output, top_k=30, batch_size=256, level=-1):
    if not 1 <= top_k <= sae.model.n_latents:
        raise ValueError("top_k must be within the latent dimension")
    if not {"patient_id", "exam_id"}.issubset(metadata):
        raise ValueError("Neuron summaries require patient_id and exam_id")
    rows = []
    for (patient, exam), frame in metadata.groupby(["patient_id", "exam_id"], sort=False):
        counts = torch.zeros(sae.model.n_latents, device=sae.mean.device)
        sums, maximum = counts.clone(), counts.clone()
        indices = frame.index.to_numpy()
        for start in range(0, len(indices), batch_size):
            z = sae.encode(features[indices[start:start+batch_size]], level)
            counts += (z > 0).sum(0)
            sums += z.sum(0)
            maximum = torch.maximum(maximum, z.max(0).values)
        for neuron in maximum.topk(top_k).indices.tolist():
            rows.append(dict(patient_id=patient, exam_id=exam, neuron_id=neuron,
                             max_activation=maximum[neuron].item(),
                             mean_activation=sums[neuron].item()/len(indices),
                             activation_rate=counts[neuron].item()/len(indices), patches=len(indices)))
    pd.DataFrame(rows).to_csv(output, index=False)


def spatial_map(values, metadata):
    required = {"patch_h", "patch_w", "grid_h", "grid_w"}
    if not required.issubset(metadata) or len(values) != len(metadata):
        raise ValueError("Spatial metadata must match activations and contain grid coordinates")
    if metadata[["grid_h", "grid_w"]].drop_duplicates().shape[0] != 1:
        raise ValueError("A heatmap must describe exactly one grid")
    coordinates = metadata[["patch_h", "patch_w", "grid_h", "grid_w"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or not np.equal(coordinates, np.floor(coordinates)).all():
        raise ValueError("Patch coordinates and grid sizes must be finite integers")
    h, w = map(int, metadata.iloc[0][["grid_h", "grid_w"]])
    y = metadata.patch_h.to_numpy(dtype=int)
    x = metadata.patch_w.to_numpy(dtype=int)
    if h < 1 or w < 1 or len(values) != h*w or len(set(zip(y, x))) != h*w:
        raise ValueError("Heatmaps require a complete grid without duplicate coordinates")
    if (y < 0).any() or (y >= h).any() or (x < 0).any() or (x >= w).any():
        raise ValueError("Patch coordinates lie outside the grid")
    result = np.empty((h, w), dtype=np.float32)
    result[y, x] = values
    return result


@torch.no_grad()
def heatmaps(sae, features, metadata, neurons, output, level=-1, batch_size=256):
    validate_neurons(neurons, sae.model.n_latents)
    if not neurons or not {"patient_id", "exam_id", "image_id"}.issubset(metadata):
        raise ValueError("Provide neurons and patient/exam/image identifiers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for image_index, (keys, frame) in enumerate(metadata.groupby(["patient_id", "exam_id", "image_id"], sort=False)):
        indices = frame.index.to_numpy()
        activations = []
        for start in range(0, len(indices), batch_size):
            z = sae.encode(features[indices[start:start+batch_size]], level)
            activations.append(z[:, neurons].cpu().numpy())
        values = np.concatenate(activations)
        for j, neuron in enumerate(neurons):
            array = spatial_map(values[:, j], frame)
            name = "image_%06d_neuron_%d" % (image_index, neuron)
            np.save(output / (name + ".npy"), array)
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            plt.imsave(output / (name + ".png"), array, cmap="hot", vmin=0)
            manifest.append(dict(patient_id=keys[0], exam_id=keys[1], image_id=keys[2],
                                 neuron_id=neuron, heatmap=name + ".npy", preview=name + ".png"))
    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)


@torch.no_grad()
def sentence_ablation(sae, features, sentences, text_adapter, neurons, level=-1):
    """One image's raw cosine-similarity differences; these are not probabilities.

    Adapter implements encode_text(list[str]) and project_image([B,D]), using
    paired trained encoders, their original pooling, and complete projections.
    """
    validate_neurons(neurons, sae.model.n_latents)
    if not sentences:
        raise ValueError("At least one sentence is required")
    z = sae.encode(features, level)
    text = F.normalize(text_adapter.encode_text(sentences).to(z.device), dim=-1)
    def score(activation):
        pooled = sae.decode(activation).mean(0, keepdim=True)
        image = F.normalize(text_adapter.project_image(pooled).to(z.device), dim=-1)
        return (image @ text.T).squeeze(0)
    baseline = score(z)
    changed = z.clone()
    changed[:, neurons] = 0
    after = score(changed)
    return [dict(sentence=s, baseline_similarity=float(b), ablated_similarity=float(a),
                 similarity_drop=float(b-a)) for s, b, a in zip(sentences, baseline, after)]


def render_sentences(rows):
    return '<!doctype html><meta charset="utf-8"><title>Sentence ablation</title>' + \
        '<h1>Sentence ablation</h1><p>Positive scores indicate reduced image–text similarity after suppression.</p>' + \
        ''.join('<p><strong>%.6f</strong> %s</p>' % (r['similarity_drop'], html.escape(r['sentence'])) for r in rows)
