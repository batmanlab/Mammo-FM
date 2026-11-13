"""Finding maps, image overlays, connected-component boxes and detection AP."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

from .analysis import spatial_map
from .core import validate_neurons
from .experiments import table

KEYS = ["patient_id", "exam_id", "image_id"]
BOX = ["x1", "y1", "x2", "y2"]


def activation_box(activation, width, height, threshold, corner_margin=0, percentile=40):
    """Best-mean component, optionally refined by within-component percentile."""
    if width <= 0 or height <= 0 or corner_margin < 0 or not 0 <= percentile <= 100:
        raise ValueError("Invalid image geometry or component settings")
    activation = np.asarray(activation).copy()
    if corner_margin:
        m = corner_margin
        activation[:m,:m] = 0
        activation[:m,-m:] = 0
        activation[-m:,:m] = 0
        activation[-m:,-m:] = 0
    components, n = ndimage.label(activation > threshold)
    if not n:
        return None
    best = max(range(1, n+1), key=lambda i: activation[components == i].mean())
    mask = components == best
    selected = mask & (activation >= np.percentile(activation[mask], percentile))
    ys, xs = np.where(selected)
    h, w = activation.shape
    return dict(x1=float(xs.min()*width/w), y1=float(ys.min()*height/h),
                x2=float((xs.max()+1)*width/w), y2=float((ys.max()+1)*height/h),
                score=float(activation[mask].max()))


@torch.no_grad()
def localize(sae, features, metadata, config, output, level=-1, batch_size=256):
    """config maps finding names to neuron IDs, threshold and optional settings."""
    if not set(KEYS+["image_path"]).issubset(metadata) or not config:
        raise ValueError("Localization needs image paths, image keys and a finding configuration")
    for settings in config.values():
        validate_neurons(settings['neurons'], sae.model.n_latents)
        if not settings['neurons'] or not np.isfinite(settings['threshold']) or settings['threshold'] < 0:
            raise ValueError("Each finding needs neurons and a nonnegative finite threshold")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    boxes, images, maps = [], [], []
    for index, (identity, frame) in enumerate(metadata.groupby(KEYS, sort=False)):
        if frame.image_path.nunique() != 1:
            raise ValueError("Each image key must refer to one image path")
        path = frame.image_path.iloc[0]
        with Image.open(path) as source:
            image = source.convert('RGB')
        width, height = image.size
        key = dict(zip(KEYS, identity))
        images.append(dict(key, image_path=path, width=width, height=height))
        indices = frame.index.to_numpy()
        values = {name: [] for name in config}
        for start in range(0, len(indices), batch_size):
            z = sae.encode(features[indices[start:start+batch_size]], level)
            for name, settings in config.items():
                values[name].append(z[:,settings['neurons']].max(1).values.cpu().numpy())
        for finding_index, (finding, settings) in enumerate(config.items()):
            activation = spatial_map(np.concatenate(values[finding]), frame)
            stem = 'image_%06d_finding_%03d' % (index, finding_index)
            np.save(output/(stem+'.npy'), activation)
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            fig, axis = plt.subplots(figsize=(6, 9))
            axis.imshow(image)
            axis.imshow(activation, extent=(0,width,height,0), cmap='hot', alpha=0.4, vmin=0)
            box = activation_box(activation, width, height, settings['threshold'],
                                 settings.get('corner_margin', 0), settings.get('percentile', 40))
            if box is not None:
                boxes.append(dict(key, finding=finding, **box))
                from matplotlib.patches import Rectangle
                axis.add_patch(Rectangle((box['x1'],box['y1']),box['x2']-box['x1'],box['y2']-box['y1'],
                                         fill=False,edgecolor='cyan',linewidth=1.5))
            axis.set_title(finding)
            axis.axis('off')
            fig.savefig(output/(stem+'.png'), bbox_inches='tight', dpi=120)
            plt.close(fig)
            maps.append(dict(key, finding=finding, activation=stem+'.npy', overlay=stem+'.png'))
    pd.DataFrame(boxes, columns=KEYS+['finding']+BOX+['score']).to_csv(output/'detections.csv', index=False)
    pd.DataFrame(images).to_csv(output/'images.csv', index=False)
    pd.DataFrame(maps).to_csv(output/'maps.csv', index=False)


def iou(box, boxes):
    minimum = np.maximum(box[:2], boxes[:,:2])
    maximum = np.minimum(box[2:], boxes[:,2:])
    intersection = np.maximum(maximum-minimum, 0).prod(1)
    union = np.prod(box[2:]-box[:2]) + np.prod(boxes[:,2:]-boxes[:,:2],axis=1) - intersection
    return intersection / np.maximum(union, np.finfo(float).eps)


def average_precision(detections, annotations, threshold):
    """Score-ordered, one-to-one matching; duplicate detections are false positives."""
    if not len(annotations):
        return dict(ap=None, annotations=0, detections=len(detections))
    truth = {key: f[BOX].to_numpy(dtype=float) for key,f in annotations.groupby(KEYS)}
    used = {key: set() for key in truth}
    tp, fp = [], []
    for row in detections.sort_values('score', ascending=False, kind='stable').to_dict('records'):
        key = tuple(row[k] for k in KEYS)
        matched = False
        if key in truth:
            overlaps = iou(np.array([row[c] for c in BOX]), truth[key])
            # Best as-yet-unmatched target, including multiple lesions per image.
            candidates = [i for i in range(len(overlaps)) if i not in used[key] and overlaps[i] >= threshold]
            if candidates:
                target = max(candidates, key=lambda i: overlaps[i])
                used[key].add(target)
                matched = True
        tp.append(int(matched)); fp.append(int(not matched))
    recall = np.cumsum(tp)/len(annotations)
    precision = np.cumsum(tp)/np.maximum(np.cumsum(tp)+np.cumsum(fp),1)
    r = np.r_[0, recall, 1]
    p = np.r_[0, precision, 0]
    p = np.maximum.accumulate(p[::-1])[::-1]
    changed = np.flatnonzero(r[1:] != r[:-1])
    ap = float(np.sum((r[changed+1]-r[changed])*p[changed+1]))
    return dict(ap=ap, annotations=len(annotations), detections=len(detections),
                true_positives=sum(tp), false_positives=sum(fp))


def evaluate(detections_path, annotations_path, images_path, output, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError("IoU must be in (0,1]")
    predictions, annotations, images = map(table, [detections_path, annotations_path, images_path])
    if not set(KEYS).issubset(images) or images[KEYS].isna().any().any() or images.duplicated(KEYS).any():
        raise ValueError("Evaluation images must have unique complete keys")
    cohort = set(map(tuple, images[KEYS].to_numpy()))
    for frame in [predictions, annotations]:
        required = KEYS+['finding']+BOX+(['score'] if frame is predictions else [])
        if not set(required).issubset(frame) or frame[required].isna().any().any():
            raise ValueError("Boxes require keys, finding and x1/y1/x2/y2; detections also require score")
        if not set(map(tuple, frame[KEYS].to_numpy())).issubset(cohort):
            raise ValueError("Box image IDs fall outside the evaluation cohort")
        coords = frame[BOX].to_numpy(dtype=float)
        if not np.isfinite(coords).all() or (coords[:,2:] <= coords[:,:2]).any():
            raise ValueError("Boxes must be finite and have positive area")
    if not np.isfinite(predictions.score.to_numpy()).all():
        raise ValueError("Detection scores must be finite")
    result = {finding: average_precision(predictions[predictions.finding==finding],
                                         annotations[annotations.finding==finding], threshold)
              for finding in sorted(set(predictions.finding) | set(annotations.finding))}
    aps = [r['ap'] for r in result.values() if r['ap'] is not None]
    Path(output).write_text(json.dumps(dict(iou=threshold, images=len(images), findings=result,
                                           mean_ap=float(np.mean(aps)) if aps else None), indent=2))
