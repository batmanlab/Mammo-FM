"""Feature contracts, grouped splits and bounded-memory feature extraction."""
import csv
import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def factory(spec, config):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)(**config)


def prepare_manifest(source, columns, output, dataset=None, dataset_column='dataset', bu_image_root=None):
    """Explicit field mapping; list values join source fields with underscores."""
    frame = pd.read_csv(source, dtype=str)
    if dataset is not None:
        if dataset_column not in frame:
            raise ValueError('Missing dataset column: ' + dataset_column)
        frame = frame.loc[frame[dataset_column].str.lower() == dataset.lower()]
    if not len(frame):
        raise ValueError('No images remain after dataset selection')
    result = pd.DataFrame(index=frame.index)
    for target, origin in columns.items():
        fields = origin if isinstance(origin, list) else [origin]
        if not fields or not set(fields).issubset(frame) or frame[fields].isna().any().any():
            raise ValueError('Missing source fields for ' + target)
        result[target] = frame[fields].agg('_'.join, axis=1)
    required = ['patient_id','exam_id','image_id','view','image_path']
    if not set(required).issubset(result) or result.duplicated(required[:3]).any():
        raise ValueError('Mapped manifest requires unique complete patient/exam/image keys and view/image_path')
    if not set(result.view).issubset({'L_CC','L_MLO','R_CC','R_MLO'}):
        raise ValueError('Mapped view names must be L_CC, L_MLO, R_CC or R_MLO')
    if bu_image_root:
        def resolve(row):
            parts = Path(row.image_path).parts
            kinds = [kind for kind in ['cases','controls'] if kind in parts]
            if len(kinds) != 1:
                raise ValueError('BU remapping requires cases or controls in each source image path')
            return str(Path(bu_image_root)/kinds[0]/'test_images_png'/row.exam_id/Path(row.image_path).name)
        result['image_path'] = result.apply(resolve, axis=1)
    result.to_csv(output, index=False)


def import_metadata(source, manifest, output):
    """Attach patient/image IDs to historical exam/view patch metadata by key."""
    frame = pd.read_csv(source, dtype={'exam_id':str})
    images = pd.read_csv(manifest, dtype=str)
    required = ['patient_id', 'exam_id', 'image_id', 'view', 'image_path']
    if not len(frame) or not len(images):
        raise ValueError('Historical metadata and image manifest must be nonempty')
    if not set(required).issubset(images) or images[required].isna().any().any():
        raise ValueError('Image manifest requires complete patient/exam/image keys, view and image_path')
    join = ['exam_id','view']
    if not set(join+['patch_h','patch_w']).issubset(frame):
        raise ValueError('Historical metadata requires exam_id, view and patch_h/patch_w')
    if images.duplicated(join).any():
        raise ValueError('Image manifest must uniquely identify each historical exam/view pair')
    if 'grid_h' not in frame or 'grid_w' not in frame:
        if 'spatial_dims' not in frame:
            raise ValueError('Historical metadata must supply explicit grid dimensions')
        shapes = frame.spatial_dims.map(ast.literal_eval)
        if not shapes.map(lambda x: isinstance(x, (tuple,list)) and len(x)==2).all():
            raise ValueError('spatial_dims must contain (height, width) pairs')
        frame['grid_h'] = shapes.map(lambda x:x[0])
        frame['grid_w'] = shapes.map(lambda x:x[1])
    overlap = [c for c in ['patient_id','image_id','image_path'] if c in frame and c in images]
    if overlap:
        check = frame[join+overlap].merge(images[join+overlap], on=join, how='left',
                                          suffixes=('_old','_new'), validate='many_to_one', sort=False)
        for column in overlap:
            if not (check[column+'_old'].astype(str) == check[column+'_new'].astype(str)).all():
                raise ValueError('Historical metadata conflicts with the image manifest: ' + column)
    needed = [c for c in images.columns if c not in frame or c in join]
    result = frame.merge(images[needed], on=join, how='left', validate='many_to_one', sort=False)
    if result[['patient_id','image_id','image_path']].isna().any().any():
        raise ValueError('Some historical patch rows lack a unique matching image')
    result.to_csv(output, index=False)


def read_features(path, metadata=None):
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or not len(features) or features.dtype.kind != "f":
        raise ValueError("Features must be a nonempty floating-point [N,D] .npy array")
    frame = None
    if metadata:
        frame = pd.read_csv(metadata, dtype={"patient_id": str, "exam_id": str, "image_id": str})
        if len(frame) != len(features):
            raise ValueError("Metadata must have exactly one row per feature vector")
        for column in ["patient_id", "exam_id", "image_id", "view"]:
            if column in frame and frame[column].isna().any():
                raise ValueError("Missing metadata values in " + column)
    return features, frame


def grouped_indices(frame, group="patient_id", fraction=0.15, seed=42):
    if not 0 < fraction < 1 or group not in frame or frame[group].isna().any():
        raise ValueError("A complete grouping column and validation fraction in (0,1) are required")
    groups = np.array(sorted(frame[group].unique()))
    if len(groups) < 2:
        raise ValueError("At least two independent groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    count = max(1, min(len(groups) - 1, int(len(groups) * fraction)))
    validation = frame[group].isin(groups[:count]).to_numpy()
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def split(features_path, metadata_path, output, group="patient_id", fraction=0.15, seed=42):
    features, frame = read_features(features_path, metadata_path)
    train, val = grouped_indices(frame, group, fraction, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, indices in [("train", train), ("val", val)]:
        array = np.lib.format.open_memmap(output / (name + ".npy"), mode="w+",
                                          dtype=np.float32, shape=(len(indices), features.shape[1]))
        for start in range(0, len(indices), 8192):
            array[start:start+8192] = features[indices[start:start+8192]]
        array.flush()
        frame.iloc[indices].to_csv(output / (name + "_metadata.csv"), index=False)
    (output / "split.json").write_text(json.dumps({"group": group, "seed": seed,
        "fraction": fraction, "features": str(Path(features_path).resolve()),
        "metadata": str(Path(metadata_path).resolve()), "train_rows": len(train), "val_rows": len(val)}, indent=2))


@torch.no_grad()
def extract(manifest, extractor, output):
    """extractor(row) returns [C,H,W], with preprocessing owned by the adapter."""
    frame = pd.read_csv(manifest, dtype=str)
    required = {"patient_id", "exam_id", "image_id", "view"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Image manifest requires patient_id, exam_id, image_id, view")
    if not len(frame) or frame.duplicated(["patient_id", "exam_id", "image_id"]).any():
        raise ValueError("Image manifest is empty or has duplicate image keys")
    if {"patch_h", "patch_w", "grid_h", "grid_w"}.intersection(frame.columns):
        raise ValueError("Image manifest must not contain pre-existing patch coordinates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    rows = 0
    dimension = None
    with (output / "metadata.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame.columns) + ["patch_h", "patch_w", "grid_h", "grid_w"])
        writer.writeheader()
        for index, row in enumerate(frame.to_dict("records")):
            spatial = torch.as_tensor(extractor(row)).detach().cpu()
            if spatial.ndim != 3 or not torch.isfinite(spatial).all() or min(spatial.shape) == 0:
                raise ValueError("Extractor must return finite, nonempty [C,H,W] features")
            c, h, w = spatial.shape
            if dimension is not None and c != dimension:
                raise ValueError("Encoder feature dimension changed between images")
            dimension = c
            path = output / ("part_%06d.npy" % index)
            np.save(path, spatial.permute(1, 2, 0).reshape(-1, c).float().numpy())
            shards.append((path, h*w))
            rows += h*w
            for y in range(h):
                for x in range(w):
                    writer.writerow(dict(row, patch_h=y, patch_w=x, grid_h=h, grid_w=w))
    combined = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float32,
                                         shape=(rows, dimension))
    offset = 0
    for path, count in shards:
        combined[offset:offset+count] = np.load(path, mmap_mode="r")
        offset += count
    combined.flush()
    for path, _ in shards:
        path.unlink()
