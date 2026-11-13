#!/usr/bin/env python3
"""Extract epoch-3 Mammo-FM feature maps from a CSV image manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

GRG_ROOT = Path(__file__).resolve().parents[1]
if str(GRG_ROOT) not in sys.path:
    sys.path.insert(0, str(GRG_ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from mammo_grg.constants import EXPECTED_EMBEDDING_SHAPE
from mammo_grg.embedding import MammoFMFeatureExtractor, preprocess_png


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save raw Mammo-FM maps as one .pt file per PNG.",
        epilog=(
            "Use the epoch-3 checkpoint Mammo-FM_BatmanlabTrained_CLIP_last.tar "
            "from https://huggingface.co/batmanLab/Mammo-FM."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="CSV containing one image path per row")
    parser.add_argument("--image-column", default="file_path")
    parser.add_argument("--image-root", default=None, help="Prepended to relative image paths")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mean", type=float, default=0.3089279)
    parser.add_argument("--std", type=float, default=0.25053555408335154)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None
    output_root = Path(args.output_root).expanduser().resolve()
    manifest = pd.read_csv(args.manifest)
    if args.image_column not in manifest:
        raise KeyError(f"Missing column {args.image_column!r} in {args.manifest}")

    model = MammoFMFeatureExtractor(args.checkpoint).to(args.device).eval()
    for raw_path in tqdm(manifest[args.image_column].astype(str), desc="Extracting embeddings"):
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            if image_root is None:
                raise ValueError("Relative manifest paths require --image-root")
            image_path = image_root / image_path
        image_path = image_path.resolve()
        if image_root is not None:
            try:
                relative = image_path.relative_to(image_root)
            except ValueError as error:
                raise ValueError(f"Image is outside --image-root: {image_path}") from error
        else:
            relative = Path(image_path.name)
        output_path = (output_root / relative).with_suffix(".pt")
        if output_path.exists() and not args.overwrite:
            continue

        image = preprocess_png(image_path, args.mean, args.std).unsqueeze(0).to(args.device)
        features = model(image).cpu()
        if tuple(features.shape) != EXPECTED_EMBEDDING_SHAPE:
            raise RuntimeError(
                f"Expected {EXPECTED_EMBEDDING_SHAPE} for {image_path}, got {tuple(features.shape)}. "
                "The public RRG configuration requires 1520x912 preprocessed PNGs."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, output_path)


if __name__ == "__main__":
    main()
