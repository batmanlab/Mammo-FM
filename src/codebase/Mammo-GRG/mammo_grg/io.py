"""Safe, dataset-neutral loading of four-view Mammo-FM embeddings."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .constants import EXPECTED_EMBEDDING_SHAPE, VIEW_TOKENS


def parse_path_list(value) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("(") or stripped.startswith("{"):
            value = ast.literal_eval(stripped)
        else:
            value = [stripped]
    if isinstance(value, Mapping):
        aliases = ("LMLO", "LCC", "RMLO", "RCC")
        value = [value[key] for key in aliases]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError("Embedding paths must be a four-item list or a LMLO/LCC/RMLO/RCC mapping")
    paths = [str(item) for item in value]
    if len(paths) != 4:
        raise ValueError(f"Expected four paths ordered {VIEW_TOKENS}; received {len(paths)}")
    return paths


def resolve_embedding_paths(value, embedding_root: str | Path | None = None) -> list[Path]:
    root = Path(embedding_root).expanduser().resolve() if embedding_root else None
    resolved = []
    for raw_path in parse_path_list(value):
        path = Path(raw_path).expanduser()
        if path.suffix.lower() != ".pt":
            path = path.with_suffix(".pt")
        if not path.is_absolute() and root is not None:
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Embedding not found: {path}")
        resolved.append(path)
    return resolved


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, found {type(value).__name__}")
    if value.shape == EXPECTED_EMBEDDING_SHAPE[1:]:
        value = value.unsqueeze(0)
    if tuple(value.shape) != EXPECTED_EMBEDDING_SHAPE:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_SHAPE} in {path}, found {tuple(value.shape)}"
        )
    return value


def load_four_view_embeddings(
    paths,
    *,
    embedding_root: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Load LMLO, LCC, RMLO and RCC tensors as ``[1,4,1,2048,48,29]``."""

    tensors = [_load_tensor(path) for path in resolve_embedding_paths(paths, embedding_root)]
    batch = torch.stack(tensors, dim=0).unsqueeze(0)
    return batch.to(device=device, dtype=dtype)
