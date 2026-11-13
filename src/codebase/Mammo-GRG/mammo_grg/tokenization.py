"""Minimal LLaVA-compatible tokenization helpers.

Modified from LLaVA: https://github.com/haotian-liu/LLaVA
Copyright 2023 Haotian Liu, Apache-2.0.
"""

from __future__ import annotations

import torch

from .constants import (
    DEFAULT_IMAGE_TOKEN,
    DIRECTIVE_TOKENS,
    IMAGE_TOKEN_INDEX,
    VIEW_TOKENS,
)


def tokenizer_image_token(
    prompt: str,
    tokenizer,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: str | None = None,
):
    """Tokenize text while replacing each ``<image>`` with a sentinel ID."""

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(chunks, separator):
        return [element for pair in zip(chunks, [separator] * len(chunks)) for element in pair][:-1]

    input_ids = []
    offset = 0
    if prompt_chunks and prompt_chunks[0] and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for chunk in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(chunk[offset:])

    if return_tensors is None:
        return input_ids
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    raise ValueError(f"Unsupported tensor type: {return_tensors}")


def add_mammo_grg_tokens(tokenizer, model=None, *, mean_resizing: bool = True) -> dict[str, int]:
    """Add tokens in the same order used by the released stage-2 checkpoint."""

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    tokenizer.add_tokens(list(DIRECTIVE_TOKENS), special_tokens=True)
    existing = list(tokenizer.special_tokens_map.get("additional_special_tokens", []))
    combined = existing + [token for token in VIEW_TOKENS if token not in existing]
    tokenizer.add_special_tokens({"additional_special_tokens": combined})

    if model is not None:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=mean_resizing)

    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in VIEW_TOKENS}
    if len(set(ids.values())) != 4 or any(value == tokenizer.unk_token_id for value in ids.values()):
        raise RuntimeError(f"Failed to register four distinct mammography view tokens: {ids}")
    return ids
