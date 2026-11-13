"""Instruction-tuning dataset for precomputed four-view Mammo-FM maps."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    TOKEN_LCC,
    TOKEN_LMLO,
    TOKEN_RCC,
    TOKEN_RMLO,
)
from .conversation import ASSISTANT_ROLE, END_OF_TURN, USER_ROLE, new_conversation
from .io import load_four_view_embeddings
from .tokenization import tokenizer_image_token


VIEW_PREFIX = " ".join(
    f"{token}{DEFAULT_IMAGE_TOKEN}" for token in (TOKEN_LMLO, TOKEN_LCC, TOKEN_RMLO, TOKEN_RCC)
)


def _add_four_views(conversations: list[dict]) -> list[dict]:
    conversations = copy.deepcopy(conversations)
    for sentence in conversations:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            clean = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            sentence["value"] = f"{VIEW_PREFIX}\n{clean}"
    return conversations


def _stage1_tokens(conversations: list[dict], tokenizer):
    if len(conversations) != 2 or conversations[1].get("from") != "gpt":
        raise ValueError("Stage 1 expects one human/gpt conversation pair")
    prompt = VIEW_PREFIX + conversations[1]["value"] + "\n"
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    labels = input_ids.clone()
    prefix_length = len(tokenizer_image_token(VIEW_PREFIX, tokenizer))
    labels[:prefix_length] = IGNORE_INDEX
    return input_ids, labels


def _stage2_tokens(conversations: list[dict], tokenizer):
    conversations = _add_four_views(conversations)
    conversation = new_conversation()
    role_map = {"human": USER_ROLE, "gpt": ASSISTANT_ROLE}
    if conversations and conversations[0].get("from") != "human":
        conversations = conversations[1:]
    for index, sentence in enumerate(conversations):
        role = role_map[sentence["from"]]
        expected = USER_ROLE if index % 2 == 0 else ASSISTANT_ROLE
        if role != expected:
            raise ValueError("Conversation roles must alternate human/gpt")
        conversation.append(role, sentence["value"])
    prompt = conversation.prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    labels = input_ids.clone()

    # Faithful masking from the original Mammo-GRG ``preprocess_llama3``.
    separator = END_OF_TURN + ASSISTANT_ROLE
    rounds = prompt.split(END_OF_TURN)
    regrouped = [END_OF_TURN.join(rounds[:3])]
    regrouped.extend(END_OF_TURN.join(rounds[i : i + 2]) for i in range(3, len(rounds), 2))
    cursor = 0
    for round_index, text in enumerate(regrouped):
        if not text:
            break
        parts = text.split(separator)
        if len(parts) != 2:
            break
        instruction = parts[0] + separator
        round_length = len(tokenizer_image_token(text, tokenizer)) + 1
        instruction_length = len(tokenizer_image_token(instruction, tokenizer))
        if round_index > 0:
            round_length -= 1
            instruction_length -= 1
        labels[cursor : cursor + instruction_length] = IGNORE_INDEX
        cursor += round_length
    labels[cursor:] = IGNORE_INDEX
    return input_ids, labels


class MammoGRGDataset(Dataset):
    def __init__(self, data_path, embedding_root, tokenizer, stage: int):
        self.samples = [item for item in json.loads(Path(data_path).read_text()) if item.get("conversations")]
        self.embedding_root = embedding_root
        self.tokenizer = tokenizer
        self.stage = stage

    def __len__(self):
        return len(self.samples)

    @property
    def modality_lengths(self):
        return [sum(len(item["value"].split()) for item in sample["conversations"]) for sample in self.samples]

    def __getitem__(self, index):
        sample = self.samples[index]
        embedding_paths = sample.get("embeddings", sample.get("image"))
        if embedding_paths is None:
            raise KeyError("Each sample requires 'embeddings' (preferred) or 'image'")
        images = load_four_view_embeddings(
            embedding_paths, embedding_root=self.embedding_root
        ).squeeze(0)
        if self.stage == 1:
            input_ids, labels = _stage1_tokens(sample["conversations"], self.tokenizer)
        else:
            input_ids, labels = _stage2_tokens(sample["conversations"], self.tokenizer)
        return {"input_ids": input_ids, "labels": labels, "images": images}


class MammoGRGCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )[:, : self.tokenizer.model_max_length]
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in instances], batch_first=True, padding_value=IGNORE_INDEX
        )[:, : self.tokenizer.model_max_length]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "images": torch.stack([item["images"] for item in instances]),
        }
