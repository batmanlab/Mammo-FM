"""Public inference API for Mammo-GRG."""

from __future__ import annotations

import torch

from .checkpoint import load_stage2_model
from .constants import DEFAULT_IMAGE_TOKEN, TOKEN_FOR_REPORT_GENERATION, VIEW_TOKENS
from .conversation import ASSISTANT_ROLE, USER_ROLE, new_conversation
from .io import load_four_view_embeddings
from .tokenization import tokenizer_image_token


def load_model(model_path, model_base, *, device="cuda", load_4bit=False):
    """Return ``(tokenizer, model)`` for a downloaded stage-2 checkpoint."""

    return load_stage2_model(model_path, model_base, device=device, load_4bit=load_4bit)


def format_question(question: str) -> str:
    view_blocks = " ".join(f"{token}{DEFAULT_IMAGE_TOKEN}" for token in VIEW_TOKENS)
    clean = question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    return f"{view_blocks}\n{clean}"


@torch.inference_mode()
def generate_report(
    model,
    tokenizer,
    embeddings: torch.Tensor,
    question: str = "Generate a complete mammography report.",
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """Generate one report from a tensor shaped ``[1,4,1,2048,48,29]``."""

    if TOKEN_FOR_REPORT_GENERATION not in question:
        question = f"{TOKEN_FOR_REPORT_GENERATION} {question}"
    conversation = new_conversation()
    conversation.append(USER_ROLE, format_question(question))
    conversation.append(ASSISTANT_ROLE, None)
    prompt = conversation.prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(model.device)
    embeddings = embeddings.to(model.device, dtype=model.dtype)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    generation_args = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=embeddings,
        do_sample=temperature > 0,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        generation_args.update(temperature=temperature, top_p=top_p)
    output_ids = model.generate(**generation_args)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


__all__ = ["load_model", "load_four_view_embeddings", "generate_report"]
