"""Loading utilities for Mammo-GRG stage-1 and stage-2 checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from .modeling_mammo_grg import MammoGRGConfig, MammoGRGForCausalLM
from .tokenization import add_mammo_grg_tokens


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_non_lora_keys(state_dict: dict) -> dict:
    normalized = {(key[11:] if key.startswith("base_model.") else key): value for key, value in state_dict.items()}
    if any(key.startswith("model.model.") for key in normalized):
        normalized = {(key[6:] if key.startswith("model.") else key): value for key, value in normalized.items()}
    return normalized


def validate_stage2_checkpoint(model_path: str | Path) -> Path:
    """Validate checkpoint contents without relying on its directory name."""

    checkpoint = Path(model_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Stage-2 checkpoint directory not found: {checkpoint}")
    required = ("adapter_config.json", "non_lora_trainables.bin", "config.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    adapters = (checkpoint / "adapter_model.safetensors", checkpoint / "adapter_model.bin")
    if not any(path.is_file() for path in adapters):
        missing.append("adapter_model.safetensors (or adapter_model.bin)")
    if missing:
        raise FileNotFoundError(f"Incomplete stage-2 checkpoint {checkpoint}: missing {', '.join(missing)}")
    return checkpoint


def load_stage2_model(
    model_path: str | Path,
    model_base: str,
    *,
    device: str = "cuda",
    load_4bit: bool = False,
):
    """Load and merge the released stage-2 LoRA checkpoint."""

    checkpoint = validate_stage2_checkpoint(model_path)
    non_lora_path = checkpoint / "non_lora_trainables.bin"
    if not non_lora_path.is_file():
        raise FileNotFoundError(f"Missing required checkpoint file: {non_lora_path}")

    config = MammoGRGConfig.from_pretrained(checkpoint)
    base_config = AutoConfig.from_pretrained(model_base)
    config.vocab_size = base_config.vocab_size
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    dtype = torch.float16 if device != "cpu" else torch.float32
    load_kwargs = {
        "config": config,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "device_map": {"": device},
    }
    if load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype")

    model = MammoGRGForCausalLM.from_pretrained(model_base, **load_kwargs)
    # Released non-LoRA weights overwrite the expanded embedding rows, so the
    # inference loader follows the original no-mean-resize path.
    view_ids = add_mammo_grg_tokens(tokenizer, model, mean_resizing=False)
    model.config.view_token_ids = list(view_ids.values())
    state_dict = _normalize_non_lora_keys(_torch_load(non_lora_path))
    incompatibility = model.load_state_dict(state_dict, strict=False)
    if incompatibility.unexpected_keys:
        raise RuntimeError(
            "Stage-2 non-LoRA tensors were not consumed after key normalization: "
            f"{incompatibility.unexpected_keys}"
        )

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.merge_and_unload()
    model.eval()
    return tokenizer, model
