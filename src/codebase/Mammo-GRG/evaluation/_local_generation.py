"""Small, deterministic local text-generation helper shared by evaluators."""

import json
from typing import Any, Dict, Iterable, List, Sequence


class GenerationConfig:
    """Configuration for a locally loaded Transformers causal language model."""

    def __init__(
        self,
        model,
        batch_size=1,
        max_new_tokens=512,
        temperature=0.0,
        device_map="auto",
        dtype="auto",
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.dtype = dtype


def batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_json_object(text: str) -> Dict[str, Any]:
    """Return the first complete JSON object embedded in model output."""

    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output did not contain a valid JSON object")


class LocalGenerator:
    """Generate responses with a causal model loaded on the current machine."""

    def __init__(self, config: GenerationConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local evaluation requires torch and transformers."
            ) from exc

        dtype_by_name = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if config.dtype not in dtype_by_name:
            raise ValueError(f"Unsupported dtype: {config.dtype}")

        self.config = config
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            device_map=config.device_map,
            torch_dtype=dtype_by_name[config.dtype],
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _render_chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    def generate(self, system_prompt: str, user_prompts: Sequence[str]) -> List[str]:
        prompts = [self._render_chat(system_prompt, prompt) for prompt in user_prompts]
        outputs: List[str] = []
        input_device = next(self.model.parameters()).device

        for prompt_batch in batched(prompts, self.config.batch_size):
            encoded = self.tokenizer(
                list(prompt_batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
            prompt_length = encoded["input_ids"].shape[1]
            do_sample = self.config.temperature > 0

            generation_options = {
                "do_sample": do_sample,
                "max_new_tokens": self.config.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                generation_options["temperature"] = self.config.temperature

            with self.torch.inference_mode():
                generated = self.model.generate(**encoded, **generation_options)

            for sequence in generated:
                outputs.append(
                    self.tokenizer.decode(
                        sequence[prompt_length:],
                        skip_special_tokens=True,
                    ).strip()
                )
        return outputs
