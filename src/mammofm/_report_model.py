"""Local four-view report inference using the released Mammo-GRG models.

The Stage-2 weights already include Stage-1 projector training. A separate,
unadapted Llama model performs Stage-3 refinement; preliminary text is never
printed or returned by the public API.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import json
from pathlib import Path
from typing import Dict, List

from ._errors import CheckpointError


VIEW_ORDER = ("LMLO", "LCC", "RMLO", "RCC")
FOUR_VIEWS = {("L", "MLO"): "LMLO", ("L", "CC"): "LCC", ("R", "MLO"): "RMLO", ("R", "CC"): "RCC"}
ZS_KEYS = {
    "mass": "mass_pred",
    "suspicious_calcification": "suspicious calcification_pred",
    "asymmetry": "asymmetry_pred",
}


def _non_lora_shapes():
    prefix = "model.mm_projector."
    result = {
        "model.embed_tokens.weight": (128264, 4096),
        prefix + "view_positional_embeddings": (4, 4096),
        prefix + "attn_pool.query": (512, 2048),
        prefix + "attn_pool.to_q.weight": (2048, 2048),
        prefix + "attn_pool.to_kv.weight": (4096, 2048),
        prefix + "attn_pool.to_out.weight": (2048, 2048),
        prefix + "proj.0.weight": (4096, 2048),
        prefix + "proj.0.bias": (4096,),
        prefix + "proj.2.weight": (4096, 4096),
        prefix + "proj.2.bias": (4096,),
    }
    for layer in ("attn_pool.ln_k", "attn_pool.ln_q", "ln"):
        for kind in ("weight", "bias"):
            result[prefix + layer + "." + kind] = (2048,)
    return result


def _adapter_shapes():
    shapes = {
        "base_model.model.lesion_head.lora_A.weight": (128, 4096),
        "base_model.model.lesion_head.lora_B.weight": (4, 128),
    }
    layers = {
        "self_attn.q_proj": (4096, 4096),
        "self_attn.k_proj": (4096, 1024),
        "self_attn.v_proj": (4096, 1024),
        "self_attn.o_proj": (4096, 4096),
        "mlp.gate_proj": (4096, 14336),
        "mlp.up_proj": (4096, 14336),
        "mlp.down_proj": (14336, 4096),
    }
    for index in range(32):
        for name, (input_size, output_size) in layers.items():
            prefix = f"base_model.model.model.layers.{index}.{name}."
            shapes[prefix + "lora_A.weight"] = (128, input_size)
            shapes[prefix + "lora_B.weight"] = (output_size, 128)
    return shapes


def _normalize_non_lora_keys(state):
    if not isinstance(state, dict):
        raise CheckpointError("Stage-2 non-LoRA checkpoint must be a tensor dictionary.")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise CheckpointError("Stage-2 tensor names must be strings.")
        key = key.removeprefix("base_model.")
        if key.startswith("model.model."):
            key = key.removeprefix("model.")
        if key in normalized:
            raise CheckpointError("Duplicate normalized Stage-2 tensor name.")
        normalized[key] = value
    return normalized


def _validate_non_lora(state):
    import torch

    expected = _non_lora_shapes()
    if set(state) != set(expected):
        raise CheckpointError(
            "Stage-2 non-LoRA tensor names do not match the 16 released tensors: "
            f"missing={sorted(set(expected) - set(state))}, "
            f"unexpected={sorted(set(state) - set(expected))}"
        )
    for key, shape in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
            raise CheckpointError(f"Incompatible Stage-2 tensor: {key}; expected floating {shape}.")


def _validate_adapter(checkpoint):
    from safetensors import safe_open

    config = json.loads((checkpoint / "adapter_config.json").read_text())
    required = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 128,
        "lora_alpha": 256, "bias": "none", "use_dora": False, "use_rslora": False,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise CheckpointError("Stage-2 LoRA configuration differs from the released model.")
    if set(config.get("target_modules", [])) != {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lesion_head"
    }:
        raise CheckpointError("Stage-2 LoRA target modules differ from the released model.")
    if any(config.get(key) for key in (
        "rank_pattern", "alpha_pattern", "modules_to_save", "layer_replication",
        "layers_pattern", "layers_to_transform", "exclude_modules", "auto_mapping",
    )):
        raise CheckpointError("Unsupported Stage-2 adapter topology.")
    expected = _adapter_shapes()
    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise CheckpointError("Stage-2 adapter does not contain the 450 expected LoRA tensors.")
        for key, shape in expected.items():
            tensor = handle.get_slice(key)
            if tuple(tensor.get_shape()) != shape or tensor.get_dtype() not in ("F16", "BF16", "F32"):
                raise CheckpointError(f"Incompatible Stage-2 adapter tensor: {key}.")


def _validate_tokenizer(tokenizer):
    tokens = ("<pad>", "<multiple_choice>", "<long_answer>", "<report_generation>", "<LMLO>", "<LCC>", "<RMLO>", "<RCC>")
    if len(tokenizer) != 128264 or tokenizer.pad_token_id != 128256:
        raise CheckpointError("Stage-2 tokenizer vocabulary or padding ID is incompatible.")
    if [tokenizer.convert_tokens_to_ids(token) for token in tokens] != list(range(128256, 128264)):
        raise CheckpointError("Stage-2 directive/view token IDs do not match training.")


def _validate_generation(max_new_tokens, seed):
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1.")


@contextmanager
def _seeded(seed, device):
    import torch

    target = torch.device(device)
    devices = []
    if target.type == "cuda":
        devices = [torch.cuda.current_device() if target.index is None else target.index]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield


def _cleanup(device):
    import torch

    gc.collect()
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _load_stage2(checkpoint_dir, base_model_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer
    from ._vendor.grg.modeling_mammo_grg import MammoGRGConfig, MammoGRGForCausalLM

    checkpoint = Path(checkpoint_dir)
    required = ("config.json", "adapter_config.json", "adapter_model.safetensors",
                "non_lora_trainables.bin", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    if not checkpoint.is_dir() or any(not (checkpoint / name).is_file() for name in required):
        raise CheckpointError("Stage-2 checkpoint directory is incomplete.")
    _validate_adapter(checkpoint)
    state = _normalize_non_lora_keys(torch.load(
        checkpoint / "non_lora_trainables.bin", map_location="cpu", weights_only=True, mmap=True
    ))
    _validate_non_lora(state)
    config = MammoGRGConfig.from_pretrained(checkpoint, local_files_only=True)
    base_config = AutoConfig.from_pretrained(base_model_path, local_files_only=True, trust_remote_code=False)
    for key, expected in {
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 14336,
    }.items():
        if getattr(config, key, None) != expected or getattr(base_config, key, None) != expected:
            raise CheckpointError(f"Report model has incompatible {key}.")
    if base_config.model_type != "llama" or base_config.vocab_size != 128256:
        raise CheckpointError("Report model requires the released Llama-3.1-8B base vocabulary.")
    for key, expected in {
        "vocab_size": 128264, "pad_token_id": 128256, "mm_hidden_size": 2048,
        "mm_context_size": 2048, "mm_n_queries": 512, "mm_projector_type": "attn_pool+mlp2x_gelu",
    }.items():
        if getattr(config, key, None) != expected:
            raise CheckpointError(f"Stage-2 checkpoint has incompatible {key}.")
    if hasattr(base_config, "quantization_config") or hasattr(config, "quantization_config"):
        raise CheckpointError("A quantized base is not supported by this full-precision report loader.")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=False, local_files_only=True, trust_remote_code=False
    )
    _validate_tokenizer(tokenizer)
    # The expanded padding ID is out of range in the original 128256-row base.
    config.vocab_size = base_config.vocab_size
    config.pad_token_id = None
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.float16
    model, loading_info = MammoGRGForCausalLM.from_pretrained(
        base_model_path, config=config, torch_dtype=dtype, device_map={"": str(device)},
        low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="eager", output_loading_info=True,
    )
    added_keys = (set(_non_lora_shapes()) - {"model.embed_tokens.weight"}) | {
        "lesion_head.weight", "lesion_head.bias"
    }
    if (set(loading_info.get("missing_keys", ())) != added_keys
            or loading_info.get("unexpected_keys") or loading_info.get("mismatched_keys")
            or loading_info.get("error_msgs")):
        raise CheckpointError("Base Llama weights did not load exactly into the report model.")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.view_token_ids = [tokenizer.convert_tokens_to_ids(f"<{view}>") for view in VIEW_ORDER]
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = set(model.state_dict()) - set(state)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected_missing:
        raise CheckpointError("Stage-2 non-LoRA tensors were not loaded exactly.")
    del state
    adapted = PeftModel.from_pretrained(
        model, checkpoint, is_trainable=False, local_files_only=True, autocast_adapter_dtype=True
    )
    model = adapted.merge_and_unload(safe_merge=True)
    model.eval()
    return tokenizer, model


def _preliminary_prompt():
    from ._vendor.grg.constants import DEFAULT_IMAGE_TOKEN, TOKEN_FOR_REPORT_GENERATION, VIEW_TOKENS
    from ._vendor.grg.conversation import ASSISTANT_ROLE, USER_ROLE, new_conversation

    blocks = " ".join(f"{token}{DEFAULT_IMAGE_TOKEN}" for token in VIEW_TOKENS)
    question = f"{blocks}\n{TOKEN_FOR_REPORT_GENERATION} Generate a complete mammography report."
    conversation = new_conversation()
    conversation.append(USER_ROLE, question)
    conversation.append(ASSISTANT_ROLE, None)
    return conversation.prompt()


def generate_preliminary(embeddings, checkpoint_dir, base_model_path, device, max_new_tokens=512, seed=0):
    """Generate one preliminary report, releasing the loaded Stage-2 LM afterward."""
    import torch
    from ._vendor.grg.tokenization import tokenizer_image_token

    _validate_generation(max_new_tokens, seed)
    if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != (1, 4, 1, 2048, 48, 29):
        raise ValueError("Report embeddings must have shape [1,4,1,2048,48,29] in LMLO,LCC,RMLO,RCC order.")
    if not embeddings.is_floating_point() or not torch.isfinite(embeddings).all().item():
        raise ValueError("Report embeddings must contain finite floating-point values.")
    model = tokenizer = input_ids = prepared = output_ids = None
    try:
        with _seeded(seed, device), torch.inference_mode():
            tokenizer, model = _load_stage2(checkpoint_dir, base_model_path, device)
            input_ids = tokenizer_image_token(_preliminary_prompt(), tokenizer, return_tensors="pt").unsqueeze(0)
            input_ids = input_ids.to(device)
            prepared = embeddings.to(device=device, dtype=model.dtype)
            output_ids = model.generate(
                inputs=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id),
                images=prepared, do_sample=False, max_new_tokens=max_new_tokens,
                use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            result = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if not result:
                raise RuntimeError("Stage-2 report generation returned empty text.")
            return result
    finally:
        model = tokenizer = input_ids = prepared = output_ids = None
        _cleanup(device)


def build_view_and_side_summaries(findings):
    """Validate all four views, then aggregate binary findings by breast and exam."""
    if not isinstance(findings, (list, tuple)) or len(findings) != 4:
        raise ValueError("Grounding requires exactly four view findings.")
    view_dict = {}
    side_presence = {finding: {"L": 0, "R": 0} for finding in ZS_KEYS}
    for item in findings:
        if not isinstance(item, dict):
            raise ValueError("Each grounding finding must be a dictionary.")
        lat, view = item.get("image_laterality"), item.get("image_view_type")
        name = FOUR_VIEWS.get((lat, view))
        if name is None or name in view_dict:
            raise ValueError("Grounding findings must contain one each of LMLO, LCC, RMLO, RCC.")
        view_dict[name] = {}
        for finding, key in ZS_KEYS.items():
            value = item.get(key)
            if not isinstance(value, (int, bool)) or value not in (0, 1):
                raise ValueError(f"Grounding finding {key} must be binary 0 or 1.")
            view_dict[name][finding] = int(value)
            side_presence[finding][lat] |= int(value)
    max_summary = {finding: max(presence.values()) for finding, presence in side_presence.items()}
    return view_dict, max_summary, side_presence


SYSTEM_PROMPT = (
    "You are a radiology report assistant for screening mammography.\n"
    "Produce a FINAL REPORT with exactly two sections:\n"
    "1) Findings\n"
    "2) Impression\n\n"
    "FOUNDATION:\n"
    "- Treat the PRELIMINARY REPORT as the base text. Preserve its clinically relevant content, density description, laterality, locations (e.g., upper outer quadrant), and recommendations.\n"
    "- You MUST include (carry forward) all non-contradicted statements from the preliminary report.\n"
    "- Reconcile contradictions using the rules below.\n\n"
    "RECONCILIATION RULES (VERY IMPORTANT):\n"
    "1) If structured findings are POSITIVE for a category (mass, suspicious calcification, asymmetry), you MUST reflect that positivity in the final report:\n"
    "   - Remove or revise any preliminary 'no X' statements for that category.\n"
    "   - State laterality accurately: Left, Right, or 'bilaterally' ONLY if both sides are positive.\n"
    "2) If structured findings are NEGATIVE and the preliminary report already says 'no X', keep that negative statement.\n"
    "3) Do NOT add 'bilaterally' unless BOTH sides are positive.\n"
    "4) BI-RADS POLICY (SCREENING): You must output BI-RADS in {0, 1, 2} ONLY—never 3–6 for screening exams.\n"
    "   - If any structured finding is positive/indeterminate (i.e., additional diagnostic work-up is needed), assign BI-RADS 0.\n"
    "   - If there are no findings, assign BI-RADS 1.\n"
    "   - If only benign/non-suspicious findings are present, assign BI-RADS 2.\n"
    "   - If the preliminary BI-RADS is provided, keep it if consistent with these rules; otherwise correct it.\n"
    "   - IMPORTANT: If  asymmetry is POSITIVE in the structured findings, override any preliminary BI-RADS and assign BI-RADS 0.\n"
    "5) Do NOT mention external models or the word 'classifier'.\n"
    "6) Use standard terminology (CC, MLO; laterality) and be concise, clinically appropriate.\n"
    "7) Preserve the overall template and sentence style of the preliminary report while outputting ONLY the two sections."
)


def build_classifier_note(max_summary: Dict[str, int], side_presence: Dict[str, Dict[str, int]]) -> str:
    """
    Human-readable bullets:
      - If both sides positive -> "...: bilaterally."
      - If only left -> "...: Left breast."
      - If only right -> "...: Right breast."
      - If none positive -> 'No positive ML findings across views.'
    """
    label_map = {
        "mass": "Suspicious Mass",
        "suspicious_calcification": "Suspicious Calcification",
        "asymmetry": "Asymmetry",
    }

    lines: List[str] = []
    for finding, label in label_map.items():
        if max_summary.get(finding, 0) != 1:
            continue
        left_pos = side_presence[finding]["L"] == 1
        right_pos = side_presence[finding]["R"] == 1
        if left_pos and right_pos:
            where = "bilaterally."
        elif left_pos:
            where = "Left breast."
        elif right_pos:
            where = "Right breast."
        else:
            where = "side unspecified."
        lines.append(f"- {label}: {where}")

    return "\n".join(lines) if lines else "- No positive ML findings across views."


def build_user_prompt(prelim_report: str, view_dict: Dict, max_summary: Dict, side_presence: Dict,
                      extra_context: str = None) -> str:
    classifier_bullets = build_classifier_note(max_summary, side_presence)
    compact_json = json.dumps(max_summary, ensure_ascii=False)
    return (
            "Preliminary report (from report generator):\n"
            f"{prelim_report.strip()}\n\n"
            "Structured ML findings (max across RCC/RMLO/LCC/LMLO):\n"
            f"{classifier_bullets}\n\n"
            "Max-summary JSON (do not quote or mention 'classifier' in the final text):\n"
            f"{compact_json}\n\n"
            "Rewrite the report into the required two sections ONLY (Findings, Impression). "
            "Follow the template and structure of the preliminary report."
            + (f"\n\nAdditional context: {extra_context}" if extra_context else "")
    )


def generate_grounded(preliminary, findings, base_model_path, device, max_new_tokens=320, seed=0):
    """Refine preliminary text with image-classifier findings; never call an API."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _validate_generation(max_new_tokens, seed)
    if not isinstance(preliminary, str) or not preliminary.strip():
        raise ValueError("A nonempty preliminary report is required for Stage-3 grounding.")
    view_dict, summary, sides = build_view_and_side_summaries(findings)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(preliminary, view_dict, summary, sides)},
    ]
    model = tokenizer = inputs = output_ids = None
    try:
        with _seeded(seed, device), torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, use_fast=True, local_files_only=True, trust_remote_code=False
            )
            dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                base_model_path, torch_dtype=dtype, device_map={"": str(device)},
                low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, attn_implementation="eager", output_loading_info=True,
            )
            if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise CheckpointError("Base Llama weights did not load exactly for Stage-3 grounding.")
            model.eval()
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.3, top_p=0.9, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = output_ids[0][inputs["input_ids"].shape[-1]:]
            result = tokenizer.decode(generated, skip_special_tokens=True).strip()
            del generated
            if not result:
                raise RuntimeError("Stage-3 grounding returned empty text.")
            return result
    finally:
        model = tokenizer = inputs = output_ids = None
        _cleanup(device)
