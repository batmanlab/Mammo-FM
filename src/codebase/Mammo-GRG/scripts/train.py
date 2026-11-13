#!/usr/bin/env python3
"""Train Mammo-GRG stage 1 (projector) or stage 2 (LoRA)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

GRG_ROOT = Path(__file__).resolve().parents[1]
if str(GRG_ROOT) not in sys.path:
    sys.path.insert(0, str(GRG_ROOT))

import torch
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names

from mammo_grg.modeling_mammo_grg import MammoGRGConfig, MammoGRGForCausalLM
from mammo_grg.tokenization import add_mammo_grg_tokens
from mammo_grg.training_data import MammoGRGCollator, MammoGRGDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--model-base", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage1-checkpoint", help="Stage-1 directory or mm_projector.bin; required for stage 2")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--mm-projector-lr", type=float, default=None)
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def _gather_parameter(parameter):
    if hasattr(parameter, "ds_id"):
        from deepspeed import zero

        with zero.GatheredParameters([parameter]):
            return parameter.detach().cpu().clone()
    return parameter.detach().cpu().clone()


def _projector_state(model):
    return {
        key: _gather_parameter(value)
        for key, value in model.named_parameters()
        if "mm_projector" in key
    }


def _non_lora_trainables(model):
    return {
        key: _gather_parameter(value)
        for key, value in model.named_parameters()
        if "lora_" not in key and value.requires_grad
    }


class MammoGRGTrainer(Trainer):
    def __init__(self, *args, stage: int, projector_lr: float, **kwargs):
        self.grg_stage = stage
        self.projector_lr = projector_lr
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        decay_names = set(get_parameter_names(self.model, ALL_LAYERNORM_LAYERS))
        decay_names = {name for name in decay_names if "bias" not in name}
        projector_names = {
            name for name, _ in self.model.named_parameters() if "mm_projector" in name
        }
        groups = []
        for is_projector, use_decay in ((False, True), (False, False), (True, True), (True, False)):
            parameters = [
                parameter
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
                and (name in projector_names) == is_projector
                and (name in decay_names) == use_decay
            ]
            if not parameters:
                continue
            group = {
                "params": parameters,
                "weight_decay": self.args.weight_decay if use_decay else 0.0,
            }
            if is_projector:
                group["lr"] = self.projector_lr
            groups.append(group)
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
        if self.grg_stage == 1:
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.config.save_pretrained(checkpoint)
            torch.save(_projector_state(model), checkpoint / "mm_projector.bin")
            self.state.save_to_json(checkpoint / "trainer_state.json")
        else:
            super()._save_checkpoint(model, trial, metrics)
            torch.save(_non_lora_trainables(model), checkpoint / "non_lora_trainables.bin")

    def _save(self, output_dir=None, state_dict=None):
        if self.grg_stage == 1:
            output = Path(output_dir or self.args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            self.model.config.save_pretrained(output)
            torch.save(_projector_state(self.model), output / "mm_projector.bin")
        else:
            super()._save(output_dir, state_dict)


def _load_projector(model, checkpoint):
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "mm_projector.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 projector not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    expected = {key for key, _ in model.named_parameters() if "mm_projector" in key}
    supplied = {key for key in state if "mm_projector" in key}
    missing_projector = sorted(expected - supplied)
    unexpected_projector = sorted(supplied - expected)
    if missing_projector or unexpected_projector:
        raise RuntimeError(
            "Incomplete or incompatible stage-1 projector: "
            f"missing={missing_projector}, unexpected={unexpected_projector}"
        )
    model.load_state_dict(state, strict=False)


def _linear_module_names(model):
    names = set()
    for name, module in model.named_modules():
        if "mm_projector" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            names.add(name.split(".")[-1])
    names.discard("lm_head")
    return sorted(names)


def _resolve_learning_rates(stage, learning_rate=None, projector_lr=None):
    """Return source-faithful general and projector learning rates."""

    default_learning_rate = 1e-3 if stage == 1 else 2e-5
    resolved_learning_rate = (
        default_learning_rate if learning_rate is None else learning_rate
    )
    if projector_lr is None:
        # Stage 1 trains only the projector, so it must use the general 1e-3
        # source-launcher rate. Stage 2 used a dedicated 2e-5 projector rate.
        resolved_projector_lr = resolved_learning_rate if stage == 1 else 2e-5
    else:
        resolved_projector_lr = projector_lr
    return resolved_learning_rate, resolved_projector_lr


def main():
    args = parse_args()
    if args.stage == 2 and not args.stage1_checkpoint:
        raise ValueError("--stage1-checkpoint is required for stage 2")

    base_config = AutoConfig.from_pretrained(args.model_base)
    base_config_dict = base_config.to_dict()
    base_config_dict.pop("model_type", None)
    base_config_dict.pop("architectures", None)
    config = MammoGRGConfig(**base_config_dict)
    config.architectures = ["MammoGRGForCausalLM"]
    config.mm_vision_tower = "Mammo-FM_BatmanlabTrained_CLIP_last.tar"
    config.mm_projector_type = "attn_pool+mlp2x_gelu"
    config.mm_hidden_size = 2048
    config.mm_context_size = 2048
    config.mm_n_queries = 512
    config.mm_use_im_start_end = False
    config.mm_use_im_patch_token = False
    max_length = args.model_max_length or (4096 if args.stage == 1 else 128000)
    config.tokenizer_model_max_length = max_length

    model = MammoGRGForCausalLM.from_pretrained(
        args.model_base,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_base, use_fast=False, model_max_length=max_length, padding_side="right"
    )
    view_ids = add_mammo_grg_tokens(tokenizer, model)
    model.config.view_token_ids = list(view_ids.values())
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.stage == 1:
        model.requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(True)
    else:
        _load_projector(model, args.stage1_checkpoint)
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=_linear_module_names(model),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.base_model.model.get_model().mm_projector.requires_grad_(True)
        model.get_input_embeddings().requires_grad_(True)
        model.enable_input_require_grads()

    dataset = MammoGRGDataset(
        args.data_path, args.embedding_root, tokenizer, stage=args.stage
    )
    defaults = {
        1: dict(epochs=1.0, batch=6, save_steps=2000, save_limit=1, workers=6),
        2: dict(epochs=2.0, batch=3, save_steps=500, save_limit=20000, workers=10),
    }[args.stage]
    learning_rate, projector_lr = _resolve_learning_rates(
        args.stage, args.learning_rate, args.mm_projector_lr
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs or defaults["epochs"],
        per_device_train_batch_size=args.batch_size or defaults["batch"],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=args.save_steps or defaults["save_steps"],
        save_total_limit=args.save_total_limit or defaults["save_limit"],
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.workers if args.workers is not None else defaults["workers"],
        deepspeed=args.deepspeed,
        report_to=[] if args.report_to == "none" else [args.report_to],
    )
    trainer = MammoGRGTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MammoGRGCollator(tokenizer),
        tokenizer=tokenizer,
        stage=args.stage,
        projector_lr=projector_lr,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(args.output_dir)
    output = Path(args.output_dir)
    model.config.save_pretrained(output)
    if args.stage == 1:
        torch.save(_projector_state(model), output / "mm_projector.bin")
    else:
        torch.save(_non_lora_trainables(model), output / "non_lora_trainables.bin")


if __name__ == "__main__":
    main()
