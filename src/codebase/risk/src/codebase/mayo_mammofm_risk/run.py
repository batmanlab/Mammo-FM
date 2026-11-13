"""Train or evaluate corrected five-horizon Mammo-FM risk prediction on Mayo."""

import argparse
from datetime import timedelta
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import (
    DEFAULT_KNOWN_COLS,
    DEFAULT_LABEL_COLS,
    YEARS,
    LegacyMayoFiveYearDataset,
    ManifestColumns,
    compute_pos_weights,
)
from .losses import binary_kl_from_probs, masked_weighted_bce
from .model import MammoFMFiveYearRiskModel, compatible_model_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Five-head Mammo-FM Stage 1 distillation and Stage 2 Mayo outcome training"
    )
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--csv", required=True, help="Image-level Mayo risk manifest")
    parser.add_argument("--mammo-checkpoint", required=True, help="Mayo Mammo-FM checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", help="Model checkpoint used by --mode predict")
    parser.add_argument("--stage1-checkpoint", help="Fold-matched best Stage-1 checkpoint")
    parser.add_argument(
        "--stage2-init", choices=["stage1", "mayo"], default="stage1",
        help="Initialize Stage 2 from Stage 1, or skip distillation for the S2-only ablation",
    )

    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="valid")
    parser.add_argument(
        "--predict-split", default="valid",
        help="Existing value in --split-col to predict; no split is generated or reassigned",
    )

    parser.add_argument("--image-col", default="file")
    parser.add_argument("--exam-col", default="AccessionNumber")
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--laterality-col", default="laterality")
    parser.add_argument("--view-col", default="view")
    parser.add_argument("--cancer-registry-col", default="cancer_registry")
    parser.add_argument("--years-to-followup-col", default="years_to_last_followup")
    parser.add_argument("--label-cols", nargs=5, default=list(DEFAULT_LABEL_COLS))
    parser.add_argument("--known-cols", nargs=5, default=list(DEFAULT_KNOWN_COLS))

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Non-master ranks wait while rank 0 validates the DICOM dataset. Give
    # that intentional wait enough room on slower shared filesystems.
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(hours=2)
    )
    return True, dist.get_rank(), local_rank, torch.device(f"cuda:{local_rank}")


def is_master(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def load_state(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = compatible_model_state(state)
    model.load_state_dict(state, strict=True)
    return checkpoint


def make_dataset(image_frame, args):
    """Wrap the collaborator's existing dataset without changing its image pipeline."""
    columns = ManifestColumns(
        image=args.image_col,
        exam=args.exam_col,
        patient=args.patient_col,
        laterality=args.laterality_col,
        view=args.view_col,
        split=args.split_col.format(fold=args.fold),
        cancer_registry=args.cancer_registry_col,
        years_to_followup=args.years_to_followup_col,
        labels=tuple(args.label_cols),
        known=tuple(args.known_cols),
    )
    return LegacyMayoFiveYearDataset(
        image_frame,
        columns=columns,
    )


def make_loader(dataset, args, shuffle=False, sampler=None, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=args.num_workers > 0,
    )


def batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def stage_loss(stage, logits, batch, pos_weight):
    if stage == "stage1":
        return binary_kl_from_probs(logits, batch["teacher_probs"])
    return masked_weighted_bce(logits, batch["labels"], batch["label_mask"], pos_weight)


def cosine_warmup_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs):
    total_steps = max(1, int(steps_per_epoch) * int(epochs))
    warmup_steps = max(0, int(float(warmup_epochs) * int(steps_per_epoch)))
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, scaler, stage, pos_weight, device, rank):
    model.train()
    total_loss = torch.zeros(2, dtype=torch.float64, device=device)
    iterator = tqdm(loader, desc=f"{stage} train", leave=False) if is_master(rank) else loader
    amp_enabled = scaler.is_enabled()
    for batch in iterator:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(
                batch["images"], batch["time_seq"], batch["view_seq"], batch["side_seq"]
            )
            loss = stage_loss(stage, logits, batch, pos_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch_weight = (
            float(batch["label_mask"].sum().item())
            if stage == "stage2" else float(batch["images"].shape[0] * 5)
        )
        total_loss[0] += float(loss.detach()) * batch_weight
        total_loss[1] += batch_weight
        if is_master(rank):
            iterator.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return float((total_loss[0] / total_loss[1].clamp_min(1.0)).item())


@torch.no_grad()
def evaluate(model, loader, stage, pos_weight, device, amp_enabled=True, description="validation"):
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    predictions, labels, masks = [], [], []
    for batch in tqdm(loader, desc=description, leave=False):
        batch = batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
            logits = model(
                batch["images"], batch["time_seq"], batch["view_seq"], batch["side_seq"]
            )
            loss = stage_loss(stage, logits, batch, pos_weight)
        batch_weight = (
            float(batch["label_mask"].sum().item())
            if stage == "stage2" else float(batch["images"].shape[0] * 5)
        )
        loss_sum += float(loss) * batch_weight
        weight_sum += batch_weight
        predictions.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        masks.append(batch["label_mask"].cpu().numpy())
    return (
        loss_sum / max(weight_sum, 1.0),
        np.concatenate(predictions),
        np.concatenate(labels),
        np.concatenate(masks),
    )


def metric_rows(predictions, labels, masks, model_name=None):
    rows = []
    for index, year in enumerate(YEARS):
        known = masks[:, index].astype(bool)
        truth = labels[known, index]
        scores = predictions[known, index]
        row = {"year": year, "n": int(known.sum()), "positives": int(truth.sum())}
        if model_name is not None:
            row["model"] = model_name
        if np.unique(truth).size == 2:
            row["auroc"] = float(roc_auc_score(truth, scores))
        else:
            row["auroc"] = np.nan
        rows.append(row)
    return rows


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, stage, args, pos_weight):
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "stage": stage,
        "fold": int(args.fold),
        "pos_weight": pos_weight.detach().cpu(),
        "config": vars(args),
    }
    torch.save(payload, path)


def train(args, splits, distributed, rank, local_rank, device):
    output_dir = Path(args.output_dir)
    if is_master(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # The adapter delegates image loading/preprocessing and incomplete-exam handling to
    # the collaborator's Mammo_CLIPMetadataset.  The split frames themselves are only
    # exact filters of the existing CSV split values.
    train_dataset = make_dataset(splits["train"], args)
    val_dataset = make_dataset(splits["val"], args)
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if distributed else None
    )
    train_loader = make_loader(
        train_dataset, args, shuffle=not distributed, sampler=train_sampler, drop_last=True
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training split is smaller than one per-rank batch. Lower --batch-size for this run."
        )
    val_loader = make_loader(val_dataset, args, shuffle=False) if is_master(rank) else None

    pos_weight = (
        compute_pos_weights(train_dataset.exams).to(device)
        if args.stage == "stage2" else torch.ones(5, dtype=torch.float32, device=device)
    )
    model = MammoFMFiveYearRiskModel(args.mammo_checkpoint, freeze_encoder=True)
    if args.stage == "stage2" and args.stage2_init == "stage1":
        if not args.stage1_checkpoint:
            raise ValueError("Stage 2 with --stage2-init stage1 requires --stage1-checkpoint")
        stage1_payload = load_state(model, args.stage1_checkpoint)
        if isinstance(stage1_payload, dict) and stage1_payload.get("stage") not in (None, "stage1"):
            raise ValueError("The supplied Stage-1 checkpoint is not marked as Stage 1")
        if isinstance(stage1_payload, dict) and stage1_payload.get("fold") not in (None, args.fold):
            raise ValueError(
                f"Stage-1 checkpoint fold {stage1_payload.get('fold')} does not match fold {args.fold}"
            )
    model.to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_warmup_scheduler(
        optimizer, len(train_loader), args.epochs, args.warmup_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type == "cuda"))
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, args.stage, pos_weight, device, rank
        )
        if distributed:
            dist.barrier()
        if is_master(rank):
            val_loss, predictions, labels, masks = evaluate(
                unwrap_model(model), val_loader, args.stage, pos_weight, device,
                amp_enabled=not args.no_amp,
            )
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            for metric in metric_rows(predictions, labels, masks):
                row[f"yr{metric['year']}_auroc"] = metric["auroc"]
            history.append(row)
            save_checkpoint(
                output_dir / "latest.pt", model, optimizer, scheduler, epoch, val_loss,
                args.stage, args, pos_weight,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, scheduler, epoch, val_loss,
                    args.stage, args, pos_weight,
                )
            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} best={best_loss:.6f}"
            )
        if distributed:
            dist.barrier()


@torch.no_grad()
def predict(args, image_frame, device):
    if not args.checkpoint:
        raise ValueError("--mode predict requires --checkpoint")
    model = MammoFMFiveYearRiskModel(args.mammo_checkpoint, freeze_encoder=True).to(device)
    checkpoint = load_state(model, args.checkpoint)
    checkpoint_stage = checkpoint.get("stage") if isinstance(checkpoint, dict) else None
    if checkpoint_stage not in (None, args.stage):
        raise ValueError(
            f"Checkpoint stage {checkpoint_stage!r} does not match requested stage {args.stage!r}"
        )
    if isinstance(checkpoint, dict) and checkpoint.get("fold") not in (None, args.fold):
        raise ValueError(
            f"Checkpoint fold {checkpoint.get('fold')} does not match requested fold {args.fold}"
        )
    if isinstance(checkpoint, dict) and args.stage == "stage2":
        saved_config = checkpoint.get("config", {})
        saved_init = saved_config.get("stage2_init") if isinstance(saved_config, dict) else None
        if saved_init not in (None, args.stage2_init):
            raise ValueError(
                f"Checkpoint Stage-2 initialization {saved_init!r} does not match "
                f"requested {args.stage2_init!r}"
            )
    dataset = make_dataset(image_frame, args)
    exams = dataset.exams
    predict_split = args.predict_split.format(fold=args.fold)
    loader = make_loader(dataset, args, shuffle=False)
    if isinstance(checkpoint, dict) and "pos_weight" in checkpoint:
        pos_weight = torch.as_tensor(checkpoint["pos_weight"], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.ones(5, dtype=torch.float32, device=device)
    loss, predictions, labels, masks = evaluate(
        model, loader, args.stage, pos_weight, device,
        amp_enabled=not args.no_amp,
        description=predict_split,
    )

    result = exams[["patient_id", "exam_id", "split"]].copy()
    fold_column = "test_fold" if str(predict_split).startswith("test") else "fold"
    result.insert(2, fold_column, args.fold)
    result["training_stage"] = args.stage
    result["stage2_init"] = args.stage2_init if args.stage == "stage2" else "not_applicable"
    result["teacher_source"] = "probability"
    for index, year in enumerate(YEARS):
        result[f"cancer{year}yr_updated"] = labels[:, index].astype(int)
        result[f"known_yr{year}"] = masks[:, index].astype(int)
        result[f"mirai_risk_yr{year}"] = exams[f"teacher_risk_yr{year}"].to_numpy()
        result[f"mammo_fm_risk_yr{year}"] = predictions[:, index]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_fold{args.fold}_{args.stage}_{predict_split}.csv"
    result.to_csv(output_path, index=False)
    mirai_predictions = np.column_stack(
        [exams[f"teacher_risk_yr{year}"].to_numpy(dtype=float) for year in YEARS]
    )
    metrics = metric_rows(predictions, labels, masks, model_name="Mammo-FM")
    metrics += metric_rows(mirai_predictions, labels, masks, model_name="MIRAI")
    pd.DataFrame(metrics).to_csv(
        output_dir / f"metrics_fold{args.fold}_{args.stage}_{predict_split}.csv", index=False
    )
    print(f"{predict_split} loss: {loss:.6f}")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Predictions saved to {output_path}")


def load_split_frames(args):
    """Read the Mayo CSV and apply only its existing split assignments.

    This intentionally does not generate folds, move patients, drop duplicate views, or
    reinterpret paths.  Those dataset behaviors remain owned by the collaborator code.
    A ``{fold}`` placeholder is supported solely for CSVs that already contain named
    fold-specific split columns/values.
    """
    split_column = args.split_col.format(fold=args.fold)
    # Match the collaborator's original training code: do not override pandas'
    # CSV dtype inference or rewrite identifier/path columns.
    image_frame = pd.read_csv(args.csv)
    if split_column not in image_frame.columns:
        raise ValueError(f"Missing split column {split_column!r} in {args.csv}")

    requested = (
        {"train": args.train_split, "val": args.val_split}
        if args.mode == "train" else {"predict": args.predict_split}
    )
    split_frames = {}
    for role, split_value in requested.items():
        resolved_value = split_value.format(fold=args.fold)
        selected = image_frame.loc[image_frame[split_column].eq(resolved_value)].copy()
        if selected.empty:
            available = sorted(image_frame[split_column].dropna().astype(str).unique().tolist())
            raise ValueError(
                f"No rows have {split_column}={resolved_value!r}; existing values: {available}"
            )
        split_frames[role] = selected
    return split_frames


def main():
    args = parse_args()
    set_seed(args.seed)
    distributed, rank, local_rank, device = setup_distributed()
    try:
        splits = load_split_frames(args)
        if is_master(rank):
            summary = {name: int(frame[args.exam_col].astype(str).nunique()) for name, frame in splits.items()}
            print(f"CSV exam counts before the existing dataset's completeness filter: {summary}")
            print("MIRAI teacher source: collaborator dataset risk probabilities")
            print(f"Device: {device}; distributed={distributed}")
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            config_path = Path(args.output_dir) / f"resolved_config_{args.mode}_{args.stage}.json"
            with open(config_path, "w") as handle:
                json.dump(vars(args), handle, indent=2, sort_keys=True)
        if args.mode == "train":
            train(args, splits, distributed, rank, local_rank, device)
        else:
            if distributed:
                raise ValueError("Prediction is single-process; invoke python rather than torchrun")
            predict(args, splits["predict"], device)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
