import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models.breast_clip_classifier import BreastClipClassifier
from Datasets.dataset_utils import get_dataloader
from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from utils import AverageMeter, seed_all


def _is_density(args):
    return args.label.lower() == "density"


def _prepare_task_labels(args, df):
    """Exclude unknown labels, retaining source folds and clinical endpoint semantics."""
    dataset = args.dataset.lower()
    df = df.copy()
    original_count = len(df)
    if _is_density(args):
        if dataset == "lumina":
            metadata_path = getattr(args, "density_metadata", None)
            if not metadata_path:
                raise ValueError("LUMINA density requires --density-metadata; cancer metadata has no density")
            keys = ["Cancer", "patient_id"]
            metadata = pd.read_csv(metadata_path, usecols=keys + ["density"])
            if metadata[keys].isna().any().any() or metadata.duplicated(keys).any():
                raise ValueError("LUMINA density requires unique nonmissing (Cancer, patient_id) keys")
            # Numeric IDs are reused between the benign and malignant source cohorts.
            # Import density only, never the cancer labels/folds of collapsed metadata.
            df = df.merge(metadata, on=keys, how="left", validate="many_to_one", sort=False)
            if df["density"].isna().any():
                raise ValueError("LUMINA density metadata does not cover every image's cohort/patient key")
        elif "density" not in df:
            raise KeyError(f"{args.dataset} CSV does not contain density labels")
        elif dataset == "rsna" and not pd.api.types.is_numeric_dtype(df["density"]):
            values = df["density"].astype("string").str.strip().str.upper()
            unknown = values[values.notna() & ~values.isin(list("ABCD"))]
            if len(unknown):
                raise ValueError(f"Unknown RSNA density labels: {sorted(unknown.unique().tolist())}")
            df["density"] = values.map(dict(zip("ABCD", range(4))))
        df = df[df["density"].notna()].copy()
        numeric = pd.to_numeric(df["density"], errors="raise")
        if not numeric.isin(range(4)).all():
            raise ValueError("Density labels must map to A=0, B=1, C=2, D=3")
        df["density"] = numeric.astype(int)
        args.num_classes = 4
        if dataset == "vindr":
            keys = ["patient_id", "image_id"]
            for col in ("density", "laterality", "split"):
                if (df.groupby(keys)[col].nunique(dropna=False) > 1).any():
                    raise ValueError(f"Conflicting VinDr {col} values for the same image")
            df = df.drop_duplicates(keys).copy()
        print(f"Density cohort: {len(df)}/{original_count} images; A=0, B=1, C=2, D=3")
    elif args.label.lower() == "invasive":
        if dataset != "rsna":
            raise ValueError("Invasive versus non-invasive classification is defined only for RSNA")
        required = {"cancer", "invasive"}
        if not required.issubset(df.columns):
            raise KeyError("RSNA invasive classification requires cancer and invasive columns")
        df = df[(df["cancer"] == 1) & df["invasive"].notna()].copy()
        if not df["invasive"].isin([0, 1]).all():
            raise ValueError("Invasive labels must be 0 or 1 among cancer-positive cases")
        df["invasive"] = df["invasive"].astype(int)
        print(f"RSNA invasive cohort: {len(df)}/{original_count} images; cancer-positive only; "
              "0=non-invasive cancer, 1=invasive cancer")
    return df.reset_index(drop=True)


def _attach_predictions(args, frame, predictions):
    frame = frame.copy()
    if _is_density(args):
        if predictions.shape != (len(frame), 4):
            raise ValueError("Expected one four-class density probability vector per image")
        for index, name in enumerate("ABCD"):
            frame[f"prediction_{name}"] = predictions[:, index]
    else:
        frame["prediction"] = predictions
    return frame


def _loss_and_probabilities(args, criterion, logits, labels):
    if _is_density(args):
        return criterion(logits, labels.long().view(-1)), logits.softmax(dim=1)
    return criterion(logits.view(-1, 1), labels.float().view(-1, 1)), logits.view(-1).sigmoid()


def _split_group_column(args, df):
    """Return the patient/exam identifier used to enforce disjoint splits."""
    if args.dataset.lower() == "nlbreast":
        if "group_id" in df.columns:
            return "group_id"
        if "folder_name" in df.columns:
            return "folder_name"
    for column in ("patient_id", "exam_id", "study_id", "group_id", "folder_name"):
        if column in df.columns:
            return column
    raise KeyError(
        f"Cannot enforce disjoint splits for {args.dataset}: no patient or exam identifier found"
    )


def _assert_disjoint_splits(args):
    group_col = _split_group_column(args, args.df)
    groups = {
        "train": set(args.train_folds[group_col].astype(str)),
        "validation": set(args.valid_folds[group_col].astype(str)),
        "test": set(args.test_folds[group_col].astype(str)),
    }
    overlaps = {
        "train/validation": groups["train"] & groups["validation"],
        "train/test": groups["train"] & groups["test"],
        "validation/test": groups["validation"] & groups["test"],
    }
    leaking = {name: len(values) for name, values in overlaps.items() if values}
    if leaking:
        raise ValueError(f"Group leakage detected for {args.dataset}: {leaking}")


def _set_fold_splits(args, df):
    """Use fold 0 as test, the largest remaining fold as validation, and all others for training."""
    if "fold" not in df.columns:
        raise KeyError(f"{args.dataset} CSV does not contain a fold column")
    df = df.copy()
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    folds = sorted(df["fold"].unique().tolist())
    test_fold = int(args.cur_fold)
    validation_fold = max(folds)
    training_folds = [fold for fold in folds if fold not in {test_fold, validation_fold}]
    if test_fold not in folds or test_fold == validation_fold or not training_folds:
        raise ValueError(
            f"Invalid split for {args.dataset}: folds={folds}, test={test_fold}, "
            f"validation={validation_fold}"
        )
    args.df = df
    args.train_folds = df[df["fold"].isin(training_folds)].copy().reset_index(drop=True)
    args.valid_folds = df[df["fold"] == validation_fold].copy().reset_index(drop=True)
    args.test_folds = df[df["fold"] == test_fold].copy().reset_index(drop=True)
    _assert_disjoint_splits(args)
    print(
        f"Split for {args.dataset}: training={training_folds}, validation={validation_fold}, "
        f"test={test_fold}; rows={len(args.train_folds)}/{len(args.valid_folds)}/{len(args.test_folds)}"
    )


def _set_vindr_official_split(args, df):
    """Hold out the official test set and split training patients 80/20 for validation."""
    if "split" not in df.columns:
        raise KeyError("VinDr CSV does not contain the official 'split' column")
    group_col = _split_group_column(args, df)
    normalized = df["split"].astype(str).str.strip().str.lower()
    training_pool = df[normalized.isin(["train", "training"])].copy()
    test_df = df[normalized == "test"].copy()
    unknown = sorted(normalized[~normalized.isin(["train", "training", "test"])].unique())
    if unknown or training_pool.empty or test_df.empty:
        raise ValueError(f"Invalid VinDr official split; unexpected={unknown}")
    if set(training_pool[group_col].astype(str)) & set(test_df[group_col].astype(str)):
        raise ValueError("Patient leakage exists in the official VinDr training/test split")
    patient_targets = training_pool.groupby(group_col)[args.label].max()
    counts = patient_targets.value_counts()
    expected_classes = 4 if _is_density(args) else 2
    if len(counts) != expected_classes or int(counts.min()) < 2:
        raise ValueError(
            f"Cannot stratify VinDr on {args.label}; patient-level counts={counts.to_dict()}"
        )
    train_ids, valid_ids = train_test_split(
        patient_targets.index.to_numpy(),
        test_size=0.20,
        random_state=int(args.seed),
        shuffle=True,
        stratify=patient_targets.to_numpy(),
    )
    train_ids, valid_ids = set(train_ids), set(valid_ids)
    train_df = training_pool[training_pool[group_col].isin(train_ids)].copy()
    valid_df = training_pool[training_pool[group_col].isin(valid_ids)].copy()
    train_df["fold"], valid_df["fold"], test_df["fold"] = 1, 4, 0
    args.train_folds = train_df.reset_index(drop=True)
    args.valid_folds = valid_df.reset_index(drop=True)
    args.test_folds = test_df.reset_index(drop=True)
    args.df = pd.concat([args.train_folds, args.valid_folds, args.test_folds], ignore_index=True)
    _assert_disjoint_splits(args)
    print(
        "VinDr split: official test held out; official training patients stratified 80/20 "
        f"for train/validation; rows={len(args.train_folds)}/{len(args.valid_folds)}/{len(args.test_folds)}"
    )


def _laterality_from_lumina(path):
    stem = Path(str(path)).stem.upper()
    if stem.startswith("L"):
        return "L"
    if stem.startswith("R"):
        return "R"
    return "UNKNOWN"


def _aggregate_for_auroc(args, df):
    """Aggregate image predictions at the task's model-selection level."""
    dataset = args.dataset.lower()
    data = df.copy()
    if dataset == "grounding":
        from .grounding import aggregate
        return aggregate(args, data)
    if _is_density(args):
        if dataset == "lumina":
            data["_laterality"] = data["image_full_path"].map(_laterality_from_lumina)
            keys = ["Cancer", "patient_id", "_laterality"]
        elif dataset == "nlbreast":
            keys = [_split_group_column(args, data)]
        else:
            keys = ["patient_id", "LeftRight" if dataset == "cmmd" else "laterality"]
        if data[keys].isna().any().any():
            raise ValueError("Density aggregation requires nonmissing patient/breast identifiers")
        if (data.groupby(keys)[args.label].nunique() != 1).any():
            raise ValueError("Inconsistent density labels within a patient-breast")
        # Averaging the four-way softmax preserves the sum-to-one constraint.
        return data.groupby(keys, as_index=False).agg(
            **{args.label: (args.label, "first")},
            **{f"prediction_{name}": (f"prediction_{name}", "mean") for name in "ABCD"},
        )
    if dataset == "nlbreast":
        group_col = _split_group_column(args, data)
        return data.groupby(group_col, as_index=False).agg(
            **{args.label: (args.label, "max"), "prediction": ("prediction", "max")}
        )
    if dataset == "lumina":
        data["_laterality"] = data["image_full_path"].map(_laterality_from_lumina)
        return data.groupby(["Cancer", "patient_id", "_laterality"], as_index=False).agg(
            **{args.label: (args.label, "max"), "prediction": ("prediction", "max")}
        )
    if dataset == "cmmd":
        return data.groupby(["patient_id", "LeftRight"], as_index=False).agg(
            **{args.label: (args.label, "max"), "prediction": ("prediction", "mean")}
        )
    if dataset in {"rsna", "vindr"}:
        reducer = "mean" if dataset == "rsna" else "max"
        return data.groupby(["patient_id", "laterality"], as_index=False).agg(
            **{args.label: (args.label, "max"), "prediction": ("prediction", reducer)}
        )
    return data


def _auroc(args, predictions_df):
    evaluated = _aggregate_for_auroc(args, predictions_df)
    if _is_density(args):
        if set(evaluated[args.label].unique()) != set(range(4)):
            raise ValueError("Four-class density AUROC requires all A/B/C/D classes in each split")
        return roc_auc_score(
            evaluated[args.label].astype(int),
            evaluated[[f"prediction_{name}" for name in "ABCD"]].to_numpy(),
            labels=list(range(4)), multi_class="ovr", average="macro",
        )
    if evaluated[args.label].nunique() < 2:
        raise ValueError(f"AUROC is undefined for a single-class {args.dataset} split")
    return roc_auc_score(evaluated[args.label].astype(int), evaluated["prediction"])


def _binary_auroc_levels(args, predictions_df):
    """Describe unique-image, whole-patient/study, and model-selection AUROC."""
    if _is_density(args):
        raise ValueError("Binary AUROC reporting does not apply to density")
    data = predictions_df.copy()
    dataset = args.dataset.lower()
    if data.empty or not data[args.label].isin([0, 1]).all():
        raise ValueError("Binary AUROC requires nonempty, known zero/one labels")
    probabilities = data["prediction"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Binary AUROC requires finite probabilities in [0, 1]")

    patient_unit = "patient"
    current_unit = "patient_breast"
    current_reducer = "max"
    if dataset == "grounding":
        image_keys = ["source_dataset", "image_path"]
        patient_keys = ["source_dataset", "patient_id"]
        current_keys = ["source_dataset", "patient_id", "exam_id", "laterality"]
        patient_unit = "source_qualified_patient_or_study"
        current_unit = "source_qualified_exam_breast"
    elif dataset == "nlbreast":
        image_keys = ["image_path"]
        patient_keys = current_keys = [_split_group_column(args, data)]
        patient_unit = current_unit = "patient_or_study_group"
    elif dataset == "lumina":
        image_keys = ["Cancer", "patient_id", "image_full_path"]
        patient_keys = ["Cancer", "patient_id"]
        data["_laterality"] = data["image_full_path"].map(_laterality_from_lumina)
        current_keys = ["Cancer", "patient_id", "_laterality"]
    elif dataset == "cmmd":
        image_keys = ["patient_id", "dicom_name"]
        patient_keys = ["patient_id"]
        current_keys = ["patient_id", "LeftRight"]
        current_reducer = "mean"
    elif dataset in {"rsna", "vindr"}:
        image_keys = ["patient_id", "image_id"]
        patient_keys = ["patient_id"]
        current_keys = ["patient_id", "laterality"]
        if dataset == "rsna":
            current_reducer = "mean"
        else:
            patient_unit, current_unit = "study", "study_breast"
    else:
        raise ValueError(f"Unsupported binary AUROC dataset: {args.dataset}")
    keys = list(dict.fromkeys(image_keys + patient_keys + current_keys))
    missing = set(keys).difference(data.columns)
    if missing:
        raise KeyError(f"Binary AUROC grouping columns are missing: {sorted(missing)}")
    if data[keys].isna().any().any() or data[keys].astype(str).apply(
            lambda column: column.str.strip().eq("").any()).any():
        raise ValueError("Binary AUROC grouping identifiers must be nonmissing")

    def pooled(group_keys):
        return data.groupby(group_keys, as_index=False).agg(
            **{args.label: (args.label, "max"), "prediction": ("prediction", "max")}
        )

    frames = {
        "image": (pooled(image_keys), image_keys, "image", "max"),
        "patient": (pooled(patient_keys), patient_keys, patient_unit, "max"),
        "current_grouping": (_aggregate_for_auroc(args, data), current_keys,
                             current_unit, current_reducer),
    }
    report = {}
    for level, (frame, group_keys, unit, reducer) in frames.items():
        defined = frame[args.label].nunique() == 2
        report[level] = {
            "auroc": float(roc_auc_score(frame[args.label].astype(int), frame["prediction"])) if defined else None,
            "n": len(frame), "positives": int(frame[args.label].sum()),
            "unit": unit, "grouping_columns": list(group_keys),
            "label_reduction": "max", "prediction_reduction": reducer,
            "undefined_reason": None if defined else "AUROC requires both positive and negative units",
        }
    report["image"]["input_rows"] = len(data)
    report["image"]["duplicate_rows_collapsed"] = len(data) - report["image"]["n"]
    return report


def _print_binary_auroc_levels(prefix, report):
    """Print supplementary levels without changing the model-selection log line."""
    for level in ("image", "patient"):
        entry = report[level]
        value = "undefined (single class)" if entry["auroc"] is None else f"{entry['auroc']:.6f}"
        print(f"{prefix} {level} AUROC: {value}; units={entry['n']} ({entry['unit']})")


def _save_binary_auroc_metrics(args, report, output_path, split_name):
    payload = {
        "dataset": args.dataset, "task": args.label, "split": split_name,
        "metric": "auroc", "checkpoint_selection_level": "current_grouping",
        "auroc_levels": report,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _save_predictions(args, frame, split_name):
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"seed_{args.seed}_fold{args.cur_fold}_{args.label}_{split_name}_predictions.csv"
    )
    frame.to_csv(output_path, index=False)
    print(f"Saved {split_name} predictions to {output_path}")
    if not _is_density(args):
        _save_binary_auroc_metrics(
            args, _binary_auroc_levels(args, frame),
            output_dir / f"seed_{args.seed}_fold{args.cur_fold}_{args.label}_{split_name}_metrics.json",
            split_name,
        )


def _prepare_splits(args):
    args.df = _prepare_task_labels(args, pd.read_csv(args.data_dir / args.csv_file))
    dataset = args.dataset.lower()
    if dataset == "grounding":
        from .grounding import prepare_splits
        prepare_splits(args, args.df)
        _assert_disjoint_splits(args)
    elif dataset == "lumina":
        required = {"patient_id", "image_full_path", "Cancer_bin", "fold"}
        missing = required.difference(args.df.columns)
        if missing:
            raise KeyError(f"LUMINA CSV missing required columns: {sorted(missing)}")
        if args.label.lower() == "cancer":
            args.df["cancer"] = args.df["Cancer_bin"].astype(int)
        _set_fold_splits(args, args.df)
    elif dataset == "vindr":
        if args.label.lower() == "cancer" and "cancer" not in args.df.columns:
            if "abnormal" not in args.df.columns:
                raise KeyError("VinDr cancer task requires the abnormal endpoint column")
            args.df["cancer"] = args.df["abnormal"].astype(int)
        elif args.label.lower() == "mass" and "mass" not in args.df.columns:
            args.df["mass"] = args.df["Mass"].astype(int)
        elif args.label.lower() == "calc" and "calc" not in args.df.columns:
            args.df["calc"] = args.df["Suspicious_Calcification"].astype(int)
        _set_vindr_official_split(args, args.df)
    elif dataset in {"rsna", "cmmd", "nlbreast"}:
        if dataset == "cmmd" and args.label.lower() == "calc" and "calc" not in args.df.columns:
            args.df["calc"] = args.df["calcification"].astype(int)
        _set_fold_splits(args, args.df)
    else:
        raise ValueError(f"Unsupported AUROC-only classification dataset: {args.dataset}")
    if args.label not in args.df.columns:
        raise KeyError(f"Label '{args.label}' is absent from the {args.dataset} CSV")
    expected = set(range(4)) if _is_density(args) else {0, 1}
    for name, frame in (("training", args.train_folds), ("validation", args.valid_folds),
                        ("test", args.test_folds)):
        if frame[args.label].isna().any() or set(frame[args.label].unique()) != expected:
            raise ValueError(f"{args.dataset}/{args.label} {name} split must contain classes {expected}")
    if args.label.lower() == "invasive":
        if (args.df.groupby(["patient_id", "laterality"])[args.label].nunique() > 1).any():
            raise ValueError("RSNA invasive label is inconsistent within a cancer-positive breast")


def do_experiments(args, device):
    if "efficientnetv2" in args.arch:
        args.model_base_name = "efficientv2_s"
    elif "efficientnet_b5_ns" in args.arch:
        args.model_base_name = "efficientnetb5"
    else:
        args.model_base_name = args.arch
    args.data_dir = Path(args.data_dir)
    test_outputs = []
    for fold in range(args.start_fold, args.n_folds):
        args.cur_fold = fold
        seed_all(args.seed)
        _prepare_splits(args)
        if args.inference_mode == "y":
            raise ValueError("Inference-only mode is not used by the LP/FT training jobs")
        test_outputs.append(train_loop(args, device))
    combined = pd.concat(test_outputs, ignore_index=True)
    combined_path = Path(args.output_path) / (
        f"seed_{args.seed}_n_folds_{args.n_folds}_{args.label}_test_outputs.csv"
    )
    combined.to_csv(combined_path, index=False)
    print(f"Combined held-out test AUROC: {_auroc(args, combined):.6f}")
    if not _is_density(args):
        combined_report = _binary_auroc_levels(args, combined)
        _print_binary_auroc_levels("Combined held-out test", combined_report)
        _save_binary_auroc_metrics(
            args, combined_report,
            Path(args.output_path) / f"seed_{args.seed}_n_folds_{args.n_folds}_{args.label}_test_metrics.json",
            "combined_test",
        )
    print(f"Saved combined test predictions to {combined_path}")


def train_loop(args, device):
    print(f"\n================== fold: {args.cur_fold} training ======================")
    positives = int((args.train_folds[args.label] == 1).sum())
    negatives = int((args.train_folds[args.label] == 0).sum())
    if not _is_density(args) and (positives == 0 or negatives == 0):
        raise ValueError(f"Training split for {args.dataset}/{args.label} must contain both classes")
    args.BCE_weights = {f"fold{args.cur_fold}": negatives / positives if positives else 1.0}
    if args.data_frac < 1.0:
        args.train_folds = args.train_folds.sample(
            frac=args.data_frac, random_state=args.seed, ignore_index=True
        )
    if args.clip_chk_pt_path is None:
        raise ValueError("A Mammo-FM checkpoint is required for LP/FT evaluation")
    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu")
    encoder_config = ckpt["config"]["model"]["image_encoder"]
    if encoder_config["model_type"] == "swin":
        args.image_encoder_type = encoder_config["model_type"]
    elif encoder_config["model_type"] == "cnn":
        args.image_encoder_type = encoder_config["name"]
    else:
        raise ValueError(f"Unsupported image encoder type: {encoder_config['model_type']}")
    train_loader, valid_loader = get_dataloader(args)
    print(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}")
    if "breast_clip" not in args.arch:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    model = BreastClipClassifier(args, ckpt=ckpt, n_class=4 if _is_density(args) else 1).to(device)
    # The full CLIP file includes an unused text encoder; release it before training.
    del ckpt
    gc.collect()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = len(train_loader) if args.warmup_epochs == 1 else max(1, args.epochs)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        total_epochs=args.epochs,
        warmup_steps=warmup_steps,
        total_steps=len(train_loader) * args.epochs,
    )
    if _is_density(args):
        criterion = torch.nn.CrossEntropyLoss()
    elif args.weighted_BCE == "y":
        pos_weight = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    logger = SummaryWriter(args.tb_logs_path / f"fold{args.cur_fold}")
    best_auroc = -np.inf
    model_name = (
        f"{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_auroc_ver{args.VER}.pth"
    )
    model_path = args.chk_pt_path / model_name
    for epoch in range(args.epochs):
        train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, logger, device
        )
        _, predictions = valid_fn(valid_loader, model, criterion, args, device, epoch)
        validation = _attach_predictions(args, args.valid_folds, predictions)
        validation_auroc = _auroc(args, validation)
        print(f"Epoch {epoch + 1} validation AUROC: {validation_auroc:.6f}")
        logger.add_scalar(f"valid/{args.label}/AUROC", validation_auroc, epoch + 1)
        validation_levels = None
        if not _is_density(args):
            validation_levels = _binary_auroc_levels(args, validation)
            _print_binary_auroc_levels(f"Epoch {epoch + 1} validation", validation_levels)
            for level in ("image", "patient"):
                if validation_levels[level]["auroc"] is not None:
                    logger.add_scalar(f"valid/{args.label}/{level}_AUROC",
                                      validation_levels[level]["auroc"], epoch + 1)
        if validation_auroc > best_auroc:
            best_auroc = validation_auroc
            torch.save(
                {
                    "model": model.state_dict(),
                    "predictions": predictions,
                    "epoch": epoch,
                    "auroc": validation_auroc,
                    "task": args.label,
                    "class_names": list("ABCD") if _is_density(args) else (
                        ["non_invasive_cancer", "invasive_cancer"] if args.label == "invasive" else ["negative", "positive"]),
                    "image_size_hw": [1520, 912],
                    "backbone_checkpoint": str(args.clip_chk_pt_path),
                    "manifest_sha256": getattr(args, "grounding_manifest_sha256", None),
                    **({"validation_auroc_levels": validation_levels} if validation_levels is not None else {}),
                },
                model_path,
            )
    print(f"Best validation AUROC: {best_auroc:.6f}")
    best_checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model"])
    validation = _attach_predictions(args, args.valid_folds, best_checkpoint["predictions"])
    _save_predictions(args, validation, "validation")
    original_validation = args.valid_folds
    args.valid_folds = args.test_folds.copy()
    _, test_loader = get_dataloader(args)
    _, test_predictions = valid_fn(test_loader, model, criterion, args, device, epoch=0)
    test = _attach_predictions(args, args.test_folds, test_predictions)
    test_auroc = _auroc(args, test)
    print(f"Held-out test AUROC: {test_auroc:.6f}")
    if not _is_density(args):
        _print_binary_auroc_levels("Held-out test", _binary_auroc_levels(args, test))
    _save_predictions(args, test, "test")
    if args.dataset.lower() == "grounding":
        from .grounding import write_evaluation_artifacts
        write_evaluation_artifacts(args, validation, test, model_path)
    completion = {
        "status": "complete", "dataset": args.dataset, "task": args.label,
        "architecture": args.arch, "epochs": args.epochs,
        "selected_epoch": int(best_checkpoint["epoch"]) + 1,
        "validation_auroc": float(best_auroc), "test_auroc": float(test_auroc),
        "metric": "macro_ovr_auroc" if _is_density(args) else "auroc",
        "class_names": best_checkpoint["class_names"], "image_size_hw": [1520, 912],
        "backbone_checkpoint": str(args.clip_chk_pt_path), "checkpoint": str(model_path),
        "test_images": len(test), "test_units": len(_aggregate_for_auroc(args, test)),
    }
    if not _is_density(args):
        completion.update({
            "checkpoint_selection_level": "current_grouping",
            "validation_auroc_levels": _binary_auroc_levels(args, validation),
            "test_auroc_levels": _binary_auroc_levels(args, test),
        })
    (Path(args.output_path) / f"seed_{args.seed}_fold{args.cur_fold}_{args.label}_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    args.valid_folds = original_validation
    logger.close()
    torch.cuda.empty_cache()
    gc.collect()
    return test


def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    progress_iter = tqdm(
        enumerate(train_loader),
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
        total=len(train_loader),
    )
    for step, data in progress_iter:
        inputs = data["x"].to(device).squeeze(1).permute(0, 3, 1, 2)
        labels = data["y"].float().to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.apex):
            logits = model(inputs)
            loss, _ = _loss_and_probabilities(args, criterion, logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        if step % args.log_freq == 0 or step == len(train_loader) - 1:
            index = step + len(train_loader) * epoch
            logger.add_scalar("train/iter_loss", losses.avg, index)
            logger.add_scalar("train/iter_lr", optimizer.param_groups[0]["lr"], index)
    return losses.avg


def valid_fn(valid_loader, model, criterion, args, device, epoch=0):
    model.eval()
    losses = AverageMeter()
    predictions = []
    progress_iter = tqdm(
        enumerate(valid_loader),
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
        total=len(valid_loader),
    )
    for _, data in progress_iter:
        inputs = data["x"].to(device).squeeze(1).permute(0, 3, 1, 2)
        labels = data["y"].float().to(device)
        with torch.no_grad():
            logits = model(inputs)
            loss, probabilities = _loss_and_probabilities(args, criterion, logits, labels)
        losses.update(loss.item(), inputs.size(0))
        predictions.append(probabilities.cpu().numpy())
    return losses.avg, np.concatenate(predictions)
