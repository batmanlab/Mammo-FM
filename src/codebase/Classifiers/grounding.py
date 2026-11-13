"""Patient-safe pooled, binary grounding classification and validation artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


TASKS = {"mass", "calcification", "asymmetry"}
SOURCES = {"vindr", "embed", "bu", "upmc"}
GROUP_KEYS = ["source_dataset", "patient_id", "exam_id", "laterality"]


def prepare_splits(args, frame):
    """Consume audited manifests without inventing a new pooled train/test split."""
    if args.label not in TASKS:
        raise ValueError(f"Unknown grounding task: {args.label}")
    if not args.arch.endswith("_ft"):
        raise ValueError("Pooled Stage-3 grounding requires full fine-tuning")
    if args.cur_fold != 0:
        raise ValueError("Pooled grounding has one fixed, source-defined held-out test set")
    required = set(GROUP_KEYS + ["image_path", "split", args.label])
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Grounding manifest is missing columns: {sorted(missing)}")
    data = frame.copy()
    if data[list(required)].isna().any().any():
        raise ValueError("Grounding manifest contains missing identifiers, paths, splits, or task labels")
    if set(data["source_dataset"].unique()) - SOURCES:
        raise ValueError("Grounding manifest contains an unsupported source dataset")
    if set(data["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Grounding requires explicit train, validation, and test splits")
    if not data[args.label].isin([0, 1]).all():
        raise ValueError("Grounding labels must be explicitly known binary values, not missing-label negatives")
    if data["image_path"].duplicated().any():
        raise ValueError("Duplicate images in grounding manifest")
    if not data["image_path"].map(lambda value: Path(value).is_absolute()).all():
        raise ValueError("Grounding image paths must be absolute PNG paths")
    if data.groupby("patient_id")["source_dataset"].nunique().gt(1).any():
        raise ValueError("Patient IDs must be qualified by source dataset")
    if data.groupby("patient_id")["split"].nunique().gt(1).any():
        raise ValueError("Grounding patient leakage between splits")
    if data.groupby("exam_id")["split"].nunique().gt(1).any():
        raise ValueError("Grounding exam leakage between splits")
    if not data["laterality"].isin(["L", "R"]).all():
        raise ValueError("Grounding laterality must be L or R")
    for (source, split), group in data.groupby(["source_dataset", "split"]):
        if set(group[args.label].unique()) != {0, 1}:
            raise ValueError(f"{source}/{split}/{args.label} must have both classes for AUROC")
    args.df = data
    args.train_folds = data[data["split"] == "train"].copy().reset_index(drop=True)
    args.valid_folds = data[data["split"] == "validation"].copy().reset_index(drop=True)
    args.test_folds = data[data["split"] == "test"].copy().reset_index(drop=True)
    args.grounding_manifest_sha256 = hashlib.sha256((args.data_dir / args.csv_file).read_bytes()).hexdigest()
    print(f"Grounding/{args.label} images: train={len(args.train_folds)}, "
          f"validation={len(args.valid_folds)}, test={len(args.test_folds)}")


def aggregate(args, frame):
    """One score per source-qualified exam/breast, preserving longitudinal exams."""
    return frame.groupby(GROUP_KEYS, as_index=False).agg(
        **{args.label: (args.label, "max"), "prediction": ("prediction", "max")}
    )


def validation_threshold(frame, label):
    """Choose an image-level Youden operating point using validation only."""
    if frame[label].nunique() != 2 or not np.isfinite(frame["prediction"]).all():
        raise ValueError("Threshold selection requires finite validation probabilities and both classes")
    fpr, tpr, thresholds = roc_curve(frame[label].astype(int), frame["prediction"])
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    candidates = thresholds[valid]
    if not len(candidates):
        raise ValueError("No valid operating point in validation predictions")
    return float(candidates[np.argmax((tpr - fpr)[valid])])


def write_evaluation_artifacts(args, validation, test, model_path):
    report = {
        "task": args.label,
        "checkpoint": str(model_path),
        "manifest_sha256": args.grounding_manifest_sha256,
        "selection_metric": "pooled_exam_breast_auroc",
        "aggregation": "max across images within source/patient/exam/laterality",
        "threshold": validation_threshold(validation, args.label),
        "threshold_source": "validation_images_only",
        "threshold_method": "Youden_J",
        "threshold_comparison": "probability >= threshold",
        "image_size_hw": [1520, 912],
        "by_source": {},
        "deployment_status": "candidate; does not replace published Stage-3 grounders automatically",
    }
    for split, frame in (("validation", validation), ("test", test)):
        for source, group in frame.groupby("source_dataset"):
            breasts = aggregate(args, group)
            patients = group.groupby(["source_dataset", "patient_id"], as_index=False).agg(
                **{args.label: (args.label, "max"), "prediction": ("prediction", "max")}
            )
            patient_auroc = (
                float(roc_auc_score(patients[args.label], patients["prediction"]))
                if patients[args.label].nunique() == 2 else None
            )
            entry = {
                "images": len(group), "exam_breasts": len(breasts),
                "image_auroc": float(roc_auc_score(group[args.label], group["prediction"])),
                "exam_breast_auroc": float(roc_auc_score(breasts[args.label], breasts["prediction"])),
                "patients": len(patients), "patient_auroc": patient_auroc,
                "patient_unit": "study" if source == "vindr" else "patient",
                "patient_auroc_undefined_reason": (
                    "AUROC requires both positive and negative units" if patient_auroc is None else None
                ),
            }
            report["by_source"].setdefault(source, {})[split] = entry
            patient_text = "undefined (single class)" if patient_auroc is None else f"{patient_auroc:.6f}"
            print(f"{source} {split} image AUROC: {entry['image_auroc']:.6f}; "
                  f"exam-breast AUROC: {entry['exam_breast_auroc']:.6f}; "
                  f"patient AUROC: {patient_text}; units={len(patients)} ({entry['patient_unit']})")
    path = Path(args.output_path) / f"seed_{args.seed}_fold{args.cur_fold}_{args.label}_grounding.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report
