"""Thin five-year adapter around the collaborator's Mayo dataset.

Image discovery, incomplete-exam handling, DICOM decoding, cropping, resizing,
and normalization intentionally remain in
``mammo_clip_metadataset_mayo_KD.Mammo_CLIPMetadataset``. This module only
adds the five outcome targets/masks and presents its tuple output in the shape
expected by the corrected risk-training loop.
"""

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


YEARS = tuple(range(1, 6))
DEFAULT_LABEL_COLS = tuple(f"cancer{year}yr_updated" for year in YEARS)
DEFAULT_RISK_COLS = tuple(f"{year}_year_risk" for year in YEARS)
# Kept for command-line compatibility. The legacy loader's five logit slots
# are placeholders and are deliberately never used by this adapter.
DEFAULT_LOGIT_COLS = tuple(f"logit_yr{year}" for year in YEARS)
DEFAULT_KNOWN_COLS = tuple(f"known_yr{year}" for year in YEARS)

# Exact order returned by the collaborator's loader. Do not reorder it here.
VIEW_ORDER = ("LCC", "RCC", "LMLO", "RMLO")
TIME_SEQ = torch.tensor([0, 0, 0, 0], dtype=torch.long)
VIEW_SEQ = torch.tensor([1, 1, 0, 0], dtype=torch.long)
SIDE_SEQ = torch.tensor([0, 1, 0, 1], dtype=torch.long)


@dataclass(frozen=True)
class ManifestColumns:
    image: str = "file"
    exam: str = "AccessionNumber"
    patient: str = "patient_id"
    laterality: str = "laterality"
    view: str = "view"
    split: str = "split"
    cancer_registry: str = "cancer_registry"
    years_to_cancer: str = "years_to_cancer"
    years_to_followup: str = "years_to_last_followup"
    labels: tuple = DEFAULT_LABEL_COLS
    teacher_risks: tuple = DEFAULT_RISK_COLS
    teacher_logits: tuple = DEFAULT_LOGIT_COLS
    known: tuple = DEFAULT_KNOWN_COLS


_LEGACY_DATASET_CLASS = None


def _legacy_dataset_class():
    """Import the preserved loader without changing its image pipeline."""
    global _LEGACY_DATASET_CLASS
    if _LEGACY_DATASET_CLASS is not None:
        return _LEGACY_DATASET_CLASS

    legacy_dir = (
        Path(__file__).resolve().parents[1]
        / "AsymMirai-master"
        / "asymmetry_model"
    )
    if not legacy_dir.is_dir():
        raise FileNotFoundError(f"Collaborator Mayo dataset directory not found: {legacy_dir}")
    legacy_dir_string = str(legacy_dir)
    if legacy_dir_string not in sys.path:
        sys.path.insert(0, legacy_dir_string)
    try:
        module = importlib.import_module("mammo_clip_metadataset_mayo_KD")
    except ModuleNotFoundError as error:
        if error.name == "pydicom":
            raise ImportError(
                "The collaborator Mayo dataset requires pydicom in the active environment."
            ) from error
        raise
    _LEGACY_DATASET_CLASS = module.Mammo_CLIPMetadataset
    return _LEGACY_DATASET_CLASS


def format_fold_value(value, fold):
    """Substitute ``{fold}`` when the existing manifest uses fold-tagged splits."""
    return str(value).format(fold=fold)


def select_split_frames(
    frame,
    split_column="split",
    fold=1,
    train_value="train",
    val_value="valid",
    predict_value="valid",
):
    """Select existing manifest split values, without creating or changing splits."""
    if split_column not in frame.columns:
        raise ValueError(f"Split column is missing: {split_column}")
    requested = {
        "train": format_fold_value(train_value, fold),
        "val": format_fold_value(val_value, fold),
        "predict": format_fold_value(predict_value, fold),
    }
    selected = {
        name: frame.loc[frame[split_column].astype(str) == value].copy()
        for name, value in requested.items()
    }
    empty = [name for name, subset in selected.items() if subset.empty]
    if empty:
        counts = frame[split_column].astype(str).value_counts(dropna=False).to_dict()
        raise ValueError(
            f"Empty requested split(s) {empty}; requested={requested}, available={counts}"
        )
    return selected


def _first_nonmissing(group, column, default=np.nan):
    if not column or column not in group.columns:
        return default
    values = group[column].dropna()
    return default if values.empty else values.iloc[0]


def _binary_any(group, column, exam_id):
    if column not in group.columns:
        raise ValueError(f"Five-year label column is missing: {column}")
    values = pd.to_numeric(group[column], errors="coerce")
    finite_values = values.dropna().to_numpy(dtype=float)
    if finite_values.size and not np.isin(finite_values, [0.0, 1.0]).all():
        raise ValueError(f"Exam {exam_id} has a non-binary value in {column}")
    # Matches the legacy loader's one-year exam-level label construction.
    return float(values.eq(1).any())


def _mask_for_group(group, columns, exam_id):
    if all(column in group.columns for column in columns.known):
        mask = np.asarray(
            [float(_first_nonmissing(group, column, 0.0)) for column in columns.known],
            dtype=np.float32,
        )
    elif (
        columns.cancer_registry in group.columns
        and columns.years_to_followup in group.columns
    ):
        registry = pd.to_numeric(
            pd.Series([_first_nonmissing(group, columns.cancer_registry)]),
            errors="coerce",
        ).iloc[0]
        followup = pd.to_numeric(
            pd.Series([_first_nonmissing(group, columns.years_to_followup)]),
            errors="coerce",
        ).iloc[0]
        if not np.isfinite(registry) or not np.isfinite(followup):
            raise ValueError(
                f"Exam {exam_id} is missing cancer_registry/years_to_last_followup "
                "needed for the five-year availability mask"
            )
        mask = (
            np.ones(5, dtype=np.float32)
            if int(registry) == 1
            else np.asarray([followup >= year for year in YEARS], dtype=np.float32)
        )
    else:
        # Do not manufacture follow-up. With no censoring columns, label
        # presence is the only information supplied by their manifest.
        mask = np.asarray(
            [group[column].notna().any() for column in columns.labels],
            dtype=np.float32,
        )
    if not np.isin(mask, [0.0, 1.0]).all():
        raise ValueError(f"Exam {exam_id} has non-binary five-year availability masks")
    return mask


def _to_chw_float(image):
    """Change layout only; preserve the legacy loader's pixel values."""
    tensor = image if torch.is_tensor(image) else torch.as_tensor(np.asarray(image))
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Legacy Mayo image must be 3-D, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    elif tensor.shape[0] not in (1, 3):
        raise ValueError(f"Cannot identify image channel axis in shape {tuple(tensor.shape)}")
    return tensor.contiguous().float()


class LegacyMayoFiveYearDataset(Dataset):
    """Add five-year supervision to the unmodified collaborator Mayo loader.

    ``frame`` is passed directly to ``Mammo_CLIPMetadataset``. Consequently,
    its exact view selection, complete-exam filtering, DICOM preprocessing, and
    image normalization are retained.
    """

    def __init__(
        self,
        frame,
        columns=ManifestColumns(),
        mode="training",
        transforms=None,
        align_images=False,
    ):
        self.columns = columns
        self.source_frame = frame
        LegacyDataset = _legacy_dataset_class()
        self.legacy_dataset = LegacyDataset(
            frame,
            path_col=columns.image,
            exam_col=columns.exam,
            lat_col=columns.laterality,
            view_col=columns.view,
            label_col=columns.labels[0],
            dataset="mayo",
            allow_incomplete=False,
            verbose=False,
            transforms=transforms,
            mode=mode,
            align_images=align_images,
            multiple_pairs_per_exam=False,
        )

        sidecar_frame = frame.copy()
        if columns.exam not in sidecar_frame.columns:
            raise ValueError(f"Exam column is missing: {columns.exam}")
        missing_labels = [name for name in columns.labels if name not in sidecar_frame.columns]
        if missing_labels:
            raise ValueError(f"Five-year label columns are missing: {missing_labels}")
        sidecar_frame["_legacy_exam_id"] = sidecar_frame[columns.exam].astype(str)
        grouped = {
            str(exam_id): group
            for exam_id, group in sidecar_frame.groupby("_legacy_exam_id", sort=True)
        }

        records = []
        for legacy_exam in self.legacy_dataset.exams:
            exam_id = str(legacy_exam.eid)
            group = grouped.get(exam_id)
            if group is None:
                raise RuntimeError(f"Could not align legacy exam {exam_id} to its source rows")
            labels = np.asarray(
                [_binary_any(group, column, exam_id) for column in columns.labels],
                dtype=np.float32,
            )
            mask = _mask_for_group(group, columns, exam_id)
            patient_id = _first_nonmissing(group, columns.patient, exam_id)
            split = _first_nonmissing(group, columns.split, "")
            record = {
                "patient_id": str(patient_id),
                "exam_id": exam_id,
                "split": str(split),
            }
            for index, year in enumerate(YEARS):
                record[f"label_yr{year}"] = float(labels[index])
                record[f"known_yr{year}"] = float(mask[index])
                # Use the exact values selected by the legacy loader, not a
                # second interpretation of duplicate image-level CSV rows.
                record[f"teacher_risk_yr{year}"] = float(legacy_exam[7 + index])
            records.append(record)

        self.exams = pd.DataFrame.from_records(records)
        if self.exams.empty:
            raise ValueError("The collaborator Mayo loader found no complete four-view exams")
        if self.exams["exam_id"].duplicated().any():
            raise RuntimeError("The collaborator Mayo loader returned duplicate exam IDs")
        if len(self.exams) != len(self.legacy_dataset):
            raise RuntimeError("Five-year sidecar is not aligned to the collaborator dataset")

        # Warn, but do not reinterpret their dataset, when no censoring metadata
        # exists and label presence is the only possible mask definition.
        has_known = all(column in frame.columns for column in columns.known)
        has_followup = (
            columns.cancer_registry in frame.columns
            and columns.years_to_followup in frame.columns
        )
        if not has_known and not has_followup:
            warnings.warn(
                "No known_yr1..5 or follow-up columns were found; five-year masks use "
                "whether each existing label is non-missing."
            )

    def __len__(self):
        return len(self.legacy_dataset)

    def __getitem__(self, index):
        sample = self.legacy_dataset[index]
        # Fixed legacy tuple positions:
        #   2:7  = dummy zero logits (intentionally ignored)
        #   7:12 = real MIRAI risk probabilities
        #   12/14/16/18 = LCC/RCC/LMLO/RMLO images
        images = torch.stack(
            [_to_chw_float(sample[position]) for position in (12, 14, 16, 18)]
        )
        teacher_probs = torch.tensor(
            [float(torch.as_tensor(value).item()) for value in sample[7:12]],
            dtype=torch.float32,
        )
        if not torch.isfinite(teacher_probs).all():
            raise ValueError(f"Exam {sample[0]} has non-finite MIRAI probabilities")
        if ((teacher_probs < 0) | (teacher_probs > 1)).any():
            raise ValueError(f"Exam {sample[0]} has MIRAI probabilities outside [0, 1]")

        row = self.exams.iloc[index]
        if str(sample[0]) != str(row["exam_id"]):
            raise RuntimeError(
                f"Legacy Mayo sample/sidecar mismatch at index {index}: "
                f"{sample[0]!r} != {row['exam_id']!r}"
            )
        if bool(row["known_yr1"]) and int(torch.as_tensor(sample[1]).item()) != int(
            row["label_yr1"]
        ):
            raise RuntimeError(f"Legacy Mayo year-1 label mismatch for exam {sample[0]}")
        return {
            "patient_id": str(row["patient_id"]),
            "exam_id": str(row["exam_id"]),
            "images": images,
            "time_seq": TIME_SEQ.clone(),
            "view_seq": VIEW_SEQ.clone(),
            "side_seq": SIDE_SEQ.clone(),
            "labels": torch.tensor(
                [row[f"label_yr{year}"] for year in YEARS], dtype=torch.float32
            ),
            "label_mask": torch.tensor(
                [row[f"known_yr{year}"] for year in YEARS], dtype=torch.float32
            ),
            "teacher_probs": teacher_probs,
        }


def compute_pos_weights(train_exams):
    """Compute per-horizon BCE weights after legacy complete-exam filtering."""
    if isinstance(train_exams, LegacyMayoFiveYearDataset):
        train_exams = train_exams.exams
    weights = []
    for year in YEARS:
        known = train_exams[f"known_yr{year}"].to_numpy(dtype=bool)
        labels = train_exams.loc[known, f"label_yr{year}"].to_numpy(dtype=float)
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        if positives == 0 or negatives == 0:
            raise ValueError(
                f"Year {year} training outcomes need both classes; "
                f"found positives={positives}, negatives={negatives}"
            )
        weights.append(negatives / positives)
    return torch.tensor(weights, dtype=torch.float32)
