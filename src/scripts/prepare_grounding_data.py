#!/usr/bin/env python3
"""Prepare private, leakage-checked image CSVs for three binary grounding models.

Input configuration is JSON with ``vindr`` (csv, image_root), ``embed`` (csv),
and ``reports`` (train_csv, test_csv, bu_image_root).
The report-image test_csv is the historical file named val, not a validation
pool. No report text is loaded or copied. Existing input files are never edited.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


TASKS = ("mass", "calcification", "asymmetry")
SPLITS = ("train", "validation", "test")
SOURCES = ("vindr", "embed", "bu", "upmc")
OUTPUT_COLUMNS = (
    "source_dataset", "patient_id", "exam_id", "laterality", "image_path", "split",
    "source_patient_id", "source_exam_id",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path, columns, optional=()):
    path = Path(path).expanduser().resolve(strict=True)
    header = pd.read_csv(path, nrows=0).columns
    missing = set(columns) - set(header)
    if missing:
        raise ValueError("{} is missing columns {}".format(path.name, sorted(missing)))
    selected = list(columns) + [name for name in optional if name in header and name not in columns]
    # IDs remain strings, including numeric-looking values and leading zeros.
    return pd.read_csv(path, usecols=selected, dtype=str, keep_default_na=False)


def _required_strings(frame, columns):
    for column in columns:
        values = frame[column].astype(str).str.strip()
        if values.eq("").any() or values.str.lower().isin(("nan", "none", "null")).any():
            raise ValueError("Missing required identifier or metadata in {}".format(column))
        frame[column] = values


def _binary(series, name):
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise ValueError("{} must contain only nonmissing binary labels".format(name))
    return numeric.astype("int64")


def _fold_splits(series, mapping, source):
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or not numeric.isin(mapping).all():
        raise ValueError("Unexpected {} source folds".format(source))
    return numeric.astype(int).map(mapping)


def _image_name(value):
    value = str(value)
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError("Expected a basename-only image identifier")
    return value


def _png_path(value):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError("Expected an absolute source image path")
    if path.suffix.lower() not in (".dcm", ".png"):
        raise ValueError("Source image paths must name DICOM or PNG files")
    return str(path.with_suffix(".png").resolve())


def _common(frame, source, patient, exam, laterality, paths, splits):
    result = pd.DataFrame(index=frame.index)
    result["source_dataset"] = source
    result["source_patient_id"] = patient.astype(str)
    result["source_exam_id"] = exam.astype(str)
    result["patient_id"] = source + "::" + result["source_patient_id"]
    result["exam_id"] = result["patient_id"] + "::" + result["source_exam_id"]
    result["laterality"] = laterality.astype(str).str.strip().str.upper()
    if not result["laterality"].isin(("L", "R")).all():
        raise ValueError("{} laterality must be L or R".format(source))
    result["image_path"] = paths
    result["split"] = splits
    for task in TASKS:
        result[task] = pd.NA
    return result


def prepare_vindr(config):
    labels = ("Mass", "Suspicious_Calcification", "Asymmetry")
    variants = ("Focal_Asymmetry", "Global_Asymmetry")
    frame = _read_csv(config["csv"], ("patient_id", "image_id", "laterality", "split", "fold") + labels, variants)
    _required_strings(frame, ("patient_id", "image_id", "laterality", "split", "fold"))
    included_variants = [name for name in variants if name in frame]
    label_columns = list(labels) + included_variants
    for name in label_columns:
        frame[name] = _binary(frame[name], name)
    key = ["patient_id", "image_id"]
    for name in ("laterality", "split", "fold"):
        if frame.groupby(key)[name].nunique(dropna=False).gt(1).any():
            raise ValueError("Conflicting VinDr metadata across annotation rows: {}".format(name))
    audit = {"input_annotation_rows": len(frame), "asymmetry_positive_if_any": ["Asymmetry"] + included_variants,
             "unavailable_asymmetry_variant_columns": [name for name in variants if name not in frame],
             "split_policy": "Preserve source folds 0=test, 1=train, 2=validation; official test remains held out.",
             "group_identifier": "Source patient_id is the official study_id (exam-level grouping)."}
    frame = frame.groupby(key, as_index=False, sort=False).agg(
        {**{name: "max" for name in label_columns}, "laterality": "first", "split": "first", "fold": "first"})
    audit["duplicate_annotation_rows_aggregated"] = audit["input_annotation_rows"] - len(frame)
    splits = _fold_splits(frame["fold"], {0: "test", 1: "train", 2: "validation"}, "VinDr")
    official = frame["split"].str.strip().str.lower()
    if not official.isin(("training", "train", "test")).all() or not official.eq("test").equals(splits.eq("test")):
        raise ValueError("VinDr source folds disagree with the official held-out test split")
    image_root = Path(config["image_root"]).expanduser().resolve()
    paths = []
    for patient, image in zip(frame["patient_id"], frame["image_id"]):
        image = _image_name(image)
        name = image if image.lower().endswith(".png") else image + ".png"
        paths.append(str((image_root / _image_name(patient) / name).resolve()))
    output = _common(frame, "vindr", frame["patient_id"], frame["patient_id"], frame["laterality"], paths, splits)
    output["mass"] = frame["Mass"]
    output["calcification"] = frame["Suspicious_Calcification"]
    output["asymmetry"] = frame[["Asymmetry"] + included_variants].max(axis=1)
    return output, audit


def prepare_embed(config):
    frame = _read_csv(config["csv"], ("patient_id", "study_id", "anon_dicom_path", "ImageLateralityFinal",
                                            "fold", "mass", "calc", "asymmetry"))
    _required_strings(frame, ("patient_id", "study_id", "anon_dicom_path", "ImageLateralityFinal", "fold"))
    paths = frame["anon_dicom_path"].map(_png_path)
    splits = _fold_splits(frame["fold"], {0: "test", 1: "train", 2: "train", 3: "validation"}, "EMBED")
    output = _common(frame, "embed", frame["patient_id"], frame["study_id"], frame["ImageLateralityFinal"], paths, splits)
    for target, original in (("mass", "mass"), ("calcification", "calc"), ("asymmetry", "asymmetry")):
        output[target] = _binary(frame[original], original)
    return output, {"input_rows": len(frame), "split_policy": "Preserve 0=test, 3=validation, 1/2=train.",
                    "calcification_definition": "Native recorded calcification finding; not established as suspicious-only.",
                    "view_policy": "Retain the historical diagnostic 2D source cohort, including spot/magnification views."}


def report_patient_split(patient):
    # This reproduces the historical helper exactly; no random seed is involved.
    bucket = int(hashlib.sha256(str(patient).encode("utf-8")).hexdigest()[:16], 16) % 4 + 1
    return "validation" if bucket == 4 else "train"


def _report_image_path(value, source, bu_image_root):
    path = Path(value)
    if not path.is_absolute() or path.suffix.lower() != ".png":
        raise ValueError("Report-image source paths must be absolute PNG paths")
    if source == "upmc":
        return str(Path(str(path).replace("/UPMC_with_embeddings/", "/UPMC/")).resolve())
    categories = [name for name in ("controls", "cases") if name in path.parts]
    if len(categories) != 1:
        raise ValueError("BU source paths must identify exactly one of controls/cases")
    return str((Path(bu_image_root).expanduser().resolve() / categories[0] / "test_images_png" /
                path.parent.name / path.name).resolve())


def prepare_reports(config):
    columns = ("dataset", "patient_id", "exam_id", "image_laterality", "image_view_type", "image_path",
               "mass_present", "calc_present", "asymmetry_present")
    training, testing = [_read_csv(config[name], columns) for name in ("train_csv", "test_csv")]
    for frame in (training, testing):
        _required_strings(frame, columns[:6])
        frame["dataset"] = frame["dataset"].str.lower()
        if not frame["dataset"].isin(("bu", "upmc")).all():
            raise ValueError("Report-image CSVs must contain only BU and UPMC")
        if not frame["image_view_type"].isin(("CC", "MLO")).all():
            raise ValueError("Report-image rows must contain CC/MLO views")
        for name in columns[-3:]:
            frame[name] = _binary(frame[name], name)
    outputs, audits = [], {}
    for source in ("bu", "upmc"):
        pool = training.loc[training["dataset"] == source].copy()
        test = testing.loc[testing["dataset"] == source].copy()
        if pool.empty or test.empty:
            raise ValueError("Missing {} report-image training/test cohort".format(source))
        heldout_patients = set(test["patient_id"])
        excluded = pool["patient_id"].isin(heldout_patients)
        audits[source] = {"input_training_rows": len(pool), "input_test_rows": len(test),
                          "excluded_training_rows_with_heldout_patient": int(excluded.sum()),
                          "excluded_training_patients_in_test": int(pool.loc[excluded, "patient_id"].nunique()),
                          "split_policy": "Exclude all test patients, then raw patient SHA256 first16hex modulo4 +1; bucket4 validation, others train. No split seed.",
                          "labels": "Native per-image report-derived findings; CC/MLO labels are not forced to agree."}
        pool = pool.loc[~excluded].copy()
        pool["split"] = pool["patient_id"].map(report_patient_split)
        test["split"] = "test"
        frame = pd.concat((pool, test), ignore_index=True)
        paths = frame["image_path"].map(lambda value: _report_image_path(value, source, config["bu_image_root"]))
        output = _common(frame, source, frame["patient_id"], frame["exam_id"], frame["image_laterality"], paths, frame["split"])
        for target, original in (("mass", "mass_present"), ("calcification", "calc_present"), ("asymmetry", "asymmetry_present")):
            output[target] = frame[original].astype("int64")
        outputs.append(output)
    return outputs, audits


def validate_pooled(frame, path_workers=4):
    if not isinstance(path_workers, int) or isinstance(path_workers, bool) or path_workers < 1:
        raise ValueError("path_workers must be a positive integer")
    if set(frame["source_dataset"].unique()) != set(SOURCES):
        raise ValueError("All four requested source datasets must be present")
    if not frame["split"].isin(SPLITS).all():
        raise ValueError("Unexpected pooled split")
    if frame[list(OUTPUT_COLUMNS)].isna().any().any():
        raise ValueError("Missing output metadata")
    if frame["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths remain in the pooled cohort")
    if frame.groupby("patient_id")["split"].nunique().gt(1).any():
        raise ValueError("Patient/group leakage across pooled splits")
    if frame.groupby("exam_id")["split"].nunique().gt(1).any():
        raise ValueError("Exam leakage across pooled splits")
    for task in TASKS:
        if frame[task].isna().any():
            raise ValueError("Missing pooled labels for {}".format(task))
        task_frame = frame
        _binary(task_frame[task], task)
        expected_sources = set(SOURCES)
        if set(task_frame["source_dataset"].unique()) != expected_sources:
            raise ValueError("Incorrect source coverage for {}".format(task))
        for source in expected_sources:
            for split in SPLITS:
                labels = task_frame.loc[(task_frame["source_dataset"] == source) & (task_frame["split"] == split), task]
                if set(labels.unique()) != {0, 1}:
                    raise ValueError("{}/{}/{} requires both binary classes".format(source, task, split))
    paths = frame["image_path"].tolist()
    if any(not Path(path).is_absolute() or Path(path).suffix.lower() != ".png" for path in paths):
        raise ValueError("All output image paths must be absolute PNG paths")
    with ThreadPoolExecutor(max_workers=path_workers) as workers:
        present = list(workers.map(lambda value: Path(value).is_file(), paths))
    missing = len(present) - sum(present)
    if missing:
        raise FileNotFoundError("{} pooled PNG paths are missing; no CSVs were written".format(missing))
    return {"checked_unique_png_paths": len(paths), "missing_png_paths": 0,
            "patient_split_overlaps": 0, "exam_split_overlaps": 0, "image_split_overlaps": 0}


def _task_counts(frame):
    result = {}
    for (source, split), group in frame.groupby(["source_dataset", "split"], sort=True):
        result.setdefault(source, {})[split] = {
            "images": len(group), "patients_or_groups": int(group["patient_id"].nunique()),
            "exams_or_surrogates": int(group["exam_id"].nunique()),
            "positive_images": int(group["label"].sum()), "negative_images": int((group["label"] == 0).sum()),
        }
    return result


def _write_private(path, write):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        write(stream)


def prepare(config, output_dir, path_workers=4):
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing output directory")
    if set(config) != {"vindr", "embed", "reports"}:
        raise ValueError("Configuration must have exactly vindr, embed and reports sections")
    input_paths = [config[name]["csv"] for name in ("vindr", "embed")]
    input_paths += [config["reports"][name] for name in ("train_csv", "test_csv")]
    fingerprints = {str(Path(path).expanduser().resolve(strict=True)): sha256_file(Path(path).expanduser())
                    for path in input_paths}
    frames, source_audits = [], {}
    for source, function in (("vindr", prepare_vindr), ("embed", prepare_embed)):
        frame, audit = function(config[source])
        frames.append(frame)
        source_audits[source] = audit
    report_frames, report_audits = prepare_reports(config["reports"])
    frames.extend(report_frames)
    source_audits.update(report_audits)
    pooled = pd.concat(frames, ignore_index=True)
    validation = validate_pooled(pooled, path_workers)
    for path, digest in fingerprints.items():
        if sha256_file(path) != digest:
            raise RuntimeError("An input CSV changed during preparation; refusing to write outputs")
    outputs, task_audits = {}, {}
    for task in TASKS:
        output = pooled.loc[:, list(OUTPUT_COLUMNS) + [task]].copy()
        output[task] = _binary(output[task], task)
        output = output.sort_values(["source_dataset", "split", "patient_id", "exam_id", "image_path"], kind="stable")
        outputs[task] = output
        task_audits[task] = {"rows": len(output), "by_source_and_split": _task_counts(output.rename(columns={task: "label"}))}
    audit = {
        "schema_version": "1.0", "status": "PASS", "private_metadata_only": True,
        "input_sha256": fingerprints, "source_audits": source_audits,
        "validation": validation, "tasks": task_audits,
        "global_split_counts": {split: int((pooled["split"] == split).sum()) for split in SPLITS},
        "manifests": {},
        "label_semantics": {
            "mass": "Native mass findings; VinDr duplicate annotations are OR-aggregated.",
            "calcification": "Pooled native calcification endpoints: EMBED calc, BU/UPMC calc_present, and VinDr Suspicious_Calcification. This is not uniformly a suspicious-only endpoint.",
            "asymmetry": "BU/UPMC and EMBED native asymmetry, VinDr OR of Asymmetry/Focal_Asymmetry/Global_Asymmetry when available.",
        },
        "threshold_policy": "Train/select models and thresholds on training/validation only; keep all source test cohorts held out.",
        "split_group_caveat": "VinDr patient_id is the official study identifier; no stronger patient identifier is available in this source CSV.",
    }
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    for task, frame in outputs.items():
        path = output_dir / ("grounding_{}.csv".format(task))
        _write_private(path, lambda stream, frame=frame: frame.to_csv(stream, index=False))
        task_audits[task]["output_sha256"] = sha256_file(path)
        task_audits[task]["filename"] = path.name
        audit["manifests"][task] = {"path": str(path), "sha256": task_audits[task]["output_sha256"]}
    _write_private(output_dir / "audit.json", lambda stream: json.dump(audit, stream, indent=2, sort_keys=True))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Private JSON source-path configuration")
    parser.add_argument("--output-dir", type=Path, required=True, help="New private directory; existing paths are refused")
    parser.add_argument("--path-workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = prepare(config, args.output_dir, args.path_workers)
    print(json.dumps({"status": audit["status"], "output_dir": str(args.output_dir.resolve()),
                      "rows": {task: entry["rows"] for task, entry in audit["tasks"].items()},
                      "validation": audit["validation"]}, indent=2))


if __name__ == "__main__":
    main()
