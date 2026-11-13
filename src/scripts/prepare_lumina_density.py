#!/usr/bin/env python3
"""Join original LUMINA density annotations without conflating cohort-local IDs.

The Benign and Malign workbooks reuse numeric patient IDs.  A patient key must
therefore include the cohort.  Original image rows and fold assignments remain
unchanged; this script writes a separate, one-row-per-cohort/patient label CSV.
No image data or report text is copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
COHORTS = {"Benign": "Benign", "Malignant": "Malign"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_density_workbook(path: Path, cohort: str) -> tuple[pd.DataFrame, dict]:
    """Read the original XLSX using stdlib XML; no optional Excel engine needed."""
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(node.text or "" for node in item.findall(".//m:t", NS))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]

        def cell_value(cell: ET.Element) -> str:
            if cell.attrib.get("t") == "inlineStr":
                return "".join(node.text or "" for node in cell.findall(".//m:t", NS))
            value = cell.find("m:v", NS)
            if value is None or value.text is None:
                return ""
            if cell.attrib.get("t") == "s":
                return strings[int(value.text)]
            return value.text

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = sheet.findall("m:sheetData/m:row", NS)
        if not rows:
            raise ValueError(f"Empty worksheet: {path.name}")
        headers = {
            re.sub(r"\d", "", cell.attrib["r"]): cell_value(cell).strip()
            for cell in rows[0]
        }
        required = {"ID", "BREAST COMPOSITION"}
        if not required.issubset(headers.values()):
            raise ValueError(f"Required columns absent in {path.name}: {sorted(required)}")

        records = []
        for row in rows[1:]:
            record = {}
            for cell in row:
                header = headers.get(re.sub(r"\d", "", cell.attrib["r"]), "")
                if header in required:
                    record[header] = cell_value(cell).strip() or None
            records.append(record)

    table = pd.DataFrame(records)
    # Workbook IDs are present only at the beginning of each exam block.
    table["ID"] = table["ID"].ffill()
    raw = table["BREAST COMPOSITION"]
    missing = raw.isna() | raw.eq("-")
    numeric = pd.to_numeric(raw.mask(missing), errors="coerce")
    invalid = ~missing & ~numeric.isin([1, 2, 3, 4])
    if invalid.any():
        raise ValueError(f"{path.name} has {int(invalid.sum())} invalid density values")
    labels = pd.DataFrame({"patient_id": table["ID"], "density": numeric - 1})
    labels = labels.loc[~missing].copy()
    ids = pd.to_numeric(labels["patient_id"], errors="raise")
    if ids.isna().any() or (ids % 1 != 0).any():
        raise ValueError(f"{path.name} contains a density without a valid integer patient ID")
    labels["patient_id"] = ids.astype("int64")
    labels["density"] = labels["density"].astype("int64")
    conflicts = labels.groupby("patient_id")["density"].nunique().gt(1)
    if conflicts.any():
        raise ValueError(f"{path.name} has {int(conflicts.sum())} conflicting patient densities")
    labels = labels.drop_duplicates().assign(Cancer=cohort)
    return labels, {
        "file": path.name,
        "sha256": sha256(path),
        "unique_cohort_patient_labels": len(labels),
        "missing_composition_cells": int(missing.sum()),
        "conflicting_labels": 0,
    }


def prepare(args: argparse.Namespace) -> dict:
    paths = [args.images_csv, args.benign_xlsx, args.malign_xlsx]
    if args.output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("Output must not overwrite an original source file")
    audit_path = args.audit or args.output.with_suffix(".audit.json")
    if audit_path.resolve() in {path.resolve() for path in paths + [args.output]}:
        raise ValueError("Audit output must be separate from source and label CSV files")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError("Choose new output/audit paths; existing artifacts are not overwritten")

    images = pd.read_csv(args.images_csv)
    required = {"Cancer", "patient_id", "fold", "image_full_path"}
    if not required.issubset(images.columns):
        raise ValueError(f"Image CSV requires {sorted(required)}")
    if images[list(required)].isna().any().any():
        raise ValueError("Image CSV contains a missing cohort, patient, fold, or image path")
    if not set(images["Cancer"]).issubset(COHORTS):
        raise ValueError("Unrecognized LUMINA Cancer cohort value")
    if images["image_full_path"].duplicated().any():
        raise ValueError("Image CSV has duplicate image paths")
    for field in ["patient_id", "fold"]:
        values = pd.to_numeric(images[field], errors="raise")
        if (values % 1 != 0).any():
            raise ValueError(f"Image CSV has noninteger {field}")
        images[field] = values.astype("int64")
    keys = ["Cancer", "patient_id"]
    if images.groupby(keys)["fold"].nunique().gt(1).any():
        raise ValueError("At least one cohort/patient occurs in multiple folds")
    for cohort, group in images.groupby("Cancer"):
        expected_directory = COHORTS[cohort]
        if not group["image_full_path"].map(
            lambda value: len(Path(value).parts) >= 3
            and Path(value).parts[-3] == expected_directory
        ).all():
            raise ValueError("Image path cohort does not agree with its Cancer column")

    benign, benign_audit = read_density_workbook(args.benign_xlsx, "Benign")
    malign, malign_audit = read_density_workbook(args.malign_xlsx, "Malignant")
    labels = pd.concat([benign, malign], ignore_index=True)
    labeled = images.merge(labels, on=keys, how="left", validate="many_to_one")
    if len(labeled) != len(images) or labeled["density"].isna().any():
        raise ValueError("Not every image has exactly one unambiguous original density label")
    labeled["density"] = labeled["density"].astype("int64")
    output = labeled[keys + ["density", "fold"]].drop_duplicates()
    if output.duplicated(keys).any():
        raise ValueError("Density output must have one row per cohort/patient")
    output = output.sort_values(keys).reset_index(drop=True)
    counts = {
        str(int(fold)): {
            "images": len(group),
            "cohort_patient_groups": len(group[keys].drop_duplicates()),
            "image_density_counts": {
                str(label): int(group["density"].eq(label).sum()) for label in range(4)
            },
        }
        for fold, group in labeled.groupby("fold", sort=True)
    }
    audit = {
        "status": "PASS",
        "source_image_csv": str(args.images_csv.resolve()),
        "source_image_csv_sha256": sha256(args.images_csv),
        "source_workbooks": [benign_audit, malign_audit],
        "join_keys": keys,
        "density_mapping": {"1 / A": 0, "2 / B": 1, "3 / C": 2, "4 / D": 3},
        "images_labeled": len(labeled),
        "cohort_patient_groups": len(output),
        "unmatched_images": 0,
        "conflicting_labels": 0,
        "cross_fold_cohort_patient_groups": 0,
        "existing_folds_preserved": True,
        "per_fold": counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    audit["output_csv"] = str(args.output.resolve())
    audit["output_csv_sha256"] = sha256(args.output)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-csv", required=True, type=Path)
    parser.add_argument("--benign-xlsx", required=True, type=Path)
    parser.add_argument("--malign-xlsx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    audit = prepare(parser.parse_args())
    print(json.dumps({key: audit[key] for key in [
        "status", "images_labeled", "cohort_patient_groups", "per_fold", "output_csv"
    ]}, indent=2))


if __name__ == "__main__":
    main()
