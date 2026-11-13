"""Synthetic cohort preparation checks; never read real images or patient rows."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


_SCRIPT = Path(__file__).resolve().parents[2] / "src/scripts/prepare_grounding_data.py"
_SPEC = importlib.util.spec_from_file_location("grounding_data", _SCRIPT)
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-path-preflight-placeholder")


def _patient(split):
    return next("synthetic-patient-{}".format(index) for index in range(100)
                if builder.report_patient_split("synthetic-patient-{}".format(index)) == split)


def _fixtures(root):
    vindr_root = root / "vindr_images"
    vindr_rows = []
    for fold, split in ((0, "test"), (1, "training"), (2, "training")):
        patient = "shared-source-id-{}".format(fold)
        for label in (0, 1):
            name = "image-{}.png".format(label)
            row = {"patient_id": patient, "image_id": name, "laterality": "L", "split": split,
                   "fold": fold, "Mass": label, "Suspicious_Calcification": label,
                   "Asymmetry": 0, "Focal_Asymmetry": label, "Global_Asymmetry": 0}
            vindr_rows.append(row)
            _image(vindr_root / patient / name)
            if label:
                vindr_rows.append({**row, "Mass": 0, "Suspicious_Calcification": 0, "Focal_Asymmetry": 0})
    vindr_csv = root / "vindr.csv"
    pd.DataFrame(vindr_rows).to_csv(vindr_csv, index=False)
    embed_rows = []
    for fold in (0, 1, 3):
        patient = "shared-source-id-{}".format(fold)
        for label in (0, 1):
            path = root / "embed_images" / patient / ("image-{}.png".format(label))
            _image(path)
            embed_rows.append({"patient_id": patient, "study_id": "exam-{}".format(fold),
                               "anon_dicom_path": str(path.with_suffix(".dcm")), "ImageLateralityFinal": "R",
                               "fold": fold, "mass": label, "calc": label, "asymmetry": label})
    embed_csv = root / "embed.csv"
    pd.DataFrame(embed_rows).to_csv(embed_csv, index=False)
    bu_root = root / "bu_images"
    training, testing = [], []
    for source in ("BU", "UPMC"):
        for split in ("train", "validation", "test"):
            patient = _patient(split) if split != "test" else "heldout-patient"
            exam = "{}-exam-{}".format(source, split)
            for label in (0, 1):
                view = "CC" if label == 0 else "MLO"
                if source == "BU":
                    source_path = root / "old_bu" / "controls" / exam / (view + ".png")
                    destination = bu_root / "controls" / "test_images_png" / exam / (view + ".png")
                else:
                    source_path = root / "UPMC_with_embeddings" / exam / (view + ".png")
                    destination = root / "UPMC" / exam / (view + ".png")
                _image(destination)
                row = {"dataset": source, "patient_id": patient, "exam_id": exam,
                       "image_laterality": "L", "image_view_type": view, "image_path": str(source_path),
                       "mass_present": label, "calc_present": label, "asymmetry_present": label,
                       "report_text": "DO_NOT_COPY_PRIVATE_REPORT_CANARY"}
                (testing if split == "test" else training).append(row)
        # A second exam of a held-out patient must be excluded from the train pool.
        row = dict(testing[-1])
        row["exam_id"] = source + "-overlap-exam"
        row["image_path"] = str(Path(row["image_path"]).with_name("overlap.png"))
        training.append(row)
    train_csv, test_csv = root / "reports_train.csv", root / "reports_test.csv"
    pd.DataFrame(training).to_csv(train_csv, index=False)
    pd.DataFrame(testing).to_csv(test_csv, index=False)
    return {
        "vindr": {"csv": str(vindr_csv), "image_root": str(vindr_root)},
        "embed": {"csv": str(embed_csv)},
        "reports": {"train_csv": str(train_csv), "test_csv": str(test_csv), "bu_image_root": str(bu_root)},
    }


class GroundingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = _fixtures(self.root)

    def frames(self):
        vindr, _ = builder.prepare_vindr(self.config["vindr"])
        embed, _ = builder.prepare_embed(self.config["embed"])
        reports, _ = builder.prepare_reports(self.config["reports"])
        return pd.concat([vindr, embed, *reports], ignore_index=True)

    def test_vindr_dedup_or_and_broad_asymmetry_preserves_official_split(self):
        frame, audit = builder.prepare_vindr(self.config["vindr"])
        self.assertEqual(len(frame), 6)
        self.assertEqual(audit["duplicate_annotation_rows_aggregated"], 3)
        self.assertEqual(frame[list(builder.TASKS)].sum().tolist(), [3, 3, 3])
        self.assertEqual(frame.groupby("split").size().to_dict(), {"test": 2, "train": 2, "validation": 2})
        self.assertEqual(audit["asymmetry_positive_if_any"], ["Asymmetry", "Focal_Asymmetry", "Global_Asymmetry"])

    def test_vindr_duplicate_metadata_conflict_rejected(self):
        path = self.config["vindr"]["csv"]
        frame = pd.read_csv(path)
        frame.loc[2, "fold"] = 1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Conflicting VinDr metadata"):
            builder.prepare_vindr(self.config["vindr"])

    def test_vindr_fold_must_preserve_official_test(self):
        path = self.config["vindr"]["csv"]
        frame = pd.read_csv(path)
        frame.loc[frame["fold"] == 0, "split"] = "training"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "official held-out test"):
            builder.prepare_vindr(self.config["vindr"])

    def test_embed_native_labels_and_fold_three_validation(self):
        frame, audit = builder.prepare_embed(self.config["embed"])
        self.assertEqual(frame.groupby("split").size().to_dict(), {"test": 2, "train": 2, "validation": 2})
        self.assertEqual(frame[list(builder.TASKS)].sum().tolist(), [3, 3, 3])
        self.assertTrue(frame.image_path.str.endswith(".png").all())
        self.assertIn("not established as suspicious-only", audit["calcification_definition"])

    def test_report_test_patient_exclusion_before_deterministic_hash(self):
        frames, audit = builder.prepare_reports(self.config["reports"])
        for frame in frames:
            source = frame.source_dataset.iloc[0]
            self.assertEqual(len(frame), 6)
            self.assertEqual(audit[source]["excluded_training_rows_with_heldout_patient"], 1)
            self.assertEqual(audit[source]["excluded_training_patients_in_test"], 1)
            self.assertTrue(frame.groupby("patient_id")["split"].nunique().eq(1).all())
            self.assertEqual(frame.groupby("split").size().to_dict(), {"test": 2, "train": 2, "validation": 2})
            self.assertTrue(all(Path(path).is_file() for path in frame.image_path))
            self.assertNotIn("report_text", frame)
            self.assertTrue(frame.groupby("exam_id")["asymmetry"].nunique().eq(2).all())
        raw = "00123"
        bucket = int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16) % 4 + 1
        self.assertEqual(builder.report_patient_split(raw), "validation" if bucket == 4 else "train")

    def test_namespaced_identifiers_do_not_merge_datasets(self):
        frame = self.frames()
        self.assertEqual(set(frame.source_dataset), {"vindr", "embed", "bu", "upmc"})
        self.assertTrue(frame.apply(lambda row: row.patient_id.startswith(row.source_dataset + "::"), axis=1).all())
        self.assertTrue(frame.apply(lambda row: row.exam_id.startswith(row.patient_id + "::"), axis=1).all())
        self.assertEqual(builder.validate_pooled(frame)["missing_png_paths"], 0)

    def test_nonbinary_missing_or_invalid_labels_rejected(self):
        for values in ((0, 2), (0, ""), (0, float("nan"))):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                builder._binary(pd.Series(values), "test")
        frame = self.frames()
        frame.loc[0, "asymmetry"] = pd.NA
        with self.assertRaisesRegex(ValueError, "Missing pooled labels"):
            builder.validate_pooled(frame)

    def test_duplicate_images_or_patient_leakage_rejected(self):
        frame = self.frames()
        duplicate = frame.copy()
        duplicate.loc[1, "image_path"] = duplicate.loc[0, "image_path"]
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            builder.validate_pooled(duplicate)
        leak = frame.copy()
        leak.loc[1, "split"] = "train"
        with self.assertRaisesRegex(ValueError, "Patient/group leakage"):
            builder.validate_pooled(leak)

    def test_missing_png_fails_before_writing(self):
        frame = self.frames()
        Path(frame.image_path.iloc[0]).unlink()
        output = self.root / "output"
        with self.assertRaisesRegex(FileNotFoundError, "no CSVs were written"):
            builder.prepare(self.config, output)
        self.assertFalse(output.exists())

    def test_unknown_dataset_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly vindr, embed and reports"):
            builder.prepare({**self.config, "unexpected_source": {}}, self.root / "output")

    def test_private_outputs_task_columns_hashes_and_no_report_text(self):
        output = self.root / "output"
        audit = builder.prepare(self.config, output, path_workers=2)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["global_split_counts"], {"train": 8, "validation": 8, "test": 8})
        self.assertEqual(audit["validation"]["checked_unique_png_paths"], 24)
        self.assertEqual({path.name for path in output.iterdir()},
                         {"audit.json", "grounding_mass.csv", "grounding_calcification.csv", "grounding_asymmetry.csv"})
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        keys = None
        for task in builder.TASKS:
            path = output / ("grounding_{}.csv".format(task))
            frame = pd.read_csv(path)
            self.assertEqual(set(frame.columns), set(builder.OUTPUT_COLUMNS) | {task})
            self.assertEqual(len(frame), 24)
            self.assertEqual(set(frame.source_dataset), set(builder.SOURCES))
            self.assertEqual(audit["manifests"][task]["sha256"], builder.sha256_file(path))
            self.assertEqual(audit["manifests"][task]["path"], str(path))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("DO_NOT_COPY_PRIVATE_REPORT_CANARY", path.read_text())
            current = list(frame[["image_path", "split"]].itertuples(index=False, name=None))
            self.assertTrue(keys is None or keys == current)
            keys = current
        saved = json.loads((output / "audit.json").read_text())
        self.assertEqual(saved, audit)
        self.assertEqual(len(audit["input_sha256"]), 4)
        self.assertNotIn("DO_NOT_COPY_PRIVATE_REPORT_CANARY", (output / "audit.json").read_text())
        before = (output / "audit.json").read_bytes()
        with self.assertRaises(FileExistsError):
            builder.prepare(self.config, output)
        self.assertEqual((output / "audit.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
