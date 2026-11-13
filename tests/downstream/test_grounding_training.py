"""Synthetic regression checks for pooled Stage-3 grounding training."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "codebase" / "Classifiers" / "grounding.py"
)
SPEC = importlib.util.spec_from_file_location("grounding_under_test", MODULE_PATH)
grounding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grounding)


def synthetic_manifest(sources=("bu", "upmc")):
    rows = []
    for source in sources:
        for split in ("train", "validation", "test"):
            for label in (0, 1):
                patient = f"{source}:{split}:patient{label}"
                exam = f"{patient}:exam0"
                for view in ("CC", "MLO"):
                    rows.append({
                        "source_dataset": source,
                        "patient_id": patient,
                        "exam_id": exam,
                        "laterality": "L",
                        "view": view,
                        "image_path": f"/synthetic/png/{source}/{split}/{label}_{view}.png",
                        "split": split,
                        "mass": label,
                        "calcification": label,
                        "asymmetry": label,
                    })
    return pd.DataFrame(rows)


class GroundingTrainingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mammofm-grounding-test-")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.args = SimpleNamespace(
            label="mass", arch="breast_clip_det_b5_period_n_ft", cur_fold=0,
            data_dir=self.directory, csv_file="manifest.csv",
            output_path=self.directory, seed=10,
        )
        self.frame = synthetic_manifest()
        self.frame.to_csv(self.directory / self.args.csv_file, index=False)

    def prepare(self, frame=None):
        with redirect_stdout(io.StringIO()):
            grounding.prepare_splits(self.args, self.frame if frame is None else frame)

    def test_valid_manifest_preserves_fixed_splits_and_records_hash(self):
        self.prepare()
        self.assertEqual(len(self.args.train_folds), 8)
        self.assertEqual(len(self.args.valid_folds), 8)
        self.assertEqual(len(self.args.test_folds), 8)
        self.assertEqual(set(self.args.train_folds.split), {"train"})
        self.assertEqual(set(self.args.valid_folds.split), {"validation"})
        self.assertEqual(set(self.args.test_folds.split), {"test"})
        self.assertRegex(self.args.grounding_manifest_sha256, r"^[0-9a-f]{64}$")

    def test_cmmd_is_rejected_for_every_grounding_task(self):
        frame = synthetic_manifest(sources=("cmmd", "bu"))
        for task in ("mass", "calcification", "asymmetry"):
            with self.subTest(task=task):
                self.args.label = task
                with self.assertRaisesRegex(ValueError, r"CMMD|unsupported source"):
                    self.prepare(frame)

    def test_linear_probe_is_rejected(self):
        self.args.arch = "breast_clip_det_b5_period_n_lp"
        with self.assertRaisesRegex(ValueError, "full fine-tuning"):
            self.prepare()

    def test_unknown_task_is_rejected(self):
        self.args.label = "density"
        with self.assertRaisesRegex(ValueError, "Unknown grounding task"):
            self.prepare()

    def test_unknown_labels_are_not_filled_with_negative(self):
        for value in (-1, 2, "unknown", np.nan):
            with self.subTest(value=value):
                frame = self.frame.copy()
                frame["mass"] = frame["mass"].astype(object)
                frame.loc[0, "mass"] = value
                with self.assertRaisesRegex(ValueError, "binary values|missing identifiers"):
                    self.prepare(frame)

    def test_patient_leakage_is_rejected(self):
        frame = self.frame.copy()
        training_patient = frame.loc[frame.split.eq("train"), "patient_id"].iloc[0]
        validation_row = frame.index[frame.split.eq("validation")][0]
        frame.loc[validation_row, "patient_id"] = training_patient
        with self.assertRaisesRegex(ValueError, "patient leakage"):
            self.prepare(frame)

    def test_exam_leakage_is_rejected_even_with_distinct_patient_ids(self):
        frame = self.frame.copy()
        training_exam = frame.loc[frame.split.eq("train"), "exam_id"].iloc[0]
        validation_row = frame.index[frame.split.eq("validation")][0]
        frame.loc[validation_row, "exam_id"] = training_exam
        with self.assertRaisesRegex(ValueError, "exam leakage"):
            self.prepare(frame)

    def test_patient_identifiers_must_be_source_qualified(self):
        frame = self.frame.copy()
        bu_patient = frame.loc[frame.source_dataset.eq("bu"), "patient_id"].iloc[0]
        upmc_row = frame.index[frame.source_dataset.eq("upmc")][0]
        frame.loc[upmc_row, "patient_id"] = bu_patient
        with self.assertRaisesRegex(ValueError, "qualified by source"):
            self.prepare(frame)

    def test_duplicate_image_paths_are_rejected(self):
        frame = self.frame.copy()
        frame.loc[1, "image_path"] = frame.loc[0, "image_path"]
        with self.assertRaisesRegex(ValueError, "Duplicate images"):
            self.prepare(frame)

    def test_source_split_needs_both_classes(self):
        frame = self.frame.copy()
        frame.loc[frame.source_dataset.eq("upmc") & frame.split.eq("validation"), "mass"] = 0
        with self.assertRaisesRegex(ValueError, "upmc/validation/mass must have both classes"):
            self.prepare(frame)

    def test_aggregation_preserves_longitudinal_exams_and_breast_sides(self):
        frame = pd.DataFrame({
            "source_dataset": ["bu"] * 6,
            "patient_id": ["bu:synthetic-patient"] * 6,
            "exam_id": ["bu:exam0"] * 2 + ["bu:exam1"] * 4,
            "laterality": ["L", "L", "L", "L", "R", "R"],
            "mass": [0, 0, 0, 1, 0, 0],
            "prediction": [0.1, 0.2, 0.6, 0.9, 0.3, 0.4],
        })
        result = grounding.aggregate(self.args, frame).set_index(["exam_id", "laterality"])
        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[("bu:exam0", "L"), "mass"], 0)
        self.assertAlmostEqual(result.loc[("bu:exam0", "L"), "prediction"], 0.2)
        self.assertEqual(result.loc[("bu:exam1", "L"), "mass"], 1)
        self.assertAlmostEqual(result.loc[("bu:exam1", "L"), "prediction"], 0.9)
        self.assertAlmostEqual(result.loc[("bu:exam1", "R"), "prediction"], 0.4)

    def test_validation_threshold_is_finite_and_within_probability_range(self):
        for probabilities in ([0.1, 0.2, 0.8, 0.9], [0.5, 0.5, 0.5, 0.5]):
            with self.subTest(probabilities=probabilities):
                frame = pd.DataFrame({"mass": [0, 0, 1, 1], "prediction": probabilities})
                threshold = grounding.validation_threshold(frame, "mass")
                self.assertTrue(np.isfinite(threshold))
                self.assertGreaterEqual(threshold, 0)
                self.assertLessEqual(threshold, 1)
        frame = pd.DataFrame({"mass": [0, 0, 1, 1], "prediction": [0.1, 0.2, 0.8, 0.9]})
        self.assertAlmostEqual(grounding.validation_threshold(frame, "mass"), 0.8)

    def test_invalid_validation_predictions_cannot_select_a_threshold(self):
        for values in ([0.1, np.nan, 0.8, 0.9], [0.1, np.inf, 0.8, 0.9]):
            with self.subTest(values=values):
                frame = pd.DataFrame({"mass": [0, 0, 1, 1], "prediction": values})
                with self.assertRaisesRegex(ValueError, "finite validation probabilities"):
                    grounding.validation_threshold(frame, "mass")
        with self.assertRaises(ValueError):
            grounding.validation_threshold(pd.DataFrame({"mass": [0, 0], "prediction": [0.1, 0.2]}), "mass")

    def prediction_frames(self):
        validation = self.frame[self.frame.split.eq("validation")].copy()
        test = self.frame[self.frame.split.eq("test")].copy()
        validation["prediction"] = [0.15, 0.25, 0.75, 0.85, 0.35, 0.45, 0.55, 0.65]
        test["prediction"] = [0.1, 0.2, 0.8, 0.9, 0.9, 0.8, 0.2, 0.1]
        return validation, test

    def test_artifacts_contain_correct_per_source_aurocs(self):
        self.prepare()
        validation, test = self.prediction_frames()
        with redirect_stdout(io.StringIO()):
            report = grounding.write_evaluation_artifacts(
                self.args, validation, test, self.directory / "synthetic_model.pth"
            )
        self.assertEqual(report["image_size_hw"], [1520, 912])
        self.assertEqual(report["manifest_sha256"], self.args.grounding_manifest_sha256)
        self.assertEqual(report["threshold_source"], "validation_images_only")
        self.assertEqual(report["threshold_comparison"], "probability >= threshold")
        self.assertAlmostEqual(report["threshold"], 0.55)
        for source in ("bu", "upmc"):
            for split in ("validation", "test"):
                entry = report["by_source"][source][split]
                self.assertEqual(entry["images"], 4)
                self.assertEqual(entry["exam_breasts"], 2)
                self.assertEqual(entry["patients"], 2)
                self.assertEqual(entry["patient_unit"], "patient")
                self.assertIsNone(entry["patient_auroc_undefined_reason"])
                expected = 0.0 if source == "upmc" and split == "test" else 1.0
                self.assertAlmostEqual(entry["image_auroc"], expected)
                self.assertAlmostEqual(entry["exam_breast_auroc"], expected)
                self.assertAlmostEqual(entry["patient_auroc"], expected)
        saved = self.directory / "seed_10_fold0_mass_grounding.json"
        self.assertEqual(json.loads(saved.read_text()), report)

    def test_source_patient_auroc_can_be_undefined_without_losing_exam_metrics(self):
        self.prepare()
        frame = pd.DataFrame({
            "source_dataset": ["vindr", "vindr"],
            "patient_id": ["vindr:study", "vindr:study"],
            "exam_id": ["vindr:study", "vindr:study"],
            "laterality": ["L", "R"],
            "mass": [0, 1], "prediction": [0.1, 0.9],
        })
        with redirect_stdout(io.StringIO()):
            report = grounding.write_evaluation_artifacts(
                self.args, frame, frame, self.directory / "synthetic_model.pth"
            )
        entry = report["by_source"]["vindr"]["test"]
        self.assertEqual(entry["patients"], 1)
        self.assertEqual(entry["patient_unit"], "study")
        self.assertIsNone(entry["patient_auroc"])
        self.assertIn("both positive and negative", entry["patient_auroc_undefined_reason"])
        self.assertEqual(entry["exam_breast_auroc"], 1.0)
        json.dumps(report, allow_nan=False)

    def test_test_predictions_do_not_influence_validation_operating_point(self):
        self.prepare()
        validation, test = self.prediction_frames()
        changed_test = test.copy()
        changed_test["prediction"] = 1 - changed_test["prediction"]
        with redirect_stdout(io.StringIO()):
            first = grounding.write_evaluation_artifacts(
                self.args, validation, test, self.directory / "synthetic_model.pth"
            )
            second = grounding.write_evaluation_artifacts(
                self.args, validation, changed_test, self.directory / "synthetic_model.pth"
            )
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertNotEqual(first["by_source"]["bu"]["test"], second["by_source"]["bu"]["test"])


if __name__ == "__main__":
    unittest.main()
