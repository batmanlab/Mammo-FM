"""Synthetic checks for binary AUROC reporting units and unchanged selection."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "codebase"))
from Classifiers import experiments


class BinaryAUROCReportingTests(unittest.TestCase):
    @staticmethod
    def args(dataset="RSNA", label="mass", **kwargs):
        return SimpleNamespace(dataset=dataset, label=label, **kwargs)

    @staticmethod
    def rsna_frame():
        return pd.DataFrame({
            "patient_id": ["one"] * 4 + ["two"] * 4,
            "image_id": list(range(8)),
            "laterality": ["L", "L", "R", "R"] * 2,
            "mass": [0, 0, 1, 1, 0, 0, 0, 0],
            "prediction": [.1, .9, .6, .7, .2, .3, .4, .5],
        })

    def test_rsna_three_levels_and_breast_mean_are_preserved(self):
        args, frame = self.args(), self.rsna_frame()
        original = frame.copy(deep=True)
        report = experiments._binary_auroc_levels(args, frame)
        self.assertEqual([report[level]["n"] for level in report], [8, 2, 4])
        self.assertAlmostEqual(report["image"]["auroc"], roc_auc_score(frame.mass, frame.prediction))
        self.assertEqual(report["patient"]["prediction_reduction"], "max")
        self.assertEqual(report["patient"]["auroc"], 1.0)
        self.assertEqual(report["current_grouping"]["prediction_reduction"], "mean")
        self.assertEqual(report["current_grouping"]["auroc"], experiments._auroc(args, frame))
        pd.testing.assert_frame_equal(frame, original)
        json.dumps(report, allow_nan=False)

    def test_cmmd_uses_patient_dicom_and_breast_keys(self):
        frame = self.rsna_frame().rename(columns={"image_id": "dicom_name", "laterality": "LeftRight"})
        report = experiments._binary_auroc_levels(self.args("CMMD"), frame)
        self.assertEqual(report["image"]["grouping_columns"], ["patient_id", "dicom_name"])
        self.assertEqual(report["patient"]["grouping_columns"], ["patient_id"])
        self.assertEqual(report["current_grouping"]["grouping_columns"], ["patient_id", "LeftRight"])
        self.assertEqual(report["current_grouping"]["prediction_reduction"], "mean")

    def test_vindr_unique_images_take_any_positive_annotation_and_max_probability(self):
        frame = pd.DataFrame({
            "patient_id": ["study1", "study1", "study1", "study2"],
            "image_id": ["a", "a", "b", "c"], "laterality": ["L", "L", "L", "R"],
            "mass": [0, 1, 0, 0], "prediction": [.9, .9, .3, .1],
        })
        report = experiments._binary_auroc_levels(self.args("VinDr"), frame)
        self.assertEqual(report["image"]["n"], 3)
        self.assertEqual(report["image"]["positives"], 1)
        self.assertEqual(report["image"]["input_rows"], 4)
        self.assertEqual(report["image"]["duplicate_rows_collapsed"], 1)
        self.assertEqual(report["image"]["auroc"], 1.0)
        self.assertNotEqual(report["image"]["auroc"], roc_auc_score(frame.mass, frame.prediction))
        self.assertEqual(report["patient"]["unit"], "study")
        self.assertEqual(report["current_grouping"]["unit"], "study_breast")
        self.assertEqual(report["current_grouping"]["auroc"], experiments._auroc(self.args("VinDr"), frame))

    def test_nlbreast_patient_or_study_group_spans_both_breasts(self):
        frame = pd.DataFrame({
            "group_id": ["patient:one", "patient:one", "study:two", "study:two"],
            "image_path": ["/one/L.png", "/one/R.png", "/two/L.png", "/two/R.png"],
            "mass": [0, 1, 0, 0], "prediction": [.1, .8, .2, .3],
        })
        report = experiments._binary_auroc_levels(self.args("NLBreast"), frame)
        self.assertEqual(report["patient"], report["current_grouping"])
        self.assertEqual(report["patient"]["n"], 2)
        self.assertEqual(report["patient"]["unit"], "patient_or_study_group")
        frame = frame.rename(columns={"group_id": "folder_name"})
        fallback = experiments._binary_auroc_levels(self.args("NLBreast"), frame)
        self.assertEqual(fallback["patient"]["grouping_columns"], ["folder_name"])

    def test_lumina_cohort_qualified_patient_and_breast_units(self):
        frame = pd.DataFrame({
            "Cancer": ["Benign", "Benign", "Malignant", "Malignant"],
            "patient_id": [1, 1, 1, 1],
            "image_full_path": ["/benign/1/LCC.dcm", "/benign/1/LMLO.dcm",
                                "/malignant/1/LCC.dcm", "/malignant/1/LMLO.dcm"],
            "mass": [0, 0, 1, 1], "prediction": [.1, .2, .7, .8],
        })
        args = self.args("LUMINA")
        report = experiments._binary_auroc_levels(args, frame)
        self.assertEqual(report["patient"]["grouping_columns"], ["Cancer", "patient_id"])
        self.assertEqual(report["current_grouping"]["grouping_columns"], ["Cancer", "patient_id", "_laterality"])
        self.assertEqual([report[level]["n"] for level in report], [4, 2, 2])
        self.assertEqual(experiments._auroc(args, frame), 1.0)

    def test_grounding_source_qualified_patients_preserve_exam_breast_selection(self):
        frame = pd.DataFrame({
            "source_dataset": ["bu", "bu", "bu", "upmc", "upmc", "upmc"],
            "patient_id": ["one"] * 6,
            "exam_id": ["first", "first", "second", "first", "first", "second"],
            "laterality": ["L"] * 6,
            "image_path": ["/a", "/b", "/c", "/a", "/b", "/c"],
            "mass": [0, 0, 1, 0, 0, 0], "prediction": [.1, .2, .9, .2, .3, .4],
        })
        args = self.args("Grounding")
        report = experiments._binary_auroc_levels(args, frame)
        self.assertEqual([report[level]["n"] for level in report], [6, 2, 4])
        self.assertEqual(report["patient"]["grouping_columns"], ["source_dataset", "patient_id"])
        self.assertEqual(report["current_grouping"]["auroc"], experiments._auroc(args, frame))

    def test_single_class_supplementary_level_is_explicitly_undefined(self):
        frame = self.rsna_frame().iloc[:4].copy()
        report = experiments._binary_auroc_levels(self.args(), frame)
        self.assertIsNone(report["patient"]["auroc"])
        self.assertIn("both positive and negative", report["patient"]["undefined_reason"])
        self.assertIsInstance(report["current_grouping"]["auroc"], float)
        json.dumps(report, allow_nan=False)
        output = io.StringIO()
        with redirect_stdout(output):
            experiments._print_binary_auroc_levels("Held-out test", report)
        self.assertIn("patient AUROC: undefined (single class)", output.getvalue())

    def test_missing_keys_and_invalid_labels_or_predictions_fail_closed(self):
        frame = self.rsna_frame()
        with self.assertRaises(KeyError):
            experiments._binary_auroc_levels(self.args(), frame.drop(columns="image_id"))
        for column, value in (("patient_id", None), ("image_id", ""), ("mass", np.nan),
                              ("mass", 2), ("prediction", np.nan), ("prediction", np.inf),
                              ("prediction", -.1), ("prediction", 1.1)):
            invalid = frame.copy()
            invalid[column] = invalid[column].astype(object)
            invalid.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                experiments._binary_auroc_levels(self.args(), invalid)

    def test_saved_binary_metrics_are_finite_identifier_free_json(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(output_path=directory, seed=10, cur_fold=0)
            with redirect_stdout(io.StringIO()):
                experiments._save_predictions(args, self.rsna_frame(), "test")
            path = Path(directory) / "seed_10_fold0_mass_test_metrics.json"
            report = json.loads(path.read_text())
            self.assertEqual(report["checkpoint_selection_level"], "current_grouping")
            self.assertEqual(report["auroc_levels"]["image"]["n"], 8)
            self.assertNotIn('"one"', path.read_text())
            self.assertNotIn('"two"', path.read_text())

    def test_density_metric_and_saved_output_behavior_stay_unchanged(self):
        frame = pd.DataFrame({"patient_id": np.repeat(np.arange(4), 2),
                              "laterality": ["L"] * 8, "density": np.repeat(np.arange(4), 2)})
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(label="density", output_path=directory, seed=10, cur_fold=0)
            frame = experiments._attach_predictions(args, frame, np.repeat(np.eye(4), 2, axis=0))
            self.assertEqual(experiments._auroc(args, frame), 1.0)
            with self.assertRaisesRegex(ValueError, "does not apply to density"):
                experiments._binary_auroc_levels(args, frame)
            with redirect_stdout(io.StringIO()):
                experiments._save_predictions(args, frame, "test")
            self.assertEqual([path.name for path in Path(directory).iterdir()],
                             ["seed_10_fold0_density_test_predictions.csv"])

    def test_training_selects_current_grouping_and_serializes_three_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(
                train_folds=self.rsna_frame(), valid_folds=self.rsna_frame(),
                test_folds=self.rsna_frame(), cur_fold=0, seed=10, data_frac=1,
                clip_chk_pt_path="synthetic.tar", arch="breast_clip_det_b5_period_n_ft",
                model_base_name="synthetic", lr=1e-4, weight_decay=0, warmup_epochs=0,
                epochs=2, weighted_BCE="n", tb_logs_path=root, chk_pt_path=root,
                output_path=root, VER="test",
            )
            model = torch.nn.Linear(1, 1)
            saved = []
            fake_clip = {"config": {"model": {"image_encoder": {"model_type": "cnn", "name": "synthetic"}}}}
            probabilities = self.rsna_frame().prediction.to_numpy()
            lower = experiments._binary_auroc_levels(args, self.rsna_frame())
            higher = json.loads(json.dumps(lower))
            lower["image"]["auroc"], higher["image"]["auroc"] = .99, .01

            def save(checkpoint, path):
                saved.append(checkpoint)

            def load(path, **kwargs):
                return fake_clip if str(path) == "synthetic.tar" else saved[-1]

            with patch.object(experiments, "BreastClipClassifier", return_value=model), \
                    patch.object(experiments, "get_dataloader", return_value=([0], [0])), \
                    patch.object(experiments, "SummaryWriter", return_value=Mock()), \
                    patch.object(experiments, "LinearWarmupCosineAnnealingLR", return_value=Mock()), \
                    patch.object(experiments, "train_fn"), \
                    patch.object(experiments, "valid_fn", return_value=(0, probabilities)), \
                    patch.object(experiments, "_auroc", side_effect=[.6, .8, .7]), \
                    patch.object(experiments, "_binary_auroc_levels", side_effect=[lower, higher, higher, higher, higher, higher, higher]), \
                    patch.object(experiments.torch, "save", side_effect=save), \
                    patch.object(experiments.torch, "load", side_effect=load), \
                    redirect_stdout(io.StringIO()) as output:
                experiments.train_loop(args, torch.device("cpu"))
            self.assertEqual(saved[-1]["epoch"], 1)
            self.assertEqual(saved[-1]["auroc"], .8)
            self.assertEqual(saved[-1]["validation_auroc_levels"]["image"]["auroc"], .01)
            complete = json.loads((root / "seed_10_fold0_mass_complete.json").read_text())
            self.assertEqual(complete["selected_epoch"], 2)
            self.assertEqual(complete["validation_auroc"], .8)
            self.assertEqual(complete["test_auroc"], .7)
            self.assertEqual(set(complete["test_auroc_levels"]), {"image", "patient", "current_grouping"})
            self.assertIn("Held-out test AUROC: 0.700000", output.getvalue())
            self.assertIn("Epoch 2 validation image AUROC", output.getvalue())


if __name__ == "__main__":
    unittest.main()
