"""CPU regression tests for density and cancer-subtype training semantics."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "codebase"))
from Classifiers.experiments import (
    _aggregate_for_auroc, _assert_disjoint_splits, _attach_predictions,
    _auroc, _loss_and_probabilities, _prepare_task_labels,
)
from Classifiers.models.breast_clip_classifier import BreastClipClassifier


class DensityTrainingTests(unittest.TestCase):
    def args(self, dataset="RSNA", label="density", **kwargs):
        return SimpleNamespace(dataset=dataset, label=label, **kwargs)

    def test_unknown_density_is_not_class_a(self):
        out = _prepare_task_labels(self.args(), pd.DataFrame({"density": ["A", "D", None]}))
        self.assertEqual(out.density.tolist(), [0, 3])

    def test_invalid_density_is_rejected(self):
        with self.assertRaises(ValueError):
            _prepare_task_labels(self.args(), pd.DataFrame({"density": ["A", "unknown"]}))

    def test_invasive_excludes_healthy_and_unknown(self):
        frame = pd.DataFrame({"cancer": [0, 1, 1, 1], "invasive": [0, 0, 1, np.nan]})
        out = _prepare_task_labels(self.args(label="invasive"), frame)
        self.assertEqual(out.invasive.tolist(), [0, 1])
        self.assertTrue((out.cancer == 1).all())

    def test_vindr_density_removes_annotation_duplicates(self):
        frame = pd.DataFrame({"patient_id": [1, 1], "image_id": [2, 2],
                              "density": [2, 2], "laterality": ["L", "L"], "split": ["train", "train"]})
        self.assertEqual(len(_prepare_task_labels(self.args("VinDr"), frame)), 1)
        frame.loc[1, "density"] = 3
        with self.assertRaises(ValueError):
            _prepare_task_labels(self.args("VinDr"), frame)

    def test_lumina_join_uses_cohort_not_just_numeric_id(self):
        frame = pd.DataFrame({"Cancer": ["Benign", "Malignant"], "patient_id": [1, 1], "fold": [0, 0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "density.csv"
            frame.assign(density=[0, 3]).to_csv(path, index=False)
            out = _prepare_task_labels(self.args("LUMINA", density_metadata=str(path)), frame)
        self.assertEqual(out.density.tolist(), [0, 3])
        self.assertEqual(out.fold.tolist(), [0, 0])

    def test_density_loss_and_probabilities(self):
        logits = torch.randn(8, 4, requires_grad=True)
        loss, probabilities = _loss_and_probabilities(
            self.args(), torch.nn.CrossEntropyLoss(), logits, torch.arange(8) % 4)
        loss.backward()
        self.assertEqual(tuple(logits.grad.shape), (8, 4))
        torch.testing.assert_close(probabilities.sum(1), torch.ones(8))

    def test_binary_loss_is_unchanged(self):
        logits = torch.tensor([[1.0], [-1.0]], requires_grad=True)
        labels = torch.tensor([1, 0])
        criterion = torch.nn.BCEWithLogitsLoss()
        loss, probabilities = _loss_and_probabilities(self.args(label="cancer"), criterion, logits, labels)
        torch.testing.assert_close(loss, criterion(logits, labels.float().view(-1, 1)))
        torch.testing.assert_close(probabilities, logits.view(-1).sigmoid())

    def test_density_auroc_breast_aggregation(self):
        frame = pd.DataFrame({"patient_id": np.repeat(np.arange(4), 2),
                              "laterality": ["L"] * 8, "density": np.repeat(np.arange(4), 2)})
        probabilities = np.repeat(np.eye(4) * .8 + .05, 2, axis=0)
        out = _attach_predictions(self.args(), frame, probabilities)
        aggregated = _aggregate_for_auroc(self.args(), out)
        self.assertEqual(len(aggregated), 4)
        self.assertAlmostEqual(_auroc(self.args(), out), 1.0)
        out.loc[1, "density"] = 1
        with self.assertRaises(ValueError):
            _aggregate_for_auroc(self.args(), out)

    def test_lumina_aggregation_keeps_reused_ids_separate(self):
        frame = pd.DataFrame({"Cancer": ["Benign", "Malignant"], "patient_id": [1, 1],
                              "image_full_path": ["/LCC.dcm", "/LCC.dcm"], "density": [0, 3]})
        out = _attach_predictions(self.args("LUMINA"), frame, np.eye(4)[[0, 3]])
        self.assertEqual(len(_aggregate_for_auroc(self.args("LUMINA"), out)), 2)

    def test_missing_density_class_is_not_silently_scored(self):
        frame = pd.DataFrame({"patient_id": [1, 2], "laterality": ["L", "L"], "density": [0, 1]})
        out = _attach_predictions(self.args(), frame, np.eye(4)[:2])
        with self.assertRaises(ValueError):
            _auroc(self.args(), out)

    def test_patient_leakage_is_rejected(self):
        args = self.args()
        args.df = pd.DataFrame({"patient_id": [1, 2, 3]})
        args.train_folds = args.df.iloc[:1]
        args.valid_folds = args.df.iloc[1:2]
        args.test_folds = args.df.iloc[2:]
        _assert_disjoint_splits(args)
        args.test_folds = args.train_folds.copy()
        with self.assertRaises(ValueError):
            _assert_disjoint_splits(args)

    def test_linear_probe_freezes_encoder_buffers_but_not_head(self):
        model = BreastClipClassifier.__new__(BreastClipClassifier)
        torch.nn.Module.__init__(model)
        model.arch = "breast_clip_det_b5_period_n_lp"
        model.image_encoder = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(.5))
        model.classifier = torch.nn.Linear(4, 4)
        model.train()
        self.assertFalse(model.image_encoder.training)
        self.assertTrue(model.classifier.training)
        before = model.image_encoder[0].running_mean.clone()
        model.image_encoder(torch.ones(8, 4))
        torch.testing.assert_close(model.image_encoder[0].running_mean, before)
        model.arch = "breast_clip_det_b5_period_n_ft"
        model.train()
        self.assertTrue(model.image_encoder.training)
        model.eval()
        self.assertFalse(model.classifier.training)


if __name__ == "__main__":
    unittest.main()
