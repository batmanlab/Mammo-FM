"""Synthetic-only safe checkpoint loading and strict parameter tests."""

from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mammofm._model import _read_checkpoint, _strict_load
from mammofm import _risk


def _bu_release_config():
    return {
        "release_format": "inference_only_head",
        "encoder_frozen": True,
        "encoder_source_hf_filename": "Mammo-FM_ASU_Trained_CLIP.tar",
        "encoder_source_sha256": _risk.BU_RISK_ENCODER["sha256"],
        "view_order": list(_risk.VIEW_ORDER),
        "time_seq": list(_risk.TIME_SEQUENCE),
        "view_seq": list(_risk.VIEW_SEQUENCE),
        "side_seq": list(_risk.SIDE_SEQUENCE),
        "prediction_calibrated": False,
        "source_training_checkpoint_sha256": "a" * 64,
    }


class _UnsupportedGlobal:
    def __reduce__(self):
        # Harmless if executed, but eval must never be accepted by our loader.
        return eval, ("1 + 1",)


class ModelCheckpointSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, payload, name="synthetic.pth"):
        path = self.root / name
        torch.save(payload, path)
        return path

    def test_minimal_checkpoint_loads_without_downloads(self):
        expected = {"weight": torch.arange(6).reshape(2, 3).float()}
        path = self.save({"model": expected, "config": {"base": {"text_max_length": 2048}}})
        state, config = _read_checkpoint(path)
        torch.testing.assert_close(state["weight"], expected["weight"], rtol=0, atol=0)
        self.assertEqual(config, {"base": {"text_max_length": 2048}})

    def test_numpy_training_metadata_is_safely_supported(self):
        path = self.save({"model": {"weight": torch.ones(1)},
                          "validation_score": np.float64(0.75),
                          "validation_predictions": np.array([0.2, 0.8], dtype=np.float32)})
        state, config = _read_checkpoint(path)
        self.assertEqual(set(state), {"weight"})
        self.assertEqual(config, {})

    def test_state_dict_and_bare_state_formats(self):
        for payload in ({"state_dict": {"weight": torch.ones(1)}},
                        {"weight": torch.ones(1)}):
            with self.subTest(wrapped="state_dict" in payload):
                state, _ = _read_checkpoint(self.save(payload))
                self.assertEqual(set(state), {"weight"})

    def test_known_parallel_prefixes_are_removed(self):
        path = self.save({"model": {"module._orig_mod.module.weight": torch.ones(1)}})
        state, _ = _read_checkpoint(path)
        self.assertEqual(set(state), {"weight"})

    def test_duplicate_keys_after_prefix_removal_are_rejected(self):
        path = self.save({"model": {"module.weight": torch.ones(1), "weight": torch.zeros(1)}})
        with self.assertRaisesRegex(ValueError, "Duplicate parameter"):
            _read_checkpoint(path)

    def test_arbitrary_pickle_global_is_rejected_without_fallback(self):
        path = self.save({"model": {"weight": torch.ones(1)}, "metadata": _UnsupportedGlobal()})
        original_load = torch.load
        with patch("mammofm._model.torch.load", wraps=original_load) as checked_load:
            with self.assertRaises(pickle.UnpicklingError):
                _read_checkpoint(path)
        self.assertEqual(checked_load.call_count, 1)
        self.assertIs(checked_load.call_args.kwargs["weights_only"], True)

    def test_torch_below_safe_minimum_is_rejected(self):
        path = self.save({"model": {"weight": torch.ones(1)}})
        with patch.object(torch, "__version__", "2.5.1"):
            with self.assertRaisesRegex(RuntimeError, "PyTorch >= 2.6"):
                _read_checkpoint(path)

    def test_non_tensor_and_empty_state_are_rejected(self):
        for payload in ({"model": {"weight": "not-a-tensor"}}, {"model": {}}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    _read_checkpoint(self.save(payload))

    def test_missing_file_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            _read_checkpoint(self.root / "absent.pth")

    def test_exact_state_loads(self):
        model = nn.Linear(3, 1)
        state = {"weight": torch.ones(1, 3), "bias": torch.zeros(1)}
        _strict_load(model, state)
        self.assertTrue(torch.equal(model.weight, state["weight"]))

    def test_missing_classifier_head_is_rejected(self):
        model = nn.Module()
        model.classifier = nn.Module()
        model.classifier.classification_head = nn.Linear(3, 1)
        state = dict(model.state_dict())
        del state["classifier.classification_head.weight"]
        with self.assertRaisesRegex(ValueError, "missing=.*classifier.classification_head.weight"):
            _strict_load(model, state)

    def test_missing_batchnorm_counter_is_not_silently_synthesized(self):
        model = nn.BatchNorm2d(3)
        state = dict(model.state_dict())
        del state["num_batches_tracked"]
        with self.assertRaisesRegex(ValueError, "num_batches_tracked"):
            _strict_load(model, state)

    def test_unexpected_keys_are_rejected(self):
        model = nn.Linear(3, 1)
        state = dict(model.state_dict())
        state["unexpected"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected=.*unexpected"):
            _strict_load(model, state)

    def test_incorrect_parameter_shape_is_rejected(self):
        model = nn.Linear(3, 1)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            _strict_load(model, {"weight": torch.zeros(1, 4), "bias": torch.zeros(1)})

    def test_bu_head_strictly_combines_with_separate_encoder(self):
        head = _risk.FourViewTransformer()
        state = dict(head.state_dict())
        config = _bu_release_config()
        constructor_encoder = nn.Linear(1, 1)
        released_encoder = nn.Linear(1, 1)
        with patch.object(_risk, "_read_checkpoint", return_value=(state, config)), \
                patch.object(_risk, "_image_encoder", return_value=constructor_encoder), \
                patch.object(
                    _risk, "load_encoder",
                    return_value=SimpleNamespace(image_encoder=released_encoder),
                ) as encoder_loader:
            model = _risk.load_risk_model(
                self.root / "head.pt", encoder_checkpoint=self.root / "encoder.tar")
        self.assertIs(model.image_encoder, released_encoder)
        self.assertFalse(hasattr(model, "risk_calibration_spec"))
        self.assertEqual(set(model.aggregator.state_dict()), set(state))
        encoder_loader.assert_called_once_with(self.root / "encoder.tar", device="cpu")

    def test_bu_head_rejects_incompatible_release_metadata(self):
        config = {
            "release_format": "wrong",
            "encoder_source_hf_filename": "Mammo-FM_ASU_Trained_CLIP.tar",
            "encoder_source_sha256": _risk.BU_RISK_ENCODER["sha256"],
            "view_order": list(_risk.VIEW_ORDER),
            "time_seq": list(_risk.TIME_SEQUENCE),
            "view_seq": list(_risk.VIEW_SEQUENCE),
            "side_seq": list(_risk.SIDE_SEQUENCE),
            "prediction_calibrated": False,
        }
        with patch.object(_risk, "_read_checkpoint", return_value=({}, config)), \
                patch.object(_risk, "_image_encoder", return_value=nn.Linear(1, 1)):
            with self.assertRaisesRegex(ValueError, "unsupported release format"):
                _risk.load_risk_model(
                    self.root / "head.pt", encoder_checkpoint=self.root / "encoder.tar")

    def test_bu_head_rejects_missing_encoder_source_digest(self):
        config = _bu_release_config()
        del config["encoder_source_sha256"]
        with patch.object(_risk, "_read_checkpoint", return_value=({}, config)), \
                patch.object(_risk, "_image_encoder", return_value=nn.Linear(1, 1)), \
                patch.object(_risk, "load_encoder") as encoder_loader:
            with self.assertRaisesRegex(ValueError, "unexpected image encoder"):
                _risk.load_risk_model(
                    self.root / "head.pt", encoder_checkpoint=self.root / "encoder.tar")
        encoder_loader.assert_not_called()

    def test_bu_head_rejects_wrong_encoder_source_digest(self):
        for digest in (None, "", "0" * 64, "not-a-sha256"):
            with self.subTest(digest=digest):
                config = _bu_release_config()
                config["encoder_source_sha256"] = digest
                with patch.object(_risk, "_read_checkpoint", return_value=({}, config)), \
                        patch.object(_risk, "_image_encoder", return_value=nn.Linear(1, 1)), \
                        patch.object(_risk, "load_encoder") as encoder_loader:
                    with self.assertRaisesRegex(ValueError, "unexpected image encoder"):
                        _risk.load_risk_model(
                            self.root / "head.pt", encoder_checkpoint=self.root / "encoder.tar")
                encoder_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
