"""Synthetic and mocked four-view reporting tests; no clinical data or model downloads."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
from PIL import Image
import torch

import mammofm as mf
from mammofm import MammoFM, CheckpointError
from mammofm import _reports as reports
from mammofm._report_registry import (
    BASE_LLM_REPO, BASE_LLM_REVISION, GROUNDING_MODELS, REPORT_ENCODER, REPORT_STAGE2_FILES,
)


def _base_directory(directory):
    config = {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
              "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer.json").write_text("{}")
    (directory / "tokenizer_config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"synthetic-test-placeholder")
    return config


class ReportInputTests(unittest.TestCase):
    def test_explicit_view_names_case_insensitive_and_fixed_order(self):
        paths = {"RCC": "right_cc", "lcc": "left_cc", "RMLO": "right_mlo", "LmLo": "left_mlo"}
        normalized = reports._view_paths(paths)
        self.assertEqual(tuple(normalized), ("lmlo", "lcc", "rmlo", "rcc"))
        self.assertEqual(tuple(normalized.values()), ("left_mlo", "left_cc", "right_mlo", "right_cc"))

    def test_incomplete_ambiguous_or_extra_views_rejected_before_reading(self):
        valid = {view: "unused.png" for view in reports.VIEW_ORDER}
        invalid = ["one.png", ["one.png"] * 4, {}, {"lcc": "x"},
                   {**valid, "LCC": "duplicate"}, {**valid, "other": "x"},
                   {**valid, 3: "x"}, {"L CC": "x", **{k: v for k, v in valid.items() if k != "lcc"}}]
        with patch("mammofm._inputs.load_image") as loader:
            for value in invalid:
                with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                    reports._load_views(value)
            loader.assert_not_called()

    def test_unified_route_and_canonical_processing_order(self):
        image = Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).convert("RGB")
        paths = {view.upper(): view + ".dcm" for view in reversed(reports.VIEW_ORDER)}
        with patch("mammofm._inputs.load_image", return_value=(image, {})) as loader, \
                patch("mammofm._inputs.inference_tensor", return_value=torch.tensor([0., 1.])) as transform:
            result = reports._load_views(paths)
        self.assertEqual(tuple(result), reports.VIEW_ORDER)
        self.assertEqual([call.args for call in loader.call_args_list],
                         [(view + ".dcm", None) for view in reports.VIEW_ORDER])
        for call in transform.call_args_list:
            self.assertEqual(call.kwargs, {"branch": "features", "dataset": None})

    def test_constant_images_rejected_before_any_download(self):
        client = MammoFM(device="cpu")
        client.store.resolve = Mock()
        with patch("mammofm._inputs.load_image", return_value=(Image.new("RGB", (8, 8)), {})), \
                patch.object(reports, "_require_dependencies") as dependencies:
            with self.assertRaisesRegex(ValueError, "constant"):
                reports.report_gen(client, {view: "unused" for view in reports.VIEW_ORDER})
        client.store.resolve.assert_not_called()
        dependencies.assert_not_called()

    def test_variation_lost_by_resize_is_rejected(self):
        image = Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).convert("RGB")
        with patch("mammofm._inputs.load_image", return_value=(image, {})), \
                patch("mammofm._inputs.inference_tensor", return_value=torch.zeros(1, 3, 8, 8)):
            with self.assertRaisesRegex(ValueError, "intensity variation"):
                reports._load_views({view: "unused" for view in reports.VIEW_ORDER})

    def test_generation_argument_validation_precedes_reading(self):
        options = [{"seed": -1}, {"seed": True}, {"seed": 2**32}, {"seed": 0.5},
                   {"max_new_tokens": 0}, {"max_new_tokens": True}, {"grounding_max_new_tokens": -1},
                   {"max_new_tokens": 4097}, {"grounding_max_new_tokens": 4097}]
        with patch.object(reports, "_load_views") as loader:
            for kwargs in options:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    reports.report_gen(MammoFM(), {}, **kwargs)
        loader.assert_not_called()

    def test_missing_optional_dependencies_are_actionable(self):
        with patch("importlib.util.find_spec", side_effect=lambda name: None if name == "peft" else object()):
            with self.assertRaisesRegex(ImportError, r"peft.*mammofm\[report\]"):
                reports._require_dependencies()


class ReportAssetTests(unittest.TestCase):
    def test_local_base_validation_and_safe_shard_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _base_directory(directory)
            self.assertEqual(reports._validate_base_directory(directory), directory)
            index = directory / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
            self.assertEqual(reports._validate_base_directory(directory), directory)
            for filename in ("../outside.safetensors", "/outside.safetensors", "weights.bin",
                             "sub\\weights.safetensors", "missing.safetensors", 123):
                index.write_text(json.dumps({"weight_map": {"weight": filename}}))
                with self.subTest(filename=filename), self.assertRaises(CheckpointError):
                    reports._validate_base_directory(directory)

    def test_wrong_or_custom_base_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = _base_directory(directory)
            for change in ({"vocab_size": 128264}, {"hidden_size": 2048},
                           {"auto_map": {"AutoModel": "custom.Model"}},
                           {"quantization_config": {"load_in_4bit": True}}):
                (directory / "config.json").write_text(json.dumps({**config, **change}))
                with self.subTest(change=change), self.assertRaises(CheckpointError):
                    reports._validate_base_directory(directory)

    def test_missing_base_assets_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(CheckpointError):
                reports._validate_base_directory(directory)
            _base_directory(directory)
            (directory / "model.safetensors").unlink()
            (directory / "pytorch_model.bin").write_bytes(b"not-accepted")
            with self.assertRaisesRegex(CheckpointError, "safetensors"):
                reports._validate_base_directory(directory)

    def test_pinned_base_download_uses_existing_auth_and_safe_allowlist(self):
        store = Mock(cache_dir="cache", local_files_only=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _base_directory(directory)
            with patch("huggingface_hub.snapshot_download", return_value=str(directory)) as download:
                actual, source = reports._resolve_base_model(store, None)
            self.assertEqual(actual, directory)
            self.assertEqual(source, "pinned_huggingface_snapshot")
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs["repo_id"], BASE_LLM_REPO)
            self.assertEqual(kwargs["revision"], BASE_LLM_REVISION)
            self.assertTrue(kwargs["local_files_only"])
            self.assertEqual(kwargs["cache_dir"], "cache")
            self.assertNotIn("token", kwargs)
            self.assertFalse(any(".py" in name or ".bin" in name for name in kwargs["allow_patterns"]))
            with patch("huggingface_hub.snapshot_download") as download:
                self.assertEqual(reports._resolve_base_model(store, directory)[1], "local_override")
                download.assert_not_called()

    def test_gated_model_failure_has_actionable_message(self):
        with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("network unavailable")):
            with self.assertRaisesRegex(CheckpointError, "Accept its license"):
                reports._resolve_base_model(Mock(cache_dir=None, local_files_only=False), None)

    def test_stage2_directory_only_contains_verified_assets_and_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for index, name in enumerate(REPORT_STAGE2_FILES):
                source = root / str(index)
                source.write_bytes(name.encode())
                paths[name] = source
            with reports._stage2_directory(paths) as staged:
                self.assertEqual({path.name for path in staged.iterdir()}, set(REPORT_STAGE2_FILES))
                for name in paths:
                    self.assertEqual((staged / name).read_bytes(), name.encode())
            self.assertFalse(staged.exists())

    def test_stage2_unsafe_filename_rejected(self):
        with self.assertRaises(CheckpointError):
            with reports._stage2_directory({"../outside": "unused"}):
                self.fail("Unsafe staging should not succeed")


class ReportModelOrchestrationTests(unittest.TestCase):
    def test_raw_visual_features_order_shape_and_encoder_checkpoint(self):
        class Encoder:
            def features(self, tensor):
                raw = tensor.reshape(1, 1, 1, 1).expand(1, 2048, 48, 29)
                return {"before_pooling": raw, "after_pooling": torch.tensor([-100.])}
        tensors = {view: torch.tensor(float(index)) for index, view in enumerate(reports.VIEW_ORDER)}
        with patch("mammofm._model.load_encoder", return_value=Encoder()) as loader:
            features = reports._visual_embeddings(tensors, "epoch3.tar", torch.device("cpu"))
        loader.assert_called_once_with("epoch3.tar", device=torch.device("cpu"))
        self.assertEqual(tuple(features.shape), (1, 4, 1, 2048, 48, 29))
        self.assertEqual(features.device.type, "cpu")
        self.assertEqual(features[0, :, 0, 0, 0, 0].tolist(), [0., 1., 2., 3.])
        self.assertFalse(features.requires_grad)

    def test_invalid_feature_shape_or_nonfinite_rejected(self):
        tensors = {view: torch.zeros(1) for view in reports.VIEW_ORDER}
        for raw in (torch.zeros(1, 4), torch.full((1, 2048, 48, 29), float("nan"))):
            encoder = Mock()
            encoder.features.return_value = {"before_pooling": raw}
            with patch("mammofm._model.load_encoder", return_value=encoder):
                with self.assertRaises(RuntimeError):
                    reports._visual_embeddings(tensors, "unused", torch.device("cpu"))

    def test_grounding_thresholds_are_inclusive_and_summaries_bilateral(self):
        thresholds = {task: asset["threshold"] for task, asset in GROUNDING_MODELS.items()}
        values = {
            "mass": [thresholds["mass"], 0., 0.9, 0.],
            "suspicious_calcification": [0., thresholds["suspicious_calcification"], 0., 0.],
            "asymmetry": [0., 0., thresholds["asymmetry"], 0.],
        }
        references = []
        class Classifier:
            def __init__(self, task):
                self.values = iter(values[task])
            def probabilities(self, tensor):
                return torch.tensor([next(self.values)])
        def load(task, **kwargs):
            self.assertTrue(all(reference() is None for reference in references))
            classifier = Classifier(task)
            references.append(weakref.ref(classifier))
            return classifier
        tensors = {view: torch.ones(1) for view in reports.VIEW_ORDER}
        with patch("mammofm._model.load_classifier", side_effect=load) as loader:
            findings, records = reports._grounding_findings(
                tensors, {task: task for task in GROUNDING_MODELS}, torch.device("cpu"))
        self.assertEqual(loader.call_count, 3)
        self.assertTrue(all(reference() is None for reference in references))
        self.assertTrue(findings["views"]["lmlo"]["mass"]["positive"])
        self.assertEqual(findings["left"], {"mass": True, "suspicious_calcification": True, "asymmetry": False})
        self.assertEqual(findings["right"], {"mass": True, "suspicious_calcification": False, "asymmetry": True})
        self.assertEqual(findings["bilateral"], {"mass": True, "suspicious_calcification": False, "asymmetry": False})
        self.assertTrue(all(findings["exam"].values()))
        self.assertEqual(set(findings), {"views", "left", "right", "exam", "bilateral"})
        self.assertEqual(records[0], {"image_laterality": "L", "image_view_type": "MLO", "mass_pred": 1,
                                      "suspicious calcification_pred": 0, "asymmetry_pred": 0})
        self.assertEqual([(record["image_laterality"], record["image_view_type"]) for record in records],
                         [("L", "MLO"), ("L", "CC"), ("R", "MLO"), ("R", "CC")])
        from mammofm._report_model import build_view_and_side_summaries
        view_summary, exam_summary, side_summary = build_view_and_side_summaries(records)
        self.assertEqual(view_summary["LMLO"]["mass"], 1)
        self.assertEqual(exam_summary, {task: int(value) for task, value in findings["exam"].items()})
        self.assertEqual(side_summary["mass"], {"L": 1, "R": 1})
        json.dumps(findings, allow_nan=False)

    def test_bad_grounding_probabilities_never_become_negative_findings(self):
        tensors = {view: torch.ones(1) for view in reports.VIEW_ORDER}
        for values in ([float("nan")], [float("inf")], [-.1], [1.1], [.2, .3]):
            classifier = Mock()
            classifier.probabilities.return_value = torch.tensor(values)
            with patch("mammofm._model.load_classifier", return_value=classifier):
                with self.subTest(values=values), self.assertRaises(RuntimeError):
                    reports._grounding_findings(tensors, {task: task for task in GROUNDING_MODELS}, torch.device("cpu"))

    def test_public_output_and_sequential_backend_contract(self):
        client = MammoFM(device="cpu")
        client.clear_cache = Mock()
        backend = types.ModuleType("mammofm._report_model")
        backend.generate_preliminary = Mock(return_value="INTERNAL_PRELIMINARY")
        backend.generate_grounded = Mock(return_value="  Findings: Grounded.\nImpression: Grounded.  ")
        embeddings = torch.zeros(1, 4, 1, 1, 1, 1)
        records = [{"image_laterality": "L", "image_view_type": "MLO", "mass_pred": 1}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in REPORT_STAGE2_FILES:
                (root / name).write_text("{}")
            client.store.resolve = Mock(side_effect=lambda asset: root / Path(asset["filename"]).name)
            with patch.dict(sys.modules, {"mammofm._report_model": backend}), \
                    patch.object(reports, "_load_views", return_value={}) as inputs, \
                    patch.object(reports, "_require_dependencies"), \
                    patch.object(reports, "_resolve_base_model", return_value=(root, "local_override")), \
                    patch.object(reports, "_visual_embeddings", return_value=embeddings) as encoder, \
                    patch.object(reports, "_grounding_findings", return_value=({"views": {}}, records)), \
                    patch.object(reports, "_release_memory"):
                result = reports.report_gen(client, {"LCC": "private-path-canary"},
                                            base_model_path=root, seed=42, max_new_tokens=13,
                                            grounding_max_new_tokens=17)
            inputs.assert_called_once_with({"LCC": "private-path-canary"})
            client.clear_cache.assert_called_once_with()
            self.assertEqual(encoder.call_args.args[1].name, Path(REPORT_ENCODER["filename"]).name)
            self.assertEqual(backend.generate_preliminary.call_args.args[0].device.type, "cpu")
            self.assertEqual(backend.generate_preliminary.call_args.kwargs, {"max_new_tokens": 13, "seed": 42})
            self.assertFalse(backend.generate_preliminary.call_args.args[1].exists())
            self.assertEqual(backend.generate_grounded.call_args.args[:2], ("INTERNAL_PRELIMINARY", records))
            self.assertEqual(backend.generate_grounded.call_args.kwargs, {"max_new_tokens": 17, "seed": 42})
        self.assertEqual(set(result), {"grounded_report", "findings"})
        self.assertEqual(result["grounded_report"], "Findings: Grounded.\nImpression: Grounded.")
        metadata = result["findings"]["metadata"]
        self.assertEqual(metadata["visual_encoder"]["filename"], REPORT_ENCODER["filename"])
        self.assertEqual(metadata["base_language_model"]["source"], "local_override")
        self.assertEqual(metadata["generation"]["seed"], 42)
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("threshold", serialized)
        self.assertNotIn("cutoff", serialized)
        self.assertNotIn("FPR", serialized)
        self.assertNotIn("INTERNAL_PRELIMINARY", serialized)
        self.assertNotIn("private-path-canary", serialized)
        self.assertNotIn("before_pooling", serialized)
        self.assertNotIn("after_pooling", serialized)

    def test_public_wrapper_preserves_client_and_generation_options(self):
        images = {view: view + ".png" for view in reports.VIEW_ORDER}
        expected = {"grounded_report": "Findings. Impression.", "findings": {}}
        with patch.object(reports, "report_gen", return_value=expected) as implementation:
            result = mf.report_gen(images, device="cpu", local_files_only=True,
                                   base_model_path="base-model", seed=9,
                                   max_new_tokens=16, grounding_max_new_tokens=12)
        self.assertIs(result, expected)
        client, actual_images = implementation.call_args.args
        self.assertIsInstance(client, MammoFM)
        self.assertIs(actual_images, images)
        self.assertTrue(client.store.local_files_only)
        self.assertEqual(implementation.call_args.kwargs,
                         {"base_model_path": "base-model", "seed": 9,
                          "max_new_tokens": 16, "grounding_max_new_tokens": 12})
        mf.clear_cache()

    def test_asset_failure_does_not_clear_existing_client_cache(self):
        client = MammoFM(device="cpu")
        client.clear_cache = Mock()
        client.store.resolve = Mock(side_effect=CheckpointError("Missing asset"))
        backend = types.ModuleType("mammofm._report_model")
        backend.generate_preliminary = Mock()
        backend.generate_grounded = Mock()
        with patch.dict(sys.modules, {"mammofm._report_model": backend}), \
                patch.object(reports, "_load_views", return_value={}), \
                patch.object(reports, "_require_dependencies"):
            with self.assertRaises(CheckpointError):
                reports.report_gen(client, {})
        client.clear_cache.assert_not_called()
        backend.generate_preliminary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
