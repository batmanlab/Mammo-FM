"""Offline release boundaries for the pooled grounding checkpoint refresh."""

import hashlib
from pathlib import PurePosixPath
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mammofm import MammoFM, exam
from mammofm import _registry as registry
from mammofm import _report_registry as report_registry
from mammofm import _report_model as report_model
from mammofm import _reports as reports
from mammofm._assets import provenance


_PIPELINE_REVISION = "7605810fab99c90eb9c9ff23206c237bafa3f500"
_HISTORICAL_GROUNDING_HASHES = {
    "f3d38fc3398d732bed8ba37b14ef403c36c94cbc53d3635e79c604f6eec58f24",
    "29e623412056d94a12b1ab57c26ab19467498de6058c6f05039cebcbf43ff44a",
    "1652187b6e63420fa4d775acec0e3188c59f9a5f70d814425cfdacf3f65722fa",
}
_STAGE2_HASHES = {
    "adapter_config.json": "dfd37ac97f71de76f51f31c1b5d2ed3c5b764d55a4c70e00655d162b66bcb50f",
    "adapter_model.safetensors": "9bfe5b38ad997167096e62a4477b7cba4ff5f7e3ac97786ad47b9f615f5a68eb",
    "config.json": "38bb94bd29647e9d174f4acdb35f81237a8e4063e7c834980c368c9f7c7a318e",
    "non_lora_trainables.bin": "5f791d265c4557314c30a59988b68588964cb3fc9b11cf765d27664c05ceb345",
    "special_tokens_map.json": "f4b026086f648d35ead59afce0a4937770a8c03a65ba50de73e3bade23f801be",
    "tokenizer.json": "045d2400aeb92b7224f6851838aff4a97a8e6bff11d321c677fc13ffd8044710",
    "tokenizer_config.json": "0ddc7105b04f38a9a9f91b7b8099fc119b2f123f4c6c92f6e48fb2b2845cad2c",
}


class GroundingReleaseTests(unittest.TestCase):
    def test_grounding_uses_three_new_weight_assets_at_one_dedicated_revision(self):
        assets = registry.GROUNDING_CLASSIFIERS
        self.assertEqual(set(assets), {"mass", "suspicious_calcification", "asymmetry"})
        self.assertEqual(
            {asset["filename"] for asset in assets.values()},
            {f"Mammo-GRG/grounding/{task}_best.pth"
             for task in ("mass", "calcification", "asymmetry")},
        )
        revision = getattr(registry, "GROUNDING_REVISION", None)
        self.assertIsInstance(revision, str)
        self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertNotIn(revision, {registry.REVISION, _PIPELINE_REVISION})
        hashes = set()
        for asset in assets.values():
            with self.subTest(filename=asset["filename"]):
                self.assertEqual(asset["revision"], revision)
                self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")
                self.assertNotIn(asset["sha256"], _HISTORICAL_GROUNDING_HASHES)
                self.assertIs(type(asset["size"]), int)
                self.assertGreater(asset["size"], 0)
                hashes.add(asset["sha256"])
        self.assertEqual(len(hashes), 3)

    def test_report_encoder_and_stage2_assets_are_not_repointed(self):
        self.assertEqual(registry.PIPELINE_REVISION, _PIPELINE_REVISION)
        self.assertEqual(registry.REPORT_ENCODER, {
            "filename": "Mammo-FM_BatmanlabTrained_CLIP_last.tar",
            "sha256": "7cf871d116e4cd52a11a5b42d40d4550f078747be0d897265f24cfa5cf26b257",
            "size": 2147265666,
            "revision": _PIPELINE_REVISION,
        })
        actual = {
            PurePosixPath(asset["filename"]).name: asset
            for asset in registry.REPORT_STAGE2_ASSETS
        }
        self.assertEqual(set(actual), set(_STAGE2_HASHES))
        for name, digest in _STAGE2_HASHES.items():
            with self.subTest(filename=name):
                self.assertEqual(actual[name], {
                    "filename": f"Mammo-GRG/stage2/checkpoint-5500/{name}",
                    "sha256": digest,
                    "revision": _PIPELINE_REVISION,
                })

    def test_risk_and_shared_image_encoder_pins_are_unchanged(self):
        self.assertEqual(registry.REVISION, "5e07ca4c4e88cd118b0c2cfaacadddfe69f30026")
        self.assertEqual(registry.ENCODER, {
            "filename": "Mammo-FM_BatmanlabTrained_CLIP.tar",
            "sha256": "58ae959de52d563acc1d43f50571498eddcd4fb7ca599e706540fdd319e3599b",
            "size": 2147265666,
        })
        expected_encoder = {**registry.ENCODER, "revision": _PIPELINE_REVISION}
        self.assertEqual(registry.BU_RISK_ENCODER, expected_encoder)
        self.assertEqual(registry.PROGNOSIS_ENCODER, expected_encoder)
        expected = {
            "mayo": (
                registry.RISK_CHECKPOINTS["mayo"],
                "risk_checkpoints/Mayo-S1_KD_S2/S1S2.pt",
                "6f48c0cdd9b5f32ed279d5ade79e14a2e96db291acb166009fc9929c9a29bb8f",
            ),
            "bu_s1s2": (
                registry.RISK_CHECKPOINTS["bu"]["s1s2"],
                "risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S1_KD_S2_head.pt",
                "6992aa888affb61b8cd897b1ee9e17fae8ef54bf0cb41f38d8935e63675c8273",
            ),
            "bu_s2_only": (
                registry.RISK_CHECKPOINTS["bu"]["s2_only"],
                "risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S2_only_head.pt",
                "cf3e602acf5b0269ca03c7f851cad5d9b933d4a11d17b84a7b334f431a93fa9c",
            ),
        }
        for name, (asset, filename, digest) in expected.items():
            with self.subTest(model=name):
                self.assertEqual(asset["filename"], filename)
                self.assertEqual(asset["sha256"], digest)
                self.assertEqual(asset["revision"], _PIPELINE_REVISION)

    def test_report_grounding_uses_the_shared_release_asset_registry(self):
        self.assertIs(report_registry.GROUNDING_CLASSIFIERS, registry.GROUNDING_CLASSIFIERS)
        self.assertIs(reports.GROUNDING_MODELS, report_registry.GROUNDING_MODELS)
        self.assertEqual(set(reports.GROUNDING_MODELS), set(registry.GROUNDING_CLASSIFIERS))
        for task, asset in registry.GROUNDING_CLASSIFIERS.items():
            with self.subTest(task=task):
                self.assertEqual(provenance(reports.GROUNDING_MODELS[task]), provenance(asset))

    def test_stage3_prompt_and_finding_names_remain_unchanged(self):
        self.assertEqual(report_model.ZS_KEYS, {
            "mass": "mass_pred",
            "suspicious_calcification": "suspicious calcification_pred",
            "asymmetry": "asymmetry_pred",
        })
        self.assertEqual(reports._PREDICTION_KEYS, report_model.ZS_KEYS)
        self.assertEqual(
            hashlib.sha256(report_model.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "bb674f3cebfcd73fe0f186ad0124e2c89a391ec0fccc57fe86e1e398fdd33c2e",
        )

    def test_stage3_aggregation_and_classifier_note_remain_unchanged(self):
        records = []
        for laterality, view, mass, calcification, asymmetry in (
            ("L", "MLO", 1, 0, 0),
            ("L", "CC", 0, 1, 0),
            ("R", "MLO", 1, 0, 0),
            ("R", "CC", 0, 0, 1),
        ):
            records.append({
                "image_laterality": laterality,
                "image_view_type": view,
                "mass_pred": mass,
                "suspicious calcification_pred": calcification,
                "asymmetry_pred": asymmetry,
            })
        views, summary, sides = report_model.build_view_and_side_summaries(records)
        self.assertEqual(tuple(views), ("LMLO", "LCC", "RMLO", "RCC"))
        self.assertEqual(views, {
            "LMLO": {"mass": 1, "suspicious_calcification": 0, "asymmetry": 0},
            "LCC": {"mass": 0, "suspicious_calcification": 1, "asymmetry": 0},
            "RMLO": {"mass": 1, "suspicious_calcification": 0, "asymmetry": 0},
            "RCC": {"mass": 0, "suspicious_calcification": 0, "asymmetry": 1},
        })
        self.assertEqual(summary, {"mass": 1, "suspicious_calcification": 1, "asymmetry": 1})
        self.assertEqual(sides, {
            "mass": {"L": 1, "R": 1},
            "suspicious_calcification": {"L": 1, "R": 0},
            "asymmetry": {"L": 0, "R": 1},
        })
        self.assertEqual(report_model.build_classifier_note(summary, sides),
                         "- Suspicious Mass: bilaterally.\n"
                         "- Suspicious Calcification: Left breast.\n"
                         "- Asymmetry: Right breast.")
        for record in records:
            for key in report_model.ZS_KEYS.values():
                record[key] = 0
        _, summary, sides = report_model.build_view_and_side_summaries(records)
        self.assertEqual(summary, {task: 0 for task in report_model.ZS_KEYS})
        self.assertEqual(report_model.build_classifier_note(summary, sides),
                         "- No positive ML findings across views.")

    def test_exam_report_delegates_to_package_and_preserves_shared_asset_identity(self):
        checkpoints = {task: provenance(asset)
                       for task, asset in registry.GROUNDING_CLASSIFIERS.items()}
        metadata = {
            "visual_encoder": provenance(registry.REPORT_ENCODER),
            "stage2": {name: provenance(asset)
                       for name, asset in report_registry.REPORT_STAGE2_FILES.items()},
            "base_language_model": {"repo_id": report_registry.BASE_LLM_REPO},
            "checkpoints": checkpoints,
            "generation": {"seed": 0},
        }
        generated = {
            "grounded_report": "Findings: synthetic report.\nImpression: synthetic report.",
            "findings": {"metadata": metadata},
        }
        paths = {view: f"synthetic_{view}.png" for view in exam.VIEW_ORDER}
        store = SimpleNamespace(cache_dir=None, model_dir=None, local_files_only=True)
        with patch.object(MammoFM, "report_gen", return_value=generated) as generate:
            result = exam._run_report(
                paths, store, torch.device("cpu"), {}, base_model_path=None,
                seed=0, preliminary_max_new_tokens=512, grounding_max_new_tokens=320,
                progress=lambda _message: None,
            )
        generate.assert_called_once()
        self.assertEqual(generate.call_args.args[0],
                         {view.lower(): paths[view] for view in exam.VIEW_ORDER})
        self.assertEqual(result["models"]["grounding_classifiers"],
                         {task: {"checkpoint": value} for task, value in checkpoints.items()})
        self.assertEqual(result["grounded_report"], generated["grounded_report"])


if __name__ == "__main__":
    unittest.main()
