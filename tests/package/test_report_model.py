"""Portable report-backend tests using synthetic tensors and mocked language models."""

import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import weakref

import torch

from mammofm import CheckpointError
from mammofm import _report_model as backend


TOKENS = ("<pad>", "<multiple_choice>", "<long_answer>", "<report_generation>",
          "<LMLO>", "<LCC>", "<RMLO>", "<RCC>")


def _findings():
    return [{"image_laterality": view[0], "image_view_type": view[1:],
             "mass_pred": 0, "suspicious calcification_pred": 0, "asymmetry_pred": 0}
            for view in backend.VIEW_ORDER]


def _meta_non_lora():
    return {name: torch.empty(shape, device="meta", dtype=torch.bfloat16)
            for name, shape in backend._non_lora_shapes().items()}


def _adapter_config():
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 128,
            "lora_alpha": 256, "bias": "none", "use_dora": False, "use_rslora": False,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                               "up_proj", "down_proj", "lesion_head"]}


class _Tokenizer:
    pad_token_id = 128256
    eos_token_id = 128009

    def __init__(self, result="Synthetic generated report."):
        self.result = result

    def __len__(self):
        return 128264

    def convert_tokens_to_ids(self, token):
        return 128256 + TOKENS.index(token)

    def decode(self, token_ids, **kwargs):
        return self.result

    def apply_chat_template(self, messages, **kwargs):
        return "Synthetic chat prompt."

    def __call__(self, text, **kwargs):
        class Batch(dict):
            def to(self, device):
                return self
        return Batch(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2))


class _Model:
    dtype = torch.float32

    def eval(self):
        return self

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


class ReportTensorTopologyTests(unittest.TestCase):
    def test_non_lora_has_exact_released_shapes_without_storage(self):
        shapes = backend._non_lora_shapes()
        self.assertEqual(len(shapes), 16)
        self.assertEqual(shapes["model.embed_tokens.weight"], (128264, 4096))
        self.assertEqual(shapes["model.mm_projector.attn_pool.query"], (512, 2048))
        self.assertEqual(shapes["model.mm_projector.view_positional_embeddings"], (4, 4096))
        self.assertEqual(shapes["model.mm_projector.proj.2.weight"], (4096, 4096))
        state = _meta_non_lora()
        self.assertTrue(all(value.is_meta for value in state.values()))
        backend._validate_non_lora(state)

    def test_non_lora_missing_extra_wrong_shape_and_integer_rejected(self):
        key = "model.embed_tokens.weight"
        for mutation in ("missing", "extra", "shape", "integer"):
            state = _meta_non_lora()
            if mutation == "missing":
                del state[key]
            elif mutation == "extra":
                state["unexpected"] = torch.empty(1, device="meta")
            elif mutation == "shape":
                state[key] = torch.empty(1, device="meta")
            else:
                state[key] = torch.empty((128264, 4096), device="meta", dtype=torch.int32)
            with self.subTest(mutation=mutation), self.assertRaises(CheckpointError):
                backend._validate_non_lora(state)

    def test_normalization_matches_released_prefix_and_rejects_duplicates(self):
        tensor = torch.empty(1, device="meta")
        normalized = backend._normalize_non_lora_keys({
            "base_model.model.model.embed_tokens.weight": tensor,
            "base_model.model.model.mm_projector.attn_pool.query": tensor,
        })
        self.assertEqual(set(normalized), {"model.embed_tokens.weight", "model.mm_projector.attn_pool.query"})
        for state in ([], {3: tensor}, {"base_model.model.model.x": tensor, "model.x": tensor}):
            with self.subTest(state_type=type(state).__name__), self.assertRaises(CheckpointError):
                backend._normalize_non_lora_keys(state)

    def test_adapter_has_exact_450_shapes_and_complete_layers(self):
        shapes = backend._adapter_shapes()
        self.assertEqual(len(shapes), 450)
        self.assertEqual(shapes["base_model.model.lesion_head.lora_B.weight"], (4, 128))
        for layer in range(32):
            prefix = f"base_model.model.model.layers.{layer}."
            self.assertEqual(sum(name.startswith(prefix) for name in shapes), 14)
            self.assertEqual(shapes[prefix + "self_attn.k_proj.lora_B.weight"], (1024, 128))
            self.assertEqual(shapes[prefix + "mlp.down_proj.lora_A.weight"], (128, 14336))

    def test_adapter_header_validation_without_weight_allocation(self):
        class Slice:
            def __init__(self, shape, dtype="BF16"):
                self.shape, self.dtype = shape, dtype
            def get_shape(self):
                return self.shape
            def get_dtype(self):
                return self.dtype
        class Header:
            def __init__(self, shapes):
                self.shapes = shapes
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def keys(self):
                return list(self.shapes)
            def get_slice(self, key):
                return Slice(self.shapes[key])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "adapter_config.json"
            config_path.write_text(json.dumps(_adapter_config()))
            shapes = backend._adapter_shapes()
            with patch("safetensors.safe_open", return_value=Header(shapes)):
                backend._validate_adapter(directory)
            for mutation in ("missing", "shape", "configuration"):
                changed = dict(shapes)
                config = _adapter_config()
                if mutation == "missing":
                    changed.pop(next(iter(changed)))
                elif mutation == "shape":
                    changed[next(iter(changed))] = (1,)
                else:
                    config["r"] = 64
                config_path.write_text(json.dumps(config))
                with self.subTest(mutation=mutation), patch("safetensors.safe_open", return_value=Header(changed)):
                    with self.assertRaises(CheckpointError):
                        backend._validate_adapter(directory)


class ReportTokenizerPromptTests(unittest.TestCase):
    def test_tokenizer_uses_training_ids_not_just_distinct_views(self):
        backend._validate_tokenizer(_Tokenizer())
        with patch.object(_Tokenizer, "pad_token_id", 0):
            with self.assertRaises(CheckpointError):
                backend._validate_tokenizer(_Tokenizer())
        with patch.object(_Tokenizer, "convert_tokens_to_ids", return_value=128260):
            with self.assertRaises(CheckpointError):
                backend._validate_tokenizer(_Tokenizer())
        with patch.object(_Tokenizer, "__len__", return_value=128256):
            with self.assertRaises(CheckpointError):
                backend._validate_tokenizer(_Tokenizer())

    def test_training_and_grounding_prompt_whitespace_is_frozen(self):
        # Hashes freeze the original Mammo-GRG prompts, including their whitespace.
        self.assertEqual(hashlib.sha256(backend._preliminary_prompt().encode()).hexdigest(),
                         "834ad58852b979169dc1241d8868aa39ae469da6bb0ee8bf2c930d6e389806fd")
        self.assertEqual(hashlib.sha256(backend.SYSTEM_PROMPT.encode()).hexdigest(),
                         "bb674f3cebfcd73fe0f186ad0124e2c89a391ec0fccc57fe86e1e398fdd33c2e")
        self.assertIn("<LMLO><image> <LCC><image> <RMLO><image> <RCC><image>", backend._preliminary_prompt())
        self.assertIn("<report_generation> Generate a complete mammography report.", backend._preliminary_prompt())

    def test_bilateral_positive_aggregation_and_exact_user_prompt(self):
        findings = _findings()
        findings[0]["mass_pred"] = 1
        findings[2]["mass_pred"] = 1
        findings[1]["suspicious calcification_pred"] = 1
        views, summary, sides = backend.build_view_and_side_summaries(findings)
        self.assertEqual(summary, {"mass": 1, "suspicious_calcification": 1, "asymmetry": 0})
        self.assertEqual(sides["mass"], {"L": 1, "R": 1})
        self.assertEqual(sides["suspicious_calcification"], {"L": 1, "R": 0})
        expected = (
            "Preliminary report (from report generator):\nSynthetic report.\n\n"
            "Structured ML findings (max across RCC/RMLO/LCC/LMLO):\n"
            "- Suspicious Mass: bilaterally.\n- Suspicious Calcification: Left breast.\n\n"
            "Max-summary JSON (do not quote or mention 'classifier' in the final text):\n"
            '{"mass": 1, "suspicious_calcification": 1, "asymmetry": 0}\n\n'
            "Rewrite the report into the required two sections ONLY (Findings, Impression). "
            "Follow the template and structure of the preliminary report."
        )
        self.assertEqual(backend.build_user_prompt(" Synthetic report. ", views, summary, sides), expected)
        views, summary, sides = backend.build_view_and_side_summaries(_findings())
        self.assertEqual(backend.build_classifier_note(summary, sides), "- No positive ML findings across views.")

    def test_invalid_view_count_duplicate_aliases_and_predictions_rejected(self):
        bad = [None, [], _findings()[:3], _findings() + [_findings()[0]], [None] * 4]
        duplicate = _findings()
        duplicate[3] = duplicate[0].copy()
        bad.append(duplicate)
        alias = _findings()
        alias[0]["image_laterality"] = "left"
        bad.append(alias)
        for value in (None, 0.5, "1", 2, -1):
            findings = _findings()
            findings[0]["mass_pred"] = value
            bad.append(findings)
        for index, findings in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError):
                backend.build_view_and_side_summaries(findings)


class ReportGenerationLifecycleTests(unittest.TestCase):
    def test_seed_scope_restores_caller_cpu_rng_even_on_failure(self):
        original = torch.random.get_rng_state().clone()
        with backend._seeded(42, "cpu"):
            first = torch.rand(6)
        self.assertTrue(torch.equal(original, torch.random.get_rng_state()))
        with backend._seeded(42, "cpu"):
            self.assertTrue(torch.equal(first, torch.rand(6)))
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with backend._seeded(42, "cpu"):
                torch.rand(6)
                raise RuntimeError("synthetic")
        self.assertTrue(torch.equal(original, torch.random.get_rng_state()))

    def test_generation_arguments_reject_boolean_negative_and_excessive(self):
        for tokens, seed in ((0, 0), (4097, 0), (True, 0), (1.5, 0), (3, -1), (3, True), (3, 2**63)):
            with self.subTest(tokens=tokens, seed=seed), self.assertRaises(ValueError):
                backend._validate_generation(tokens, seed)

    def test_preliminary_empty_output_fails_and_releases_model(self):
        references = []
        def load(*args):
            model = _Model()
            references.append(weakref.ref(model))
            return _Tokenizer(result="  "), model
        embeddings = torch.tensor(0.).expand(1, 4, 1, 2048, 48, 29)
        with patch.object(backend, "_load_stage2", side_effect=load), \
                patch("mammofm._vendor.grg.tokenization.tokenizer_image_token", return_value=torch.tensor([1, 2])):
            with self.assertRaisesRegex(RuntimeError, "empty text"):
                backend.generate_preliminary(embeddings, "unused", "unused", "cpu", max_new_tokens=3)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_language_models_load_sequentially_and_release_before_return(self):
        references = []
        generation_options = []
        class Model(_Model):
            def generate(self, **kwargs):
                generation_options.append({key: kwargs[key] for key in (
                    "do_sample", "max_new_tokens", "eos_token_id", "pad_token_id")})
                return super().generate(**kwargs)
        def create_model():
            self.assertTrue(all(reference() is None for reference in references))
            model = Model()
            references.append(weakref.ref(model))
            return model
        def load_preliminary(*args):
            return _Tokenizer(), create_model()
        def load_grounded(*args, **kwargs):
            self.assertTrue(kwargs["local_files_only"])
            self.assertTrue(kwargs["use_safetensors"])
            self.assertFalse(kwargs["trust_remote_code"])
            self.assertTrue(kwargs["output_loading_info"])
            return create_model(), {}
        embeddings = torch.tensor(0.).expand(1, 4, 1, 2048, 48, 29)
        with patch.object(backend, "_load_stage2", side_effect=load_preliminary), \
                patch("mammofm._vendor.grg.tokenization.tokenizer_image_token", return_value=torch.tensor([1, 2])), \
                patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: _Tokenizer()), \
                patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=load_grounded):
            preliminary = backend.generate_preliminary(embeddings, "unused", "unused", "cpu", max_new_tokens=3)
            self.assertTrue(all(reference() is None for reference in references))
            result = backend.generate_grounded(preliminary, _findings(), "unused", "cpu", max_new_tokens=4)
        self.assertEqual(result, "Synthetic generated report.")
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual([item["do_sample"] for item in generation_options], [False, True])
        self.assertEqual([item["max_new_tokens"] for item in generation_options], [3, 4])

    def test_grounding_empty_and_incomplete_base_failures_release_models(self):
        for empty_text in (True, False):
            references = []
            def load(*args, **kwargs):
                model = _Model()
                references.append(weakref.ref(model))
                return model, {} if empty_text else {"missing_keys": ["missing.weight"]}
            with self.subTest(empty_text=empty_text), \
                    patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: _Tokenizer(result="")), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=load):
                with self.assertRaisesRegex((RuntimeError, CheckpointError), "empty text|did not load exactly"):
                    backend.generate_grounded("Synthetic preliminary.", _findings(), "unused", "cpu", max_new_tokens=3)
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))

    def test_invalid_embeddings_and_findings_fail_before_model_loading(self):
        with patch.object(backend, "_load_stage2") as loader:
            for embeddings in (None, torch.zeros(1), torch.tensor(float("nan")).expand(1, 4, 1, 2048, 48, 29)):
                with self.subTest(embedding_type=type(embeddings).__name__), self.assertRaises(ValueError):
                    backend.generate_preliminary(embeddings, "unused", "unused", "cpu")
            loader.assert_not_called()
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as loader:
            with self.assertRaises(ValueError):
                backend.generate_grounded("", _findings(), "unused", "cpu")
            with self.assertRaises(ValueError):
                backend.generate_grounded("Synthetic report.", [], "unused", "cpu")
            loader.assert_not_called()


class Stage2LoaderContractTests(unittest.TestCase):
    def test_padding_cleared_for_base_then_restored_and_loads_are_safe(self):
        # Mock all 8B construction/loading; meta tensors allocate no parameter storage.
        from transformers import AutoConfig, AutoTokenizer
        stage_config = types.SimpleNamespace(
            hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
            num_key_value_heads=8, intermediate_size=14336, vocab_size=128264,
            pad_token_id=128256, mm_hidden_size=2048, mm_context_size=2048,
            mm_n_queries=512, mm_projector_type="attn_pool+mlp2x_gelu",
        )
        base_config = types.SimpleNamespace(
            hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
            num_key_value_heads=8, intermediate_size=14336, vocab_size=128256, model_type="llama",
        )
        state = _meta_non_lora()
        expected_added = (set(state) - {"model.embed_tokens.weight"}) | {"lesion_head.weight", "lesion_head.bias"}
        events = []
        test = self
        class Model(_Model):
            def __init__(self, config):
                self.config = config
                self.generation_config = types.SimpleNamespace(pad_token_id=None)
            def resize_token_embeddings(self, count, **kwargs):
                events.append(("resize", count, kwargs))
            def load_state_dict(self, loaded, **kwargs):
                test.assertEqual(set(loaded), set(state))
                test.assertFalse(kwargs["strict"])
                return types.SimpleNamespace(missing_keys=["base.weight"], unexpected_keys=[])
            def state_dict(self):
                return {**state, "base.weight": None}
        def construct(*args, **kwargs):
            test.assertIsNone(kwargs["config"].pad_token_id)
            test.assertEqual(kwargs["config"].vocab_size, 128256)
            for flag in ("local_files_only", "use_safetensors", "output_loading_info"):
                test.assertTrue(kwargs[flag])
            test.assertFalse(kwargs["trust_remote_code"])
            events.append(("construct",))
            return Model(kwargs["config"]), {"missing_keys": list(expected_added)}
        def adapt(model, *args, **kwargs):
            test.assertEqual(model.config.pad_token_id, 128256)
            test.assertEqual(model.config.view_token_ids, [128260, 128261, 128262, 128263])
            test.assertTrue(kwargs["local_files_only"])
            test.assertFalse(kwargs["is_trainable"])
            class Adapted:
                def merge_and_unload(self, **kwargs):
                    test.assertTrue(kwargs["safe_merge"])
                    events.append(("merge",))
                    return model
            return Adapted()
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = types.SimpleNamespace(from_pretrained=adapt)
        fake_modeling = types.ModuleType("mammofm._vendor.grg.modeling_mammo_grg")
        fake_modeling.MammoGRGConfig = types.SimpleNamespace(from_pretrained=lambda *a, **k: stage_config)
        fake_modeling.MammoGRGForCausalLM = types.SimpleNamespace(from_pretrained=construct)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename in ("config.json", "adapter_config.json", "adapter_model.safetensors",
                             "non_lora_trainables.bin", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                (directory / filename).touch()
            with patch.dict(sys.modules, {"peft": fake_peft, fake_modeling.__name__: fake_modeling}), \
                    patch.object(backend, "_validate_adapter"), \
                    patch.object(AutoConfig, "from_pretrained", return_value=base_config), \
                    patch.object(AutoTokenizer, "from_pretrained", return_value=_Tokenizer()), \
                    patch("torch.load", return_value=state) as load:
                tokenizer, model = backend._load_stage2(directory, "unused", "cpu")
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertTrue(load.call_args.kwargs["mmap"])
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
            self.assertEqual(model.config.pad_token_id, tokenizer.pad_token_id)
            self.assertEqual(events, [("construct",), ("resize", 128264, {"mean_resizing": False}), ("merge",)])


class TinyMultimodalGenerationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("einops") is not None,
                         "Install mammofm[report] to run the optional tiny multimodal-model test.")
    def test_tiny_model_cached_and_uncached_generation_without_weights(self):
        from mammofm._vendor.grg.modeling_mammo_grg import MammoGRGConfig, MammoGRGForCausalLM
        config = MammoGRGConfig(
            vocab_size=100, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, bos_token_id=1, eos_token_id=None,
            pad_token_id=0, view_token_ids=[4, 5, 6, 7],
        )
        with backend._seeded(17, "cpu"):
            model = MammoGRGForCausalLM(config).eval()
        model.encode_images = types.MethodType(
            lambda self, images: [torch.ones(1, 2, 16) * index for index in range(4)], model
        )
        tokens = torch.tensor([[1, 4, -200, 5, -200, 6, -200, 7, -200, 8]])
        outputs = [model.generate(inputs=tokens, images=torch.empty(1), attention_mask=torch.ones_like(tokens),
                                  do_sample=False, max_new_tokens=3, use_cache=cache)
                   for cache in (True, False)]
        self.assertEqual(tuple(outputs[0].shape), (1, 3))
        self.assertTrue(torch.equal(outputs[0], outputs[1]))


if __name__ == "__main__":
    unittest.main()
