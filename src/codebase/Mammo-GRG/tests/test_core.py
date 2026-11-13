import tempfile
import unittest
from pathlib import Path
import torch

from mammo_grg.checkpoint import validate_stage2_checkpoint
from mammo_grg.io import load_four_view_embeddings
from scripts.train import _resolve_learning_rates


class CoreTests(unittest.TestCase):
    def test_stage_specific_projector_learning_rates(self):
        self.assertEqual(_resolve_learning_rates(1), (1e-3, 1e-3))
        self.assertEqual(_resolve_learning_rates(2), (2e-5, 2e-5))
        self.assertEqual(_resolve_learning_rates(1, learning_rate=5e-4), (5e-4, 5e-4))
        self.assertEqual(_resolve_learning_rates(2, learning_rate=1e-5), (1e-5, 2e-5))
        self.assertEqual(
            _resolve_learning_rates(2, learning_rate=1e-5, projector_lr=3e-5),
            (1e-5, 3e-5),
        )

    def test_neutral_checkpoint_name_is_detected_structurally(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage2" / "checkpoint-5500"
            checkpoint.mkdir(parents=True)
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "non_lora_trainables.bin",
                "config.json",
            ):
                (checkpoint / name).touch()
            self.assertEqual(validate_stage2_checkpoint(checkpoint), checkpoint.resolve())

    def test_four_view_stack_shape_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, name in enumerate(("lmlo", "lcc", "rmlo", "rcc")):
                path = root / f"{name}.pt"
                torch.save(torch.full((1, 2048, 48, 29), index, dtype=torch.float16), path)
                paths.append(path)
            embeddings = load_four_view_embeddings(paths)
            self.assertEqual(embeddings.shape, (1, 4, 1, 2048, 48, 29))
            self.assertEqual(
                [embeddings[0, i, 0, 0, 0, 0].item() for i in range(4)], [0, 1, 2, 3]
            )


if __name__ == "__main__":
    unittest.main()
