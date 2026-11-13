"""Synthetic-only regression tests for the shared portrait classifier geometry."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

CODEBASE = Path(__file__).resolve().parents[2] / "src" / "codebase"
sys.path.insert(0, str(CODEBASE))

from Datasets.dataset_concepts import MammoDataset, resize_classification_image
from Datasets.dataset_utils import get_dataloader, get_transforms

# The legacy package imports a trainer that downloads NLTK data at import time.
# Geometry tests need neither tokenizers nor a network connection.
with patch("nltk.download", return_value=True):
    from breastclip.data.datasets.image_classification_zs import ImageClassificationZSDataset


class ImageGeometryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pixels = np.random.default_rng(4).integers(0, 256, (25, 40, 3), dtype=np.uint8)
        self.image = self.root / "images" / "synthetic" / "image.png"
        self.image.parent.mkdir(parents=True)
        Image.fromarray(self.pixels).save(self.image)
        lumina_image = self.root / "images" / "Malignant" / "synthetic" / "image.png"
        lumina_image.parent.mkdir(parents=True)
        Image.fromarray(self.pixels).save(lumina_image)

    def tearDown(self):
        self.temporary.cleanup()

    def args_and_frame(self, dataset, label="cancer"):
        row = {
            "patient_id": "synthetic", "image_id": "image" if dataset == "rsna" else "image.png",
            "dicom_name": "image.dcm", "image_path": str(self.image),
            "image_full_path": "/LUMINA_RAW/Malign/synthetic/image.dcm",
            "cancer": 1, "Cancer_bin": 1, "mass": 1, "calcification": 0, "density": 3,
        }
        for column in (
            "Mass", "Suspicious_Calcification", "Architectural_Distortion", "Asymmetry",
            "Focal_Asymmetry", "Global_Asymmetry", "Nipple_Retraction", "Skin_Retraction",
            "Skin_Thickening", "Suspicious_Lymph_Node",
        ):
            row[column] = 0
        frame = pd.DataFrame([row])
        args = SimpleNamespace(
            dataset=dataset, model_type="classifier", img_size=[1520, 912],
            data_dir=self.root, img_dir="images", image_encoder_type="tf_efficientnet_b5_ns-detect",
            label=label, mean=0.5, std=0.5, alpha=10.0, sigma=15.0, p=0.0,
            arch="breast_clip_det_b5_period_n_lp", train_folds=frame, valid_folds=frame,
            batch_size=1, num_workers=0, balanced_dataloader="n",
        )
        return args, frame

    def test_resize_uses_height_width_and_preserves_correct_images(self):
        portrait = resize_classification_image(self.pixels)
        self.assertEqual(portrait.shape, (1520, 912, 3))
        np.testing.assert_array_equal(
            portrait, cv2.resize(self.pixels, (912, 1520), interpolation=cv2.INTER_LINEAR)
        )
        self.assertIs(resize_classification_image(portrait), portrait)

    def test_all_classification_splits_have_the_same_portrait_geometry(self):
        for dataset in ("rsna", "vindr", "cmmd", "nlbreast", "lumina"):
            with self.subTest(dataset=dataset):
                args, frame = self.args_and_frame(dataset)
                train_loader, valid_loader = get_dataloader(args)
                train_sample = train_loader.dataset[0]
                valid_sample = valid_loader.dataset[0]
                test_sample = MammoDataset(args, frame)[0]
                for sample in (train_sample, valid_sample, test_sample):
                    self.assertEqual(tuple(sample["x"].shape), (1, 1520, 912, 3))
                    self.assertEqual(sample["y"].item(), 1)
                    self.assertTrue(torch.isfinite(sample["x"]).all().item())
                # Mandatory resize still happens when all random augmentation is disabled.
                torch.testing.assert_close(train_sample["x"], valid_sample["x"])
                torch.testing.assert_close(valid_sample["x"], test_sample["x"])

    def test_density_class_index_is_preserved(self):
        args, frame = self.args_and_frame("vindr", label="density")
        sample = MammoDataset(args, frame)[0]
        self.assertEqual(sample["y"].item(), 3)
        self.assertEqual(tuple(sample["x"].shape), (1, 1520, 912, 3))

    def test_custom_augmentation_cannot_reverse_the_final_geometry(self):
        args, frame = self.args_and_frame("cmmd")
        transform = lambda image: {"image": cv2.resize(image, (1520, 912))}
        sample = MammoDataset(args, frame, transform=transform)[0]
        self.assertEqual(tuple(sample["x"].shape), (1, 1520, 912, 3))

    def test_reversed_or_nonstandard_requested_size_is_rejected(self):
        args, frame = self.args_and_frame("cmmd")
        for image_size in ([912, 1520], [512, 512], [1520]):
            with self.subTest(image_size=image_size):
                args.img_size = image_size
                with self.assertRaisesRegex(ValueError, "height, width"):
                    MammoDataset(args, frame)
                with self.assertRaisesRegex(ValueError, "height, width"):
                    get_transforms(args)

    def test_zero_shot_all_datasets_use_portrait_geometry(self):
        for dataset in ("rsna", "vindr", "cmmd", "nlbreast", "lumina"):
            with self.subTest(dataset=dataset):
                _, frame = self.args_and_frame(dataset)
                loader = ImageClassificationZSDataset(
                    tokenizer=None, split="valid", df=frame, dataset=dataset,
                    data_dir=str(self.root), image_dir="images", mean=0.5, std=0.5,
                    image_encoder_type="tf_efficientnet_b5_ns-detect",
                    transform_config={"valid": {"Resize": {"size_h": 1520, "size_w": 912}}},
                )
                with patch.object(loader, "_get_img_path", return_value=self.image):
                    sample = loader[0]
                self.assertEqual(tuple(sample["images"].shape), (1, 1520, 912, 3))
                self.assertTrue(torch.isfinite(sample["images"]).all().item())


if __name__ == "__main__":
    unittest.main()
