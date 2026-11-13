import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from breastclip.data.data_utils import load_transform

log = logging.getLogger(__name__)


class ImageClassificationZSDataset(Dataset):
    """Public mammography datasets used by the Mammo-FM zero-shot evaluation."""

    def __init__(
            self,
            tokenizer,
            split: str,
            df,
            dataset: str,
            data_dir: str,
            image_dir: str,
            text_max_length: int = 256,
            transform_config: Dict = None,
            mean=0,
            std=1,
            image_encoder_type="swin",
            **kwargs
    ):
        log.info("Loading zero-shot image classification dataset: [%s/%s]", dataset, split)
        self.df = df
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.root_dir = Path(data_dir)
        self.img_dir = Path(image_dir)
        self.dataset = dataset
        self.transform = load_transform(split=split, transform_config=transform_config)

        self.mean = mean
        self.std = std
        self.image_encoder_type = image_encoder_type

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _relative_after_marker(path_value, marker):
        parts = Path(str(path_value)).parts
        if marker not in parts:
            raise ValueError(f"Expected path marker {marker!r} in {path_value!r}")
        return Path(*parts[parts.index(marker) + 1:])

    def _get_img_path(self, data):
        dataset = self.dataset.lower()
        if dataset == "cmmd":
            rel_path = Path(str(data["patient_id"])) / str(data["dicom_name"])
        elif dataset == "nlbreast":
            rel_path = self._relative_after_marker(data["image_path"], "NLBS_Data")
        elif dataset == "lumina":
            rel_path = self._relative_after_marker(data["image_full_path"], "LUMINA_RAW")
            if rel_path.parts and rel_path.parts[0] == "Malign":
                rel_path = Path("Malignant", *rel_path.parts[1:])
        else:
            rel_path = Path(str(data["patient_id"])) / str(data["image_id"])

        img_path = self.root_dir / self.img_dir / rel_path
        return img_path.with_suffix(".png")

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_path = self._get_img_path(data)
        if not img_path.is_file():
            raise FileNotFoundError(f"Mammogram not found: {img_path}")

        if self.image_encoder_type in {
            "swin", "resnet101", "resnet152", "tf_efficientnet_b5_ns-detect",
            "tf_efficientnetv2-detect"
        }:
            img = np.asarray(Image.open(img_path).convert("RGB"))
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not decode mammogram: {img_path}")

        if self.transform:
            img = self.transform(image=img)["image"]

        # All datasets share the classifier encoder's portrait input geometry.
        # OpenCV accepts (width, height), unlike the tensor's (height, width).
        if img.shape[:2] != (1520, 912):
            img = cv2.resize(img, (912, 1520), interpolation=cv2.INTER_LINEAR)

        img = img.astype("float32")
        img -= img.min()
        max_value = img.max()
        if max_value > 0:
            img /= max_value
        img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32).unsqueeze(0)

        patient_id = str(data.get("patient_id", data.get("group_id", "")))
        output = {
            "images": img,
            "img_path": str(img_path),
            "patient_id": patient_id,
        }
        dataset = self.dataset.lower()

        if dataset == "vindr":
            columns = {
                "mass": "Mass",
                "calc": "Suspicious_Calcification",
                "architectural_distortion": "Architectural_Distortion",
                "asymmetry": "Asymmetry",
                "focal_asymmetry": "Focal_Asymmetry",
                "global_asymmetry": "Global_Asymmetry",
                "nipple_retraction": "Nipple_Retraction",
                "skin_retraction": "Skin_Retraction",
                "skin_thickening": "Skin_Thickening",
                "suspicious_lymph_node": "Suspicious_Lymph_Node",
                "density": "density",
            }
            output.update({key: torch.tensor(data[column], dtype=torch.long)
                           for key, column in columns.items()})
        elif dataset == "vindr_abnormal":
            cancer = torch.tensor(data["abnormal"], dtype=torch.long)
            output["cancer"] = cancer
        elif dataset == "cmmd":
            cancer = torch.tensor(data["cancer"], dtype=torch.long)
            output.update({
                "cancer": cancer,
                "mass": torch.tensor(data["mass"], dtype=torch.long),
                "calc": torch.tensor(data["calcification"], dtype=torch.long),
            })
        elif dataset == "nlbreast":
            cancer = torch.tensor(data["cancer"], dtype=torch.long)
            output["cancer"] = cancer
        elif dataset == "lumina":
            output["cancer"] = torch.tensor(data["Cancer_bin"], dtype=torch.long)
        elif dataset == "rsna":
            cancer = torch.tensor(data["cancer"], dtype=torch.long)
            output["cancer"] = cancer
        else:
            raise KeyError(f"Unsupported zero-shot dataset: {self.dataset}")

        return output
