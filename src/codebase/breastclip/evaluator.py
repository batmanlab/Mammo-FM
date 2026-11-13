import copy
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn import metrics
from tqdm import tqdm

from .data import DataModule
from .model import build_model

log = logging.getLogger(__name__)


class Evaluator:
    """Evaluation helper for Mammo-FM zero-shot classification and retrieval."""

    _LABEL_KEYS = (
        "mass",
        "calc",
        "architectural_distortion",
        "asymmetry",
        "focal_asymmetry",
        "global_asymmetry",
        "nipple_retraction",
        "skin_retraction",
        "skin_thickening",
        "suspicious_lymph_node",
        "density",
        "cancer",
    )
    _PROMPT_TO_LABEL = {
        "suspicious calcification": "calc",
        "architectural distortion": "architectural_distortion",
        "focal asymmetry": "focal_asymmetry",
        "global asymmetry": "global_asymmetry",
        "nipple retraction": "nipple_retraction",
        "skin retraction": "skin_retraction",
        "skin thickening": "skin_thickening",
        "suspicious lymph node": "suspicious_lymph_node",
        "cancer": "cancer",
    }

    def __init__(self, config: Dict, ckpt_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.ckpt_config = copy.deepcopy(checkpoint["config"])

        modern_bert_cache = self.config["base"].get("modern_bert_cache")
        tokenizer_config = copy.deepcopy(self.ckpt_config.get("tokenizer"))
        if modern_bert_cache:
            tokenizer_config["cache_dir"] = modern_bert_cache
            self.ckpt_config["model"]["text_encoder"]["cache_dir"] = modern_bert_cache

        image_encoder = self.ckpt_config["model"]["image_encoder"]
        self.clip_image_encoder = image_encoder.get("model_type")
        if image_encoder.get("name") in {
            "resnet101", "resnet152", "tf_efficientnet_b5_ns-detect", "tf_efficientnetv2-detect"
        }:
            self.clip_image_encoder = image_encoder["name"]

        self.datamodule = DataModule(
            data_config=self.config["data_test"],
            dataloader_config=self.config["dataloader"],
            tokenizer_config=tokenizer_config,
            transform_config=self.config.get("transform", self.ckpt_config.get("transform")),
            mean=self.config["base"]["mean"],
            std=self.config["base"]["std"],
            image_encoder_type=self.clip_image_encoder,
            cur_fold=self.config["base"]["fold"],
        )
        self.test_dataloader_dict = self.datamodule.valid_dataloader()
        if not self.test_dataloader_dict:
            raise ValueError("No zero-shot evaluation datasets were configured.")

        self.model = build_model(
            model_config=self.ckpt_config["model"],
            loss_config=self.ckpt_config["loss"],
            tokenizer=self.datamodule.tokenizer,
        ).to(self.device)
        incompatible = self.model.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            log.warning("Checkpoint load: missing=%s unexpected=%s",
                        incompatible.missing_keys, incompatible.unexpected_keys)
        self.model.eval()
        del checkpoint

    def get_embeddings(self, test_dataset_name):
        dataloader = self.test_dataloader_dict[test_dataset_name]
        image_embeddings = []
        text_embeddings = []
        collected = {key: [] for key in self._LABEL_KEYS}
        collected.update({"texts": [], "label_names": [], "labels": [],
                          "patient_id": [], "img_path": []})

        for batch in tqdm(dataloader, desc=f"Embedding {test_dataset_name}"):
            images = batch["images"]
            if images.ndim == 5 and images.shape[1] == 1:
                images = images.squeeze(1)
            if images.ndim == 4 and images.shape[-1] in {1, 3}:
                images = images.permute(0, 3, 1, 2)
            image_embeddings.append(self.encode_image(images))

            if "texts" in batch:
                collected["texts"].extend(batch["texts"])
            if "text_tokens" in batch:
                text_embeddings.append(self.encode_text(batch["text_tokens"]))
            for key in ("label_names", "patient_id", "img_path"):
                if key in batch:
                    collected[key].extend(batch[key])
            if "labels" in batch:
                collected["labels"].extend(batch["labels"].cpu().numpy().tolist())
            for key in self._LABEL_KEYS:
                if key in batch:
                    collected[key].extend(batch[key].cpu().numpy().tolist())

        if not image_embeddings:
            raise ValueError(f"Dataset {test_dataset_name!r} has no evaluation images.")
        collected["image_embeddings"] = np.concatenate(image_embeddings, axis=0)
        collected["text_embeddings"] = (
            np.concatenate(text_embeddings, axis=0) if text_embeddings else np.empty((0, 0))
        )
        return collected

    def encode_image(self, image: torch.Tensor):
        with torch.no_grad():
            image_embedding = self.model.encode_image(image.to(self.device))
            if self.model.projection:
                image_embedding = self.model.image_projection(image_embedding)
            image_embedding = torch.nn.functional.normalize(image_embedding, dim=1)
        return image_embedding.detach().cpu().numpy()

    def encode_text(self, text_token: Union[str, List[str], Dict, torch.Tensor]):
        if isinstance(text_token, (str, list)):
            text_token = self.datamodule.tokenizer(
                text_token,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=self.ckpt_config["base"]["text_max_length"],
            )
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_token.to(self.device))
            if self.model.projection:
                text_embedding = self.model.text_projection(text_embedding)
            text_embedding = torch.nn.functional.normalize(text_embedding, dim=1)
        return text_embedding.detach().cpu().numpy()

    def _mean_prompt_embedding(self, prompts):
        if not prompts:
            raise ValueError("A zero-shot prompt class cannot be empty.")
        return self.encode_text(list(prompts)).mean(axis=0, keepdims=True)

    @staticmethod
    def _binary_metrics(y_true, scores):
        y_true = np.asarray(y_true, dtype=int)
        scores = np.asarray(scores, dtype=float)
        predictions = (scores >= 0.5).astype(int)
        result = {
            "n": int(len(y_true)),
            "positives": int(y_true.sum()),
            "accuracy": float(metrics.accuracy_score(y_true, predictions)),
            "f1": float(metrics.f1_score(y_true, predictions, zero_division=0)),
            "auroc": None,
            "auprc": None,
        }
        if np.unique(y_true).size == 2:
            result["auroc"] = float(metrics.roc_auc_score(y_true, scores))
            result["auprc"] = float(metrics.average_precision_score(y_true, scores))
        return result

    def eval_zeroshot(self, checkpoint, test_dataset_name, zs_prompts, save_path):
        # ``checkpoint`` remains in the signature for compatibility with the
        # original launcher; weights are loaded once during initialization.
        del checkpoint
        embeddings = self.get_embeddings(test_dataset_name)
        image_embeddings = embeddings["image_embeddings"]
        sample_count = image_embeddings.shape[0]
        rows = [
            {
                "image_path": embeddings["img_path"][i] if i < len(embeddings["img_path"]) else None,
                "patient_id": embeddings["patient_id"][i] if i < len(embeddings["patient_id"]) else None,
            }
            for i in range(sample_count)
        ]
        results = {}

        for prompt_name, prompt_entry in zs_prompts.items():
            prompt_key = prompt_name.lower()
            if prompt_key == "density":
                class_embeddings = np.vstack([
                    self._mean_prompt_embedding(prompt_entry[density_class])
                    for density_class in ("A", "B", "C", "D")
                ])
                probabilities = softmax(
                    metrics.pairwise.cosine_similarity(image_embeddings, class_embeddings), axis=1
                )
                predictions = probabilities.argmax(axis=1)
                y_true = np.asarray(embeddings["density"], dtype=int)
                if len(y_true) != sample_count:
                    raise ValueError(f"Missing density labels for {test_dataset_name}.")
                results[prompt_name] = {
                    "n": int(sample_count),
                    "accuracy": float(metrics.accuracy_score(y_true, predictions)),
                }
                for i in range(sample_count):
                    rows[i]["density"] = int(y_true[i])
                    rows[i]["density_pred"] = int(predictions[i])
                    for class_index, density_class in enumerate(("A", "B", "C", "D")):
                        rows[i][f"density_{density_class}_score"] = float(probabilities[i, class_index])
                continue

            positive = self._mean_prompt_embedding(prompt_entry.get("positive", []))
            negative = self._mean_prompt_embedding(prompt_entry.get("negative", []))
            class_embeddings = np.vstack([negative, positive])
            scores = softmax(
                metrics.pairwise.cosine_similarity(image_embeddings, class_embeddings), axis=1
            )[:, 1]
            label_key = self._PROMPT_TO_LABEL.get(prompt_key, prompt_key.replace(" ", "_"))
            y_true = np.asarray(embeddings.get(label_key, []), dtype=int)
            if len(y_true) != sample_count:
                raise ValueError(
                    f"Missing labels for prompt {prompt_name!r} in dataset {test_dataset_name!r}."
                )
            results[prompt_name] = self._binary_metrics(y_true, scores)
            for i in range(sample_count):
                rows[i][prompt_key] = int(y_true[i])
                rows[i][f"{prompt_key}_score"] = float(scores[i])
                rows[i][f"{prompt_key}_pred"] = int(scores[i] >= 0.5)

        output_path = Path(save_path)
        output_path.mkdir(parents=True, exist_ok=True)
        predictions_path = output_path / f"df_zs_{test_dataset_name}.csv"
        pd.DataFrame(rows).to_csv(predictions_path, index=False)
        log.info("Wrote zero-shot predictions to %s", predictions_path)
        return results

    def eval_img_text_retrieval(self, checkpoint, test_dataset_name, save_path):
        del checkpoint, save_path
        embeddings = self.get_embeddings(test_dataset_name)
        image_embeddings = embeddings["image_embeddings"]
        text_embeddings = embeddings["text_embeddings"]
        text_list = embeddings["texts"]

        unique_texts = []
        unique_indexes = []
        index_to_label = {}
        for index, text in enumerate(text_list):
            if text not in unique_texts:
                unique_texts.append(text)
                unique_indexes.append(index)
            index_to_label[index] = unique_texts.index(text)

        unique_text_embeddings = text_embeddings[unique_indexes]
        similarities = metrics.pairwise.cosine_similarity(image_embeddings, unique_text_embeddings)
        recall = {1: 0, 5: 0, 10: 0, 15: 0}
        mean_rank = 0
        for index, similarity in enumerate(similarities):
            label = index_to_label[index]
            rank = len(unique_texts) - np.argwhere(similarity.argsort() == label).ravel()[0]
            mean_rank += rank
            for k in recall:
                if rank <= k:
                    recall[k] += 1

        sample_count = len(image_embeddings)
        result = {f"Recall@{k}": value / sample_count for k, value in recall.items()}
        result["MeanRank"] = mean_rank / sample_count
        return {"retrieval_i2t": result}
