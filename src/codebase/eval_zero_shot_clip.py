import json
import logging
import os
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf

from breastclip import seed_everything
from breastclip.evaluator import Evaluator

log = logging.getLogger(__name__)


def _binary_prompts(positive, negative):
    return {"positive": [positive], "negative": [negative]}


def build_prompt_sets(prompts_json):
    """Build the public-dataset prompt sets used by the official B5 evaluation."""
    with open(prompts_json, "r", encoding="utf-8") as prompt_file:
        loaded_prompts = json.load(prompt_file)
    loaded_prompts.pop("No Finding", None)

    finding_prompts = {}
    for finding_name, prompts in loaded_prompts.items():
        finding_prompts[finding_name.lower()] = {
            "positive": prompts["pos_left"] + prompts["pos_right"],
            "negative": prompts["neg_left"] + prompts["neg_right"],
        }

    finding_prompts["density"] = {
        "A": [
            "the breasts are almost entirely fatty",
            "the breast tissue is almost completely fatty",
            "the breast is composed almost entirely of fat",
            "very little fibroglandular tissue is present",
            "the breasts show minimal dense tissue",
        ],
        "B": [
            "there are scattered areas of fibroglandular density",
            "the breast tissue is mostly fatty with some scattered density",
            "scattered fibroglandular elements are present",
            "some dense tissue scattered throughout the breasts",
            "scattered regions of density seen in the breasts",
        ],
        "C": [
            "the breast tissue is heterogeneously dense",
            "the breasts are composed of heterogeneously dense tissue",
            "there are extensive areas of dense fibroglandular tissue",
            "the density of the breasts is heterogeneous",
            "dense tissue is distributed unevenly throughout the breasts",
        ],
        "D": [
            "the breasts are extremely dense",
            "the breast tissue is almost entirely dense",
            "there is minimal fat in the breasts",
            "dense tissue occupies nearly all of the breast",
            "the breasts show extreme fibroglandular density",
        ],
    }

    cancer = _binary_prompts("cancer", "no cancer")
    return {
        "vindr": finding_prompts,
        "vindr_abnormal": {"cancer": cancer},
        "cmmd": {
            "cancer": cancer,
            "mass": finding_prompts["mass"],
            "suspicious calcification": finding_prompts["suspicious calcification"],
        },
        "nlbreast": {"cancer": cancer},
        "lumina": {"cancer": cancer},
        "rsna": {"cancer": cancer},
    }


@hydra.main(version_base=None, config_path="configs", config_name="zs_clip")
def main(cfg: DictConfig):
    seed_everything(cfg.base.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    OmegaConf.resolve(cfg)

    checkpoint_path = cfg.model.clip_check_point
    save_path = cfg.base.output.save_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "args_dict.pkl"), "wb") as config_file:
        pickle.dump(cfg, config_file)

    log.info("Configurations:\n%s", OmegaConf.to_yaml(cfg))
    prompt_sets = build_prompt_sets(cfg.base.prompts_json)
    evaluator = Evaluator(OmegaConf.to_container(cfg), checkpoint_path)

    for dataset_name in cfg.data_test:
        if dataset_name not in prompt_sets:
            raise KeyError(f"No zero-shot prompts defined for dataset {dataset_name!r}.")
        log.info("Evaluating zero-shot dataset: %s", dataset_name)
        results = evaluator.eval_zeroshot(
            checkpoint_path,
            dataset_name,
            zs_prompts=prompt_sets[dataset_name],
            save_path=save_path,
        )
        with open(os.path.join(save_path, f"results-{dataset_name}.json"), "w", encoding="utf-8") as output_file:
            json.dump({str(checkpoint_path): results}, output_file, indent=2)

    log.info("All zero-shot outputs were written to: %s", save_path)


if __name__ == "__main__":
    main()
