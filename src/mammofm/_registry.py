"""Pinned public inference artifacts and the endpoints they actually support.

Checkpoint names and SHA-256 digests were verified against the public
``batmanLab/Mammo-FM`` repository. An unavailable endpoint must not be filled
with a different classifier or an invented zero-shot prompt.
"""

REPO_ID = "batmanLab/Mammo-FM"
REVISION = "5e07ca4c4e88cd118b0c2cfaacadddfe69f30026"
PIPELINE_REVISION = "7605810fab99c90eb9c9ff23206c237bafa3f500"
GROUNDING_REVISION = "a48f94320712c9c84a662aa3db31dbb018a2f422"
LUMINA_CANCER_REVISION = "2b54c62d95fbfcc48fe05ad87e7729df7035b84a"
PORTRAIT_CLASSIFIER_REVISION = "8388d8b5e645282d3b5a6df30cfac7839de643c6"

ENCODER = {
    "filename": "Mammo-FM_BatmanlabTrained_CLIP.tar",
    "sha256": "58ae959de52d563acc1d43f50571498eddcd4fb7ca599e706540fdd319e3599b",
    "size": 2147265666,
}

# The joint CLIP checkpoint already includes the fine-tuned text weights.
# Only the ModernBERT architecture configuration and tokenizer are needed here;
# downloading the separate 599 MB MLM initialization weights is unnecessary.
TEXT_ASSETS = {
    "config.json": {
        "filename": "text_encoders/modernbert_mlm_5epochs/config.json",
        "sha256": "f04cc05ba3c0185fa6802f47a9b818279cc52e5a98edfcfc1a1b413339881362",
        "size": 1269,
    },
    "tokenizer_config.json": {
        "filename": "text_encoders/modernbert_mlm_5epochs/tokenizer_config.json",
        "sha256": "9654072f7c873161814043cf08cb5ed72f71d0b935abcd4e267935cb34352c21",
        "size": 20867,
    },
    "special_tokens_map.json": {
        "filename": "text_encoders/modernbert_mlm_5epochs/special_tokens_map.json",
        "sha256": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
        "size": 694,
    },
    "tokenizer.json": {
        "filename": "text_encoders/modernbert_mlm_5epochs/tokenizer.json",
        "sha256": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
        "size": 3583228,
    },
}

# Four-view prognosis/reporting assets were published after the original
# single-image package revision. Each entry is independently SHA-256 checked.
REPORT_ENCODER = {
    "filename": "Mammo-FM_BatmanlabTrained_CLIP_last.tar",
    "sha256": "7cf871d116e4cd52a11a5b42d40d4550f078747be0d897265f24cfa5cf26b257",
    "size": 2147265666,
    "revision": PIPELINE_REVISION,
}

BU_RISK_ENCODER = {**ENCODER, "revision": PIPELINE_REVISION}
PROGNOSIS_ENCODER = BU_RISK_ENCODER
DEFAULT_BU_RISK_MODEL = "s1s2"

RISK_CHECKPOINTS = {
    "mayo": {
        "filename": "risk_checkpoints/Mayo-S1_KD_S2/S1S2.pt",
        "sha256": "6f48c0cdd9b5f32ed279d5ade79e14a2e96db291acb166009fc9929c9a29bb8f",
        "size": 184466246,
        "revision": PIPELINE_REVISION,
    },
    "bu": {
        "s1s2": {
            "filename": "risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S1_KD_S2_head.pt",
            "sha256": "6992aa888affb61b8cd897b1ee9e17fae8ef54bf0cb41f38d8935e63675c8273",
            "size": 23381538,
            "revision": PIPELINE_REVISION,
            "protocol": "s1s2",
            "format": "mammofm_risk_head_release_v1",
        },
        "s2_only": {
            "filename": "risk_checkpoints/BU_S1_KD_S2/Mammo-FM_BU_S2_only_head.pt",
            "sha256": "cf3e602acf5b0269ca03c7f851cad5d9b933d4a11d17b84a7b334f431a93fa9c",
            "size": 23381474,
            "revision": PIPELINE_REVISION,
            "protocol": "s2_only",
            "format": "mammofm_risk_head_release_v1",
        },
    },
}

GROUNDING_CLASSIFIERS = {
    "mass": {
        "filename": "Mammo-GRG/grounding/mass_best.pth",
        "sha256": "affce86fcdcbf16979c2d22abe560c397462aa6cbc0643ce0587cd2921393f56",
        "size": 114367866,
        "revision": GROUNDING_REVISION,
    },
    "suspicious_calcification": {
        "filename": "Mammo-GRG/grounding/calcification_best.pth",
        "sha256": "2063e1d84efe310a90ca2f7235510caf761418185154f989e0b0cb1dce1bfd16",
        "size": 114375652,
        "revision": GROUNDING_REVISION,
    },
    "asymmetry": {
        "filename": "Mammo-GRG/grounding/asymmetry_best.pth",
        "sha256": "91392085e1297168139d8f9e762efa10a15c48ac141c9d019d9f55b290916f05",
        "size": 114372156,
        "revision": GROUNDING_REVISION,
    },
}

_REPORT_STAGE2 = (
    ("adapter_config.json", "dfd37ac97f71de76f51f31c1b5d2ed3c5b764d55a4c70e00655d162b66bcb50f"),
    ("adapter_model.safetensors", "9bfe5b38ad997167096e62a4477b7cba4ff5f7e3ac97786ad47b9f615f5a68eb"),
    ("config.json", "38bb94bd29647e9d174f4acdb35f81237a8e4063e7c834980c368c9f7c7a318e"),
    ("non_lora_trainables.bin", "5f791d265c4557314c30a59988b68588964cb3fc9b11cf765d27664c05ceb345"),
    ("special_tokens_map.json", "f4b026086f648d35ead59afce0a4937770a8c03a65ba50de73e3bade23f801be"),
    ("tokenizer.json", "045d2400aeb92b7224f6851838aff4a97a8e6bff11d321c677fc13ffd8044710"),
    ("tokenizer_config.json", "0ddc7105b04f38a9a9f91b7b8099fc119b2f123f4c6c92f6e48fb2b2845cad2c"),
)
REPORT_STAGE2_ASSETS = tuple({
    "filename": f"Mammo-GRG/stage2/checkpoint-5500/{name}",
    "sha256": digest,
    "revision": PIPELINE_REVISION,
} for name, digest in _REPORT_STAGE2)
del _REPORT_STAGE2

TASKS = {
    "rsna": ("cancer", "invasive_cancer", "density"),
    "vindr": ("abnormality", "mass", "suspicious_calcification", "density"),
    "cmmd": ("cancer", "mass", "calcification"),
    "nlbreast": ("cancer",),
    "lumina": ("cancer",),
}

TASK_DEFINITIONS = {
    ("rsna", "cancer"): "RSNA binary cancer label; not an invasive-only classifier.",
    ("rsna", "invasive_cancer"): (
        "Invasive versus non-invasive classification among RSNA cancer-positive "
        "cases with known subtype. This is conditional on cancer, not population "
        "invasive-cancer risk; it is not combined with the separate cancer score."
    ),
    ("rsna", "density"): "Four-class breast density in A, B, C, D order; softmax probabilities.",
    ("vindr", "abnormality"): (
        "BI-RADS 4/5 versus BI-RADS 1/2, with BI-RADS 3 excluded during evaluation. "
        "This is a BI-RADS surrogate, not pathology-confirmed cancer or any finding. "
        "The original compatibility label and zero-shot prompt are called cancer."
    ),
    ("vindr", "mass"): "VinDr Mass finding label.",
    ("vindr", "suspicious_calcification"): "VinDr Suspicious_Calcification finding label.",
    ("vindr", "density"): "Four-class breast density in A, B, C, D order; softmax probabilities.",
    ("cmmd", "cancer"): "CMMD binary cancer label.",
    ("cmmd", "mass"): "CMMD mass finding label.",
    ("cmmd", "calcification"): (
        "CMMD calcification label; zero-shot preserves the original evaluator's "
        "suspicious-calcification prompt mapping."
    ),
    ("nlbreast", "cancer"): "NL-Breast binary cancer label.",
    ("lumina", "cancer"): "LUMINA binary cancer label.",
}

PROMPT_PROVENANCE = {
    "source": "src/codebase/eval_zero_shot_clip.py:build_prompt_sets and src/zs_prompts/zs_prompts.json",
    "abnormality": (
        "Preserves the original VinDr BI-RADS-surrogate evaluation: positive cancer, "
        "negative no cancer. These prompts do not define a general any-finding classifier."
    ),
    "ensemble": "Positive and negative lists concatenate pos_left+pos_right and neg_left+neg_right, respectively.",
    "scoring": (
        "Normalize projected text features individually; average each class; compute "
        "cosine similarity between projected image features and the two class means; "
        "softmax in negative, positive order without temperature or logit-scale multiplication."
    ),
}

# Each tuple contains the linear-probe digest followed by the full-finetune digest.
_CLASSIFIER_HASHES = {
    ("rsna", "cancer"): (
        "8aa57af05a4bbbe8570f79431bfe9ca098fec32f4bcd38cd569c129e9c0ba0b0",
        "5a6b888306ca59042c1842e9527f5c72336819f6dae7166e78d0bde1eecd9b9e",
    ),
    ("vindr", "abnormality"): (
        "7cd854ec6db276026c551431ae5877fbf45ede833c344349effe7a73cb7a48e0",
        "96894b1bf98c1e57f69f4a84c72e5a4db6dae3096432e4e4d61195d380f39ad1",
    ),
    ("vindr", "mass"): (
        "82e507534fa1c5d3f513d5caae9134961ae11f2b3a784df866256033007747a8",
        "21864375bef44c339188bca07157e23c769612ae81aa5854315846c1b6f60d5d",
    ),
    ("vindr", "suspicious_calcification"): (
        "dcab865e5baed0fd7fa7049b35ef71cc8fa04d2f6c8cc14078a94a8e3e46f3ca",
        "a7eb4c9950e692b2d5d1bd772105e462099ba4caa21b9bc581112900dca020dc",
    ),
    ("nlbreast", "cancer"): (
        "75b57af686153ef833d092d1d9abe3343897c1f398b066c5147fb2d1d01e3c1b",
        "bd52a1332c1c527b1ddbf4226e5a78324bbc4c55c321695775d0d27c54c9933f",
    ),
    ("lumina", "cancer"): (
        "416b7eff952bff6101480e1a074188a243a74009cb4d75b13eb5f1f551821fa6",
        "7d8ace59af336932ac3887f5028ae5923a837764128dba70ebb5d29e80ef5e06",
    ),
}

CLASSIFIERS = {}
for (_dataset, _task), _hashes in _CLASSIFIER_HASHES.items():
    _remote_dataset = "nl_breast" if _dataset == "nlbreast" else _dataset
    for _method, _folder, _digest in zip(
        ("linear_probe", "full_finetune"),
        ("linear_probe_mammo_FM_image_encoder", "fullfinetune_mammo_FM_image_encoder"),
        _hashes,
    ):
        CLASSIFIERS[(_dataset, _task, _method)] = {
            "filename": (
                f"{_folder}/{_remote_dataset}/{_task}_classification/checkpoint/"
                f"mammo_fm_{_remote_dataset}_{_task}_{_method}_seed10_fold0_best_validation_auroc.pth"
            ),
            "sha256": _digest,
        }
        if _dataset == "lumina":
            CLASSIFIERS[(_dataset, _task, _method)]["revision"] = LUMINA_CANCER_REVISION

# Inference-only checkpoints from completed 15-epoch portrait training runs.
# Each entry includes the complete image encoder and a one- or four-output head.
_PORTRAIT_CLASSIFIERS = {
    ("rsna", "density", "linear_probe"): (
        "78496b9e1be093b78665d8841c2be89733effdc6f373db7049578b59f0ad4109", 114457042, 4,
    ),
    ("rsna", "density", "full_finetune"): (
        "9c2083f67a6cad82f04460048659f6f2004360946f08f8401ec906a9a7014cb7", 114496492, 4,
    ),
    ("vindr", "density", "linear_probe"): (
        "ccd794b0059dbb840b973f5f0dabd42441ed87c79fabd50baffc918d207d76c5", 114496492, 4,
    ),
    ("vindr", "density", "full_finetune"): (
        "419f0294dcf80bc6a802ea4ba877bcacbe7a0ebcf66ea09560f3a0d5a5b364df", 114501062, 4,
    ),
    ("rsna", "invasive_cancer", "linear_probe"): (
        "e8c0e3ffa0994a9c3baa0c7a8e506b5f8fb0446da65c9e2bd727edc690661f9e", 114482210, 1,
    ),
    ("rsna", "invasive_cancer", "full_finetune"): (
        "b48f0f4c0e8f6768aa29426396baad8c36cf5cb27be768c4b19949a28732c2f0", 114483068, 1,
    ),
    ("cmmd", "cancer", "linear_probe"): (
        "b2d73669ac003a68332bff328d8f321da017f02ee84057370ae3f4a6db47749e", 114431480, 1,
    ),
    ("cmmd", "cancer", "full_finetune"): (
        "2c93326d4037cf7a7e30aafe608a69298fecbb7bf7c080d4118033174ee84575", 114432338, 1,
    ),
    ("cmmd", "mass", "linear_probe"): (
        "a2941e1c95b02dd91f5cb04ba81132aca8e70d51fdb29c70d547665a6c05cb83", 114429700, 1,
    ),
    ("cmmd", "mass", "full_finetune"): (
        "c4a11f1666b3364f1562369b0624f2377e89966c62e717d062d074ce990d9703", 114430622, 1,
    ),
    ("cmmd", "calcification", "linear_probe"): (
        "45ba7ad249e12e3348252acae5325c44abf63b9e37f07ff4e8b2772f47428010", 114480302, 1,
    ),
    ("cmmd", "calcification", "full_finetune"): (
        "7c8990edfd0534183867e067826c014075b0c92d057c4e4969e70e043e0b31ea", 114481160, 1,
    ),
}
for (_dataset, _task, _method), (_digest, _size, _classes) in _PORTRAIT_CLASSIFIERS.items():
    _folder = ("linear_probe_mammo_FM_image_encoder" if _method == "linear_probe"
               else "fullfinetune_mammo_FM_image_encoder")
    CLASSIFIERS[(_dataset, _task, _method)] = {
        "filename": (
            f"{_folder}/{_dataset}/{_task}_classification/portrait_15epochs/checkpoint/"
            f"mammo_fm_{_dataset}_{_task}_{_method}_seed10_fold0_best_validation_auroc.pth"
        ),
        "sha256": _digest,
        "size": _size,
        "num_classes": _classes,
        "revision": PORTRAIT_CLASSIFIER_REVISION,
    }
del _PORTRAIT_CLASSIFIERS, _size, _classes

UNAVAILABLE_REASON = {
    ("rsna", "invasive_cancer", "zero_shot"): (
        "The original evaluation does not define invasive-cancer prompts. "
        "The cancer prompt is not an invasive-cancer substitute."
    ),
}
for _dataset in ("rsna", "vindr"):
    UNAVAILABLE_REASON[(_dataset, "density", "zero_shot")] = (
        "No validated packaged zero-shot density endpoint is released. "
        "Use the released linear-probe or full-finetuning density classifier."
    )

del _dataset, _task, _hashes, _remote_dataset, _method, _folder, _digest
