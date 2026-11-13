"""Pinned assets for four-view report generation and image-level grounding."""

from pathlib import PurePosixPath

from ._registry import (
    GROUNDING_CLASSIFIERS,
    REPORT_ENCODER,
    REPORT_STAGE2_ASSETS,
)


REPORT_STAGE2_FILES = {
    PurePosixPath(asset["filename"]).name: dict(asset)
    for asset in REPORT_STAGE2_ASSETS
}

# Preserve the released classifier's comparison precision.
_THRESHOLDS = {
    "mass": 0.056776825338602066,
    "suspicious_calcification": 0.0676499530673027,
    "asymmetry": 0.049404192715883255,
}
GROUNDING_MODELS = {
    task: {**asset, "threshold": _THRESHOLDS[task]}
    for task, asset in GROUNDING_CLASSIFIERS.items()
}

BASE_LLM_REPO = "meta-llama/Llama-3.1-8B-Instruct"
BASE_LLM_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

__all__ = [
    "REPORT_ENCODER",
    "REPORT_STAGE2_FILES",
    "GROUNDING_MODELS",
    "BASE_LLM_REPO",
    "BASE_LLM_REVISION",
]
