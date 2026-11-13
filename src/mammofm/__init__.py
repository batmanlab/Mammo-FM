"""Local, research-only Mammo-FM inference. Importing does not download models."""

from .api import (
    MammoFM, availability, clear_cache, diagnosis, features, image_embeddings,
    preprocess, prognosis, report_gen, text_embeddings, zero_shot,
)
from ._errors import CheckpointError, MissingModelError


def run_exam(*args, **kwargs):
    """Lazily run the four-view report and direct-risk-probability pipeline."""
    from .exam import run_exam as _run_exam
    return _run_exam(*args, **kwargs)


__version__ = "0.4.1"
__all__ = ["MammoFM", "preprocess", "diagnosis", "features", "prognosis", "report_gen", "run_exam", "availability", "clear_cache",
           "image_embeddings", "text_embeddings", "zero_shot", "CheckpointError", "MissingModelError"]
