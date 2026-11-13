"""Public errors for predictable inference failures."""


class CheckpointError(RuntimeError):
    """A checkpoint is unavailable, corrupt, or incompatible."""


class MissingModelError(ValueError):
    """A requested task has no released model or canonical zero-shot prompts."""
