"""Resolve pinned public model assets; never transmit images or image paths."""

import hashlib
import os
from pathlib import Path, PurePosixPath

from ._errors import CheckpointError
from ._registry import REPO_ID, REVISION


class AssetStore:
    def __init__(self, cache_dir=None, model_dir=None, local_files_only=False):
        self.cache_dir = cache_dir or os.environ.get("MAMMOFM_CACHE_DIR")
        local = model_dir or os.environ.get("MAMMOFM_MODEL_DIR")
        self.model_dir = Path(local).expanduser().resolve() if local else None
        self.local_files_only = bool(local_files_only)
        self._verified = {}

    def resolve(self, asset):
        filename, expected = asset["filename"], asset["sha256"]
        revision = asset.get("revision", REVISION)
        relative = PurePosixPath(filename)
        if relative.is_absolute() or ".." in relative.parts or "\\" in filename:
            raise CheckpointError("Unsafe model asset filename")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise CheckpointError("Model asset must have a pinned SHA-256")
        path = self.model_dir / filename if self.model_dir else None
        if path is None or not path.is_file():
            from huggingface_hub import hf_hub_download
            try:
                path = Path(hf_hub_download(
                    repo_id=REPO_ID, filename=filename, revision=revision,
                    cache_dir=self.cache_dir, local_files_only=self.local_files_only,
                    token=False,
                ))
            except Exception as error:
                raise CheckpointError(
                    "Could not obtain {} at revision {}. Check connectivity or supply "
                    "a model_dir containing the published HF directory structure.".format(filename, revision)
                ) from error
        self._verify(path, expected)
        return path

    def _verify(self, path, expected):
        stat = path.stat()
        stamp = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)
        if stamp not in self._verified:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise CheckpointError("SHA-256 mismatch for {}; refusing to load weights".format(path.name))
            self._verified[stamp] = True


def provenance(asset):
    return {"repo_id": REPO_ID, "revision": asset.get("revision", REVISION),
            "filename": asset["filename"], "sha256": asset["sha256"]}
