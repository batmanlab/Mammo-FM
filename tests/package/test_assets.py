"""Synthetic asset-resolution tests; no credentials, network, or model weights."""

import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from mammofm._assets import AssetStore, provenance
from mammofm._errors import CheckpointError
from mammofm._registry import REPO_ID, REVISION


class AssetStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.payload = b"synthetic model fixture, not actual weights"
        self.asset = {
            "filename": "models/example.pth",
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        self.local = self.root / self.asset["filename"]
        self.local.parent.mkdir()
        self.local.write_bytes(self.payload)
        self.download = Mock(side_effect=AssertionError("Network must never be used"))
        self.hub = types.ModuleType("huggingface_hub")
        self.hub.hf_hub_download = self.download
        self.module_patch = patch.dict("sys.modules", {"huggingface_hub": self.hub})
        self.module_patch.start()
        self.env_patch = patch.dict(os.environ, {}, clear=False)
        self.env_patch.start()
        os.environ.pop("MAMMOFM_MODEL_DIR", None)
        os.environ.pop("MAMMOFM_CACHE_DIR", None)

    def tearDown(self):
        self.env_patch.stop()
        self.module_patch.stop()
        self.temporary.cleanup()

    def test_local_asset_never_downloads(self):
        self.assertEqual(AssetStore(model_dir=self.root).resolve(self.asset), self.local)
        self.download.assert_not_called()

    def test_unsafe_asset_names_rejected_before_download(self):
        for filename in ("/etc/passwd", "../outside.pth", "models/../../outside.pth", "models\\outside.pth"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(CheckpointError, "Unsafe"):
                    AssetStore().resolve({**self.asset, "filename": filename})
        self.download.assert_not_called()

    def test_unpinned_or_malformed_hash_rejected(self):
        for digest in ("", "a" * 63, "a" * 65, "A" * 64, "z" * 64):
            with self.subTest(digest=digest):
                with self.assertRaisesRegex(CheckpointError, "SHA-256"):
                    AssetStore(model_dir=self.root).resolve({**self.asset, "sha256": digest})
        self.download.assert_not_called()

    def test_local_hash_mismatch_fails_without_network_fallback(self):
        self.local.write_bytes(b"corrupt")
        with self.assertRaisesRegex(CheckpointError, "SHA-256 mismatch"):
            AssetStore(model_dir=self.root).resolve(self.asset)
        self.download.assert_not_called()

    def test_offline_cache_lookup_passes_exact_pinned_arguments(self):
        self.download.side_effect = None
        self.download.return_value = str(self.local)
        store = AssetStore(cache_dir="synthetic-cache", local_files_only=True)
        self.assertEqual(store.resolve(self.asset), self.local)
        self.download.assert_called_once_with(
            repo_id=REPO_ID, filename=self.asset["filename"], revision=REVISION,
            cache_dir="synthetic-cache", local_files_only=True, token=False,
        )

    def test_asset_specific_revision_is_used_and_reported(self):
        self.download.side_effect = None
        self.download.return_value = str(self.local)
        asset = {**self.asset, "revision": "a" * 40}
        AssetStore(local_files_only=True).resolve(asset)
        self.assertEqual(self.download.call_args.kwargs["revision"], "a" * 40)
        self.assertEqual(provenance(asset)["revision"], "a" * 40)

    def test_online_lookup_never_uses_implicit_hf_token(self):
        self.download.side_effect = None
        self.download.return_value = str(self.local)
        AssetStore().resolve(self.asset)
        self.assertFalse(self.download.call_args.kwargs["token"])
        self.assertFalse(self.download.call_args.kwargs["local_files_only"])
        self.assertEqual(self.download.call_args.kwargs["revision"], REVISION)

    def test_missing_model_directory_file_can_use_offline_hf_cache(self):
        self.download.side_effect = None
        self.download.return_value = str(self.local)
        store = AssetStore(model_dir=self.root / "empty", local_files_only=True)
        self.assertEqual(store.resolve(self.asset), self.local)
        self.assertTrue(self.download.call_args.kwargs["local_files_only"])

    def test_download_errors_are_actionable_checkpoint_errors(self):
        error = RuntimeError("synthetic offline cache miss")
        self.download.side_effect = error
        with self.assertRaisesRegex(CheckpointError, "model_dir") as caught:
            AssetStore(local_files_only=True).resolve(self.asset)
        self.assertIs(caught.exception.__cause__, error)
        self.assertIn(REVISION, str(caught.exception))

    def test_downloaded_file_checksum_is_not_trusted(self):
        self.local.write_bytes(b"corrupt downloaded content")
        self.download.side_effect = None
        self.download.return_value = str(self.local)
        with self.assertRaisesRegex(CheckpointError, "SHA-256 mismatch"):
            AssetStore().resolve(self.asset)

    def test_environment_paths_and_explicit_override(self):
        os.environ["MAMMOFM_MODEL_DIR"] = str(self.root)
        os.environ["MAMMOFM_CACHE_DIR"] = "environment-cache"
        store = AssetStore()
        self.assertEqual(store.resolve(self.asset), self.local)
        self.assertEqual(store.cache_dir, "environment-cache")
        override = AssetStore(model_dir=self.root / "override", cache_dir="explicit-cache")
        self.assertEqual(override.model_dir, self.root / "override")
        self.assertEqual(override.cache_dir, "explicit-cache")

    def test_unchanged_asset_is_not_rehashed(self):
        store = AssetStore(model_dir=self.root)
        store.resolve(self.asset)
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected second hash read")):
            self.assertEqual(store.resolve(self.asset), self.local)

    def test_changed_file_invalidates_successful_verification(self):
        store = AssetStore(model_dir=self.root)
        store.resolve(self.asset)
        old = self.local.stat()
        self.local.write_bytes(b"X" * len(self.payload))
        os.utime(self.local, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000000))
        with self.assertRaisesRegex(CheckpointError, "SHA-256 mismatch"):
            store.resolve(self.asset)

    def test_expected_digest_is_part_of_verification_cache_key(self):
        store = AssetStore(model_dir=self.root)
        store.resolve(self.asset)
        with self.assertRaisesRegex(CheckpointError, "SHA-256 mismatch"):
            store.resolve({**self.asset, "sha256": "0" * 64})

    def test_provenance_contains_only_public_model_identity(self):
        result = provenance(self.asset)
        self.assertEqual(result, {
            "repo_id": REPO_ID, "revision": REVISION,
            "filename": self.asset["filename"], "sha256": self.asset["sha256"],
        })
        self.assertNotIn(str(self.root), str(result))


if __name__ == "__main__":
    unittest.main()
