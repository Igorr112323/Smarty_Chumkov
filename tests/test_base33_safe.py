import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import download_rubert_base as downloader


class DownloaderTests(unittest.TestCase):
    def test_manifest_uses_actual_hash_and_copies_auxiliary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.mkdir()
            for name in downloader.FILES:
                (source / name).write_text(name, encoding="utf-8")
            cached = root / "cached"
            cached.write_bytes(b"data")
            def fake_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], "DeepPavlov/rubert-base-cased")
                self.assertEqual(kwargs["filename"], "pytorch_model.bin")
                return str(cached)
            with patch.object(downloader, "EXPECTED_SIZE", 4):
                result = downloader.download(fake_download, target, source)
            self.assertEqual(result["size"], 4)
            self.assertEqual(result["sha256"], downloader.sha256(cached))
            self.assertFalse(result["sha256_verified_against_official"])
            self.assertEqual((target / "pytorch_model.bin").read_bytes(), b"data")
            self.assertEqual(json.loads((target / "manifest.json").read_text())["sha256"], result["sha256"])

    def test_size_mismatch_rejected(self):
        with patch.object(downloader, "EXPECTED_SIZE", 1000):
            downloader.validate_size(1000)
            with self.assertRaises(ValueError):
                downloader.validate_size(500)


if __name__ == "__main__":
    unittest.main()
