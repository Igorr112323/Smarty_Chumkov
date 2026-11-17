import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import verify_parts46


class VerifyParts46Tests(unittest.TestCase):
    def test_success_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold in range(1, 6):
                folder = root / f"weights/ablation46/v1/fold{fold}"
                entries = []
                for seed in (0, 1):
                    seed_folder = folder / f"seed_{seed}"
                    seed_folder.mkdir(parents=True)
                    data = f"{fold}-{seed}".encode()
                    digest = hashlib.sha256(data).hexdigest()
                    (seed_folder / "part-000.bin").write_bytes(data)
                    entries.append({"seed": seed, "size": len(data), "sha256": digest, "restore_predictions_equal": True, "chunks": [{"name": "part-000.bin", "bytes": len(data), "sha256": digest}]})
                manifest = {"variant": "v1", "outer_fold": fold, "status": "готово", "restore_verified": True, "checkpoints": entries}
                (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(verify_parts46, "ROOT", root):
                self.assertEqual(verify_parts46.check("v1")["checkpoint_count"], 10)
                (root / "weights/ablation46/v1/fold3/seed_1/part-000.bin").write_bytes(b"bad")
                with self.assertRaises(ValueError):
                    verify_parts46.check("v1")


if __name__ == "__main__":
    unittest.main()
