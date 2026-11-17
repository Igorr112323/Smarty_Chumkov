import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class CollectFinal45Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "tools").mkdir()
        (self.root / "artifacts").mkdir()
        (self.root / "tools/collect_final45.py").write_bytes((ROOT / "tools/collect_final45.py").read_bytes())
        self.write("nested38_summary.json", {"summary": {"token_f1": {"mean": 0.4506}}})

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        (self.root / "artifacts" / name).write_text(json.dumps(value), encoding="utf-8")

    def collect(self):
        subprocess.run([sys.executable, str(self.root / "tools/collect_final45.py")], check=True, capture_output=True, text=True)
        return json.loads((self.root / "artifacts/final45_reproduction_summary.json").read_text(encoding="utf-8"))

    def test_partial_cannot_claim_reproduction(self):
        self.write("reproduce45_fold1.json", {"status": "в процессе", "seeds": [{"seed": 0}]})
        result = self.collect()
        self.assertEqual(result["folds_complete"], 0)
        self.assertFalse(result["all_metrics_match_original"])
        self.assertNotIn("macro_token_f1", result)

    def test_five_complete_folds_require_each_match(self):
        for number in range(1, 6):
            self.write(f"reproduce45_fold{number}.json", {"status": "готово", "seeds": list(range(5)), "metrics_equal": number != 3, "metrics": {"token_f1": number / 10, "fpr_token": number / 100, "fpr_pair": number / 200}})
        self.write("final45_restore_verification.json", {"matches_original": True})
        result = self.collect()
        self.assertEqual(result["folds_complete"], 5)
        self.assertEqual(result["macro_token_f1"], 0.3)
        self.assertFalse(result["all_metrics_match_original"])
        self.assertTrue(result["restore_verified"])
        self.assertEqual(result["max_fpr_token"], 0.05)


if __name__ == "__main__":
    unittest.main()
