import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_inner43 import inspect, search_inner_prediction_keys


class InnerAuditTests(unittest.TestCase):
    def test_inner_prediction_key_detection(self):
        data = {"inner_grid": [{"token": {"precision": 0.5}}], "inner_cache": {"pieces": []}, "outer_predictions": {"a": [[1, 2, 0.5]]}}
        self.assertEqual(search_inner_prediction_keys(data), ["inner_cache", "inner_cache.pieces"])

    def test_inspect_preserves_aggregate_and_provenance(self):
        data = {"status": "готово", "outer_fold": 1, "inner_fold": 2, "inner_grid": [{"start": 0.5, "fpr_token": 0.3}], "seeds": [{"seed": 0}], "outer_predictions": {"a": [[0, 1, 0.3]]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fold.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = inspect(path, 1, 2)
        self.assertEqual(result["inner_grid_columns"], ["fpr_token", "start"])
        self.assertEqual(result["candidate_inner_prediction_keys"], [])
        self.assertEqual(result["outer_prediction_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
