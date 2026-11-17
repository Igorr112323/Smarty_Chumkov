import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import collect46


class Collect46Tests(unittest.TestCase):
    def test_selection_finishes_before_outer_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            reference = {"gold_train_val_sha256": "sha", "fold_ids_sha256": ["id"] * 5}
            (root / "artifacts/nested38_summary.json").write_text(json.dumps(reference), encoding="utf-8")
            folds = [[{"pair_id": f"pair{n}", "model_output": "aaa", "spans": []}] for n in range(1, 6)]
            for outer in range(1, 6):
                (root / f"weights/ablation46/v1/fold{outer}").mkdir(parents=True)
                for inner in range(1, 6):
                    if outer == inner:
                        continue
                    primary = inner == (outer % 5) + 1
                    record = {"status": "готово", "variant": "v1", "git_sha": "git", "gold_sha256": "sha", "fold_ids_sha256": ["id"] * 5, "train_id_sha256": "train", "outer_id_sha256": "outer", "inner_fold": inner, "primary": primary, "seeds": [{"seed": seed, "epochs_ran": 5, "best_epoch": 4, "checkpoint": {"restore_predictions_equal": True}} for seed in (0, 1)], "inner_grid": [], "elapsed_s": 2}
                    if primary:
                        record["outer_predictions"] = {f"pair{outer}": [[0, 3, 0.5]]}
                    (root / f"artifacts/ablation46_v1_outer{outer}_inner{inner}.json").write_text(json.dumps(record), encoding="utf-8")
            choices = []
            def choose(_):
                choices.append(True)
                return {"start": 0.5, "extend": 0.25}
            actual = {name: 0.0 for name in ("token_precision", "token_recall", "token_f1", "pair_f1", "fpr_pair", "fpr_token")}
            def evaluate(*args):
                self.assertEqual(len(choices), 5)
                return dict(actual)
            extra = {**actual, "exact_f1": 0.0, "overlap_f1": 0.0}
            with patch.object(collect46, "ROOT", root), patch.object(collect46, "read_jsonl", return_value=sum(folds, [])), patch.object(collect46, "grouped_folds", return_value=folds), patch.object(collect46, "file_digest", return_value="sha"), patch.object(collect46, "id_digest", return_value="id"), patch.object(collect46, "select_worst", side_effect=choose), patch.object(collect46, "evaluate", side_effect=evaluate), patch.object(collect46, "verify_fast"), patch.object(collect46, "predict", return_value=[]), patch.object(collect46, "compute", return_value=extra):
                result = collect46.collect("v1")
            self.assertEqual(result["status"], "готово")
            self.assertEqual(result["saved_checkpoints"], 10)
            self.assertEqual(len(result["epochs_ran"]), 40)
            self.assertTrue((root / "weights/ablation46/v1/fold5/manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
