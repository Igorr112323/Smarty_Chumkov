import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import collect47


class Collect47Tests(unittest.TestCase):
    def test_selection_finishes_before_outer_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            reference = {"gold_train_val_sha256": "sha", "fold_ids_sha256": ["id"] * 5}
            (root / "artifacts/nested38_summary.json").write_text(json.dumps(reference), encoding="utf-8")
            folds = [[{"pair_id": f"pair{n}", "model_output": "aaa", "spans": []}] for n in range(1, 6)]
            for outer in range(1, 6):
                (root / f"weights/ablation47/v4/fold{outer}").mkdir(parents=True)
                for inner in range(1, 6):
                    if outer == inner:
                        continue
                    primary = inner == (outer % 5) + 1
                    record = {"status": "готово", "variant": "v4", "git_sha": "git", "gold_sha256": "sha", "fold_ids_sha256": ["id"] * 5, "train_id_sha256": "train", "outer_id_sha256": "outer", "inner_fold": inner, "primary": primary, "seeds": [{"seed": seed, "epochs_ran": 5, "best_epoch": 4, "checkpoint": {"restore_predictions_equal": True}, **({"paired_v2": {"epochs_match": True, "best_epoch_match": True, "torch_match": True, "transformers_match": True, "max_loss_abs_diff": 0.0}} if seed in (0, 1) else {})} for seed in (0, 1, 2, 3, 4)], "inner_grid": [], "elapsed_s": 2}
                    if primary:
                        record["outer_predictions"] = {f"pair{outer}": [[0, 3, 0.5]]}
                    (root / f"artifacts/ablation47_v4_outer{outer}_inner{inner}.json").write_text(json.dumps(record), encoding="utf-8")
            choices = []
            def choose(_):
                choices.append(True)
                return {"start": 0.5, "extend": 0.25}
            actual = {name: 0.0 for name in ("token_precision", "token_recall", "token_f1", "pair_f1", "fpr_pair", "fpr_token")}
            def evaluate(*args):
                self.assertEqual(len(choices), 5)
                return dict(actual)
            extra = {**actual, "exact_f1": 0.0, "overlap_f1": 0.0}
            with patch.object(collect47, "ROOT", root), patch.object(collect47, "read_jsonl", return_value=sum(folds, [])), patch.object(collect47, "grouped_folds", return_value=folds), patch.object(collect47, "file_digest", return_value="sha"), patch.object(collect47, "id_digest", return_value="id"), patch.object(collect47, "select_worst", side_effect=choose), patch.object(collect47, "evaluate", side_effect=evaluate), patch.object(collect47, "verify_fast"), patch.object(collect47, "predict", return_value=[]), patch.object(collect47, "compute", return_value=extra):
                result = collect47.collect("v4")
            self.assertEqual(result["status"], "готово")
            self.assertEqual(result["saved_checkpoints"], 25)
            self.assertEqual(len(result["epochs_ran"]), 100)
            self.assertTrue((root / "weights/ablation47/v4/fold5/manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
