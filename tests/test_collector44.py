import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import collect44
from tools.preflight_13 import components, id_digest, our_keys, split_groups


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def pieces(row):
    return [[0, len(row["model_output"]), 0.9 if row["spans"] else 0.1]]


class Collector44Tests(unittest.TestCase):
    def test_complete_synthetic_nested_protocol(self):
        order = list(range(10))
        random.Random(0).shuffle(order)
        positives = set(order[:5])
        data = []
        for index in range(10):
            positive = index in positives
            answer = "Ошибка." if positive else "Верно."
            data.append({"pair_id": f"p{index}", "doc_ids": [f"d{index}"], "model_output": answer, "spans": [{"start": 0, "end": 6, "type": "contradiction"}] if positive else []})
        folds = split_groups(components(data, our_keys), 5, 0)
        self.assertTrue(all(sum(bool(row["spans"]) for row in fold) == 1 for fold in folds))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            source = artifacts / "ours33.jsonl"
            write_jsonl(source, data)
            gold_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            fold_ids = [id_digest(fold) for fold in folds]
            for outer in range(1, 6):
                for inner in range(1, 6):
                    if outer == inner:
                        continue
                    seeds = []
                    for seed in (0, 1):
                        rows = [{"pair_id": row["pair_id"], "doc_ids": row["doc_ids"], "fold": outer, "inner_fold": inner, "seed": seed, "answer": row["model_output"], "gold_spans": row["spans"], "pieces": pieces(row), "predicted_spans": []} for row in folds[inner - 1]]
                        name = f"inner_predictions_fold{outer}_inner{inner}_seed{seed}.jsonl"
                        path = artifacts / name
                        write_jsonl(path, rows)
                        seeds.append({"seed": seed, "inner_file": f"artifacts/{name}", "inner_initial_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "epochs_ran": 6, "best_epoch": 5, "seconds": 1.0})
                    report = {"status": "готово", "outer_fold": outer, "inner_fold": inner, "gold_sha256": gold_sha, "fold_ids_sha256": fold_ids, "inner_id_sha256": fold_ids[inner - 1], "outer_id_sha256": fold_ids[outer - 1], "git_sha": "synthetic", "seeds": seeds, "elapsed_s": 2.0}
                    write_json(artifacts / f"train44_fold{outer}_inner{inner}.json", report)
                for seed in (0, 1):
                    raw = [{"pair_id": row["pair_id"], "doc_ids": row["doc_ids"], "fold": outer, "seed": seed, "answer": row["model_output"], "pieces": pieces(row)} for row in folds[outer - 1]]
                    write_jsonl(artifacts / f"outer44_raw_fold{outer}_seed{seed}.jsonl", raw)
            original = collect44.outer_evaluation
            def evaluate_after_all_choices(fold_data, reports, selections, seed_count):
                self.assertEqual(len(selections), 5)
                self.assertTrue(all((artifacts / f"selection44_fold{fold}.json").is_file() for fold in range(1, 6)))
                return original(fold_data, reports, selections, seed_count)
            with patch.object(collect44, "ROOT", root), patch.object(collect44, "EXPECTED_GOLD_SHA", gold_sha), patch.object(collect44, "EXPECTED_IDS", id_digest(data)), patch.object(collect44, "validate_data"), patch.object(collect44, "outer_evaluation", side_effect=evaluate_after_all_choices):
                collect44.main()
            summary = json.loads((artifacts / "summary44.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["seeds_used"], [0, 1])
            self.assertEqual(summary["outer_evaluations"], 1)
            self.assertEqual(summary["summary"]["token_f1"]["mean"], 1.0)
            self.assertEqual(len(list(artifacts.glob("outer_predictions_fold*_seed*.jsonl"))), 10)
            result = json.loads((artifacts / "inner_predictions_fold1_inner2_seed0.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(result["prediction_status"], "selected_on_inner_ensemble")


if __name__ == "__main__":
    unittest.main()
