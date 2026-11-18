import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.collect_nested38 import select_worst
from tools.evaluate_snap42 import compute
from tools.nested38 import file_digest, target_inner
from tools.nested_cv35 import evaluate, predict, verify_fast
from tools.preflight_13 import id_digest
from tools.train_base33 import grouped_folds
from tools.train_scheme13_full import save_json

METRICS = ("token_precision", "token_recall", "token_f1", "exact_f1", "overlap_f1", "pair_f1", "fpr_pair", "fpr_token")


def collect(variant, pairs_name="artifacts/ours33.jsonl", prefix="ablation51_ten",
            summary_name=None):
    source = ROOT / pairs_name
    reference = json.loads((ROOT / "artifacts/nested38_summary.json").read_text(encoding="utf-8"))
    pairs = read_jsonl(source)
    folds = grouped_folds(pairs, 10)
    if file_digest(source) != reference["gold_train_val_sha256"] and source.name == "ours33.jsonl":
        raise ValueError("не совпадают train/val или фолды")
    records = {}
    for outer in range(1, 11):
        for inner in [((outer - 1 + offset) % 10) + 1 for offset in range(1, 5)]:
            if outer == inner:
                continue
            path = ROOT / f"artifacts/{prefix}_outer{outer}_inner{inner}.json"
            if path.exists():
                records[outer, inner] = json.loads(path.read_text(encoding="utf-8"))
    target = ROOT / (summary_name or f"artifacts/ablation51_{variant}_summary.json")
    output = {"status": "неполный прогон", "variant": variant, "jobs_expected": 40, "jobs_available": len(records), "jobs_complete": sum(row.get("status") == "готово" for row in records.values()), "seeds": [0, 1, 2, 3, 4], "test301": False, "external_data": False, "sentence_snap": False, "clause_expand": True, "inner_fpr_limit": 0.4, "warning": "Внешние фолды использовались ранее и теперь в правиле остановки абляции; независимого подтверждения нет."}
    save_json(target, output)
    if len(records) != 40 or any(row.get("status") != "готово" for row in records.values()):
        return output
    shas = {row["git_sha"] for row in records.values()}
    expected_sha = file_digest(source)
    if len(shas) != 1 or any(row["variant"] != variant or row["gold_sha256"] != expected_sha or row["fold_ids_sha256"] != [id_digest(fold) for fold in folds] or [seed["seed"] for seed in row["seeds"]] != [0, 1, 2, 3, 4] for row in records.values()):
        raise ValueError("несовпадение варианта, исходных данных, версии или сидов")
    choices = {}
    for outer in range(1, 11):
        group = [records[outer, inner] for inner in [((outer - 1 + offset) % 10) + 1 for offset in range(1, 5)]]
        choice = select_worst([row["inner_grid"] for row in group])
        if choice is None:
            output["status"] = f"STOP: нет допустимого порога для внешнего фолда {outer}"
            save_json(target, output)
            return output
        choices[outer] = choice
    results = []
    for outer in range(1, 11):
        primary = records[outer, outer % 10 + 1]
        held = folds[outer - 1]
        raw = primary["outer_predictions"]
        if not primary["primary"] or set(raw) != {row["pair_id"] for row in held} or any(not seed["checkpoint"] or not seed["checkpoint"]["restore_predictions_equal"] for seed in primary["seeds"]):
            raise ValueError("нет внешних вероятностей или проверенных весов")
        cache = {row["pair_id"]: {"answer": row["model_output"], "pieces": raw[row["pair_id"]]} for row in held}
        threshold = choices[outer]["start"]
        actual = evaluate(held, cache, {row["pair_id"]: threshold for row in held})
        verify_fast(held, cache, threshold, actual)
        rows = [{"pair_id": row["pair_id"], "answer": row["model_output"], "gold_spans": row["spans"], "predicted_spans": predict(cache[row["pair_id"]], threshold)} for row in held]
        extra = compute(rows)
        for name in ("token_precision", "token_recall", "token_f1", "pair_f1", "fpr_pair", "fpr_token"):
            if extra[name] != actual[name]:
                raise ValueError(f"метрики не совпадают: {name}")
        actual.update({"exact_f1": extra["exact_f1"], "overlap_f1": extra["overlap_f1"]})
        results.append({"fold": outer, "threshold": choices[outer], "metrics": actual})
        manifest = {"status": "готово", "variant": variant, "outer_fold": outer, "inner_fold": primary["inner_fold"], "git_sha": primary["git_sha"], "gold_train_val_sha256": primary["gold_sha256"], "train_id_sha256": primary["train_id_sha256"], "outer_id_sha256": primary["outer_id_sha256"], "model": "rubert-tiny2 + 5 token lexical features", "gate": 0.45, "threshold": choices[outer], "clause_expand": True, "sentence_snap": False, "checkpoints": [seed["checkpoint"] for seed in primary["seeds"]], "restore_verified": True}
        save_json(ROOT / "weights/ablation51" / variant / f"fold{outer}" / "manifest.json", manifest)
    histories = [seed for record in records.values() for seed in record["seeds"]]
    output.update({"status": "готово", "git_sha": shas.pop(), "folds": results, "summary": {name: {"mean": round(statistics.mean(row["metrics"][name] for row in results), 4), "max": round(max(row["metrics"][name] for row in results), 4)} for name in METRICS}, "epochs_ran": [seed["epochs_ran"] for seed in histories], "best_epochs": [seed["best_epoch"] for seed in histories], "sum_job_seconds": round(sum(row["elapsed_s"] for row in records.values()), 3), "max_job_seconds": round(max(row["elapsed_s"] for row in records.values()), 3), "saved_checkpoints": 50, "primary_only": True})
    save_json(target, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("ten",), required=True)
    parser.add_argument("--pairs", default="artifacts/ours33.jsonl")
    parser.add_argument("--records-prefix", default="ablation51_ten")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()
    result = collect(args.variant, args.pairs, args.records_prefix, args.summary_out)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "готово" else 2


if __name__ == "__main__":
    raise SystemExit(main())
