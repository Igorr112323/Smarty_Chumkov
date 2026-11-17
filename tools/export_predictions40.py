import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl, token_positions
from tools.nested38 import file_digest, target_inner
from tools.nested_cv35 import GATE, evaluate, predict, verify_fast
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.train_base33 import grouped_folds


def word_probabilities(answer, pieces):
    result = []
    for start, end in token_positions(answer):
        overlaps = [float(score) for left, right, score in pieces if left <= end and right > start]
        result.append({"start": start, "end": end, "probability": max(overlaps) if overlaps else None})
    return result


def export_rows(pairs, probabilities, start, fold):
    if set(probabilities) != {row["pair_id"] for row in pairs}:
        raise ValueError("ID вероятностей не совпадают с внешним фолдом")
    output = []
    for row in pairs:
        pieces = probabilities[row["pair_id"]]
        value = {"answer": row["model_output"], "pieces": pieces}
        score = max((piece[2] for piece in pieces), default=0.0)
        output.append({"pair_id": row["pair_id"], "doc_ids": row["doc_ids"], "fold": fold, "source": "nested38_baseline", "answer": value["answer"], "pair_prob": score, "pair_prob_definition": "max_piece_probability_gate_score_not_separate_pair_head", "gate_threshold": GATE, "start_threshold": start, "extend_threshold": start / 2, "word_probabilities": word_probabilities(value["answer"], pieces), "pieces": pieces, "gold_spans": row["spans"], "predicted_spans": predict(value, start)})
    return output


def main():
    pairs_path = ROOT / "artifacts/ours33.jsonl"
    pairs = read_jsonl(pairs_path)
    summary = json.loads((ROOT / "artifacts/nested38_summary.json").read_text(encoding="utf-8"))
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS or file_digest(pairs_path) != summary["gold_train_val_sha256"]:
        raise ValueError("неверный набор train/val")
    folds = grouped_folds(pairs, 5)
    if [id_digest(bucket) for bucket in folds] != summary["fold_ids_sha256"]:
        raise ValueError("разбиение на фолды изменилось")
    outputs = []
    for outer, held in enumerate(folds, 1):
        inner = target_inner(outer - 1) + 1
        source = ROOT / f"artifacts/nested38_outer_{outer}_inner_{inner}.json"
        record = json.loads(source.read_text(encoding="utf-8"))
        if not record["outer_primary"] or record["status"] != "готово" or record["outer_fold"] != outer or record["inner_fold"] != inner or record["gold_train_val_sha256"] != summary["gold_train_val_sha256"] or record["fold_ids_sha256"] != summary["fold_ids_sha256"]:
            raise ValueError("неверное происхождение вероятностей")
        probabilities = record["outer_predictions"]
        threshold = summary["folds"][outer - 1]["threshold"]["start"]
        cache = {row["pair_id"]: {"answer": row["model_output"], "pieces": probabilities[row["pair_id"]]} for row in held}
        actual = evaluate(held, cache, {row["pair_id"]: threshold for row in held})
        if actual != summary["folds"][outer - 1]["metrics"]:
            raise ValueError("метрики по восстановленному кэшу не совпадают")
        verify_fast(held, cache, threshold, actual)
        rows = export_rows(held, probabilities, threshold, outer)
        target = ROOT / f"artifacts/predictions_fold{outer}.jsonl"
        target.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        outputs.append({"path": str(target.relative_to(ROOT)), "pairs": len(rows), "sha256": file_digest(target), "bytes": target.stat().st_size})
    (ROOT / "artifacts/predictions40_manifest.json").write_text(json.dumps({"status": "baseline nested38 recovered, not round40", "run_id": 43914581442, "gold_train_val_sha256": summary["gold_train_val_sha256"], "outputs": outputs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
