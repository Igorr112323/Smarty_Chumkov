import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import match_spans


def analyze(rows):
    exact = overlap = pred_count = gold_count = unequal = overlapping = multi = 0
    for row in rows:
        predicted = row["predicted_spans"]
        gold = row["gold_spans"]
        pred_count += len(predicted)
        gold_count += len(gold)
        multi += len(predicted) > 1
        exact += match_spans(predicted, gold, 1.0)[0]
        overlap += match_spans(predicted, gold, 1e-15)[0]
        for span in predicted:
            matches = [item for item in gold if min(span["end"], item["end"]) >= max(span["start"], item["start"])]
            overlapping += bool(matches)
            unequal += bool(matches) and not any((span["start"], span["end"]) == (item["start"], item["end"]) for item in matches)
    total = pred_count + gold_count
    return {"pairs": len(rows), "predicted_spans": pred_count, "gold_spans": gold_count, "pairs_with_multiple_predictions": multi, "predictions_overlapping_gold": overlapping, "overlapping_without_exact_gold": unequal, "exact_tp": exact, "any_overlap_tp": overlap, "exact_f1": 2 * exact / total if total else 0.0, "any_overlap_f1": 2 * overlap / total if total else 0.0}


def main():
    manifest = json.loads((ROOT / "artifacts/predictions40_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "baseline nested38 recovered, not round40":
        raise ValueError("ожидался кэш старой базовой линии")
    folds = []
    for item in manifest["outputs"]:
        raw = (ROOT / item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("SHA предсказаний не совпал")
        rows = [json.loads(line) for line in raw.splitlines() if line]
        if len(rows) != item["pairs"] or any(row["source"] != "nested38_baseline" for row in rows):
            raise ValueError("неверный набор предсказаний")
        folds.append(analyze(rows))
    if len(folds) != 5 or sum(row["pairs"] for row in folds) != 699:
        raise ValueError("нужно пять внешних фолдов, 699 пар")
    exact = statistics.mean(row["exact_f1"] for row in folds)
    overlap = statistics.mean(row["any_overlap_f1"] for row in folds)
    result = {"status": "baseline only; no round40 training", "protocol": "После gate; спаны сопоставляются 1:1 в порядке предсказаний и gold в пределах пары, тип не учитывается. Exact: равны обе включительные границы; overlap: хотя бы один общий символ. F1=2TP/(predicted+gold); затем среднее по пяти фолдам и population std. Это не посимвольная F1 из spanver/lt/evaluator.py.", "folds": folds, "mean_exact_f1": exact, "std_exact_f1": statistics.pstdev(row["exact_f1"] for row in folds), "mean_any_overlap_f1": overlap, "std_any_overlap_f1": statistics.pstdev(row["any_overlap_f1"] for row in folds), "gap_percentage_points": 100 * (overlap - exact), "relative_increase_percent": 100 * (overlap - exact) / exact if exact else None, "predicted_spans": sum(row["predicted_spans"] for row in folds), "gold_spans": sum(row["gold_spans"] for row in folds), "pairs_with_multiple_predictions": sum(row["pairs_with_multiple_predictions"] for row in folds), "overlapping_without_exact_gold": sum(row["overlapping_without_exact_gold"] for row in folds)}
    target = ROOT / "artifacts/predictions40_baseline_analysis.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
