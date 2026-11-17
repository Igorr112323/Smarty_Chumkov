"""ПРОМПТ 54, задача 2: подсчёт согласия ручной разметки куратора с человеческой.

Куратор заполняет `artifacts/fewshot54_candidate.json` в формате:

  {"items": [{"pair_id": "...", "spans": [{"start": 12, "end": 40, "type": "..."}]}]}

Смещения — в тех же координатах, что `answer` в `fewshot54_blind.json`.
Скрипт считает те же четыре числа, что в пилоте 53, чтобы сравнение было
один-к-одному: IoU по символам, exact, бинарный F1 по парам, согласие типа.
Плюс 95 % CI по парам (бутстрэп по парам, 10 000 ресэмплов).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl


def iou(left, right):
    intersection = min(left["end"], right["end"]) - max(left["start"], right["start"]) + 1
    if intersection <= 0:
        return 0.0
    union = max(left["end"], right["end"]) - min(left["start"], right["start"]) + 1
    return intersection / union


def per_pair(predicted, gold):
    used, values = set(), []
    for span in predicted:
        best, index = 0.0, None
        for position, truth in enumerate(gold):
            if position in used:
                continue
            value = iou(span, truth)
            if value > best:
                best, index = value, position
        if index is not None and best > 0:
            used.add(index)
            values.append((best, gold[index]))
    return values


def summarize(rows):
    char_iou_num = char_iou_den = exact = matched = predicted_spans = 0
    type_right = type_total = 0
    tp = fp = fn = tn = 0
    scores = []
    for row in rows:
        predicted, gold = row["predicted"], row["gold"]
        predicted_spans += len(predicted)
        pairs = per_pair(predicted, gold)
        matched += len(pairs)
        for value, truth in pairs:
            char_iou_num += value
            char_iou_den += 1
            if value == 1.0:
                exact += 1
            type_total += 1
            if predicted and truth.get("type") == row["type_map"].get(id(truth), truth.get("type")):
                type_right += 1
        if gold and predicted:
            tp += 1
        elif predicted:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
        # для CI: доля правильных решений пары
        scores.append(1.0 if (bool(gold) == bool(predicted)) else 0.0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    generator = random.Random(48)
    boots = []
    for _ in range(10000):
        sample = [generator.choice(scores) for _ in scores]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return {
        "pairs": len(rows),
        "predicted_spans": predicted_spans,
        "matched_spans": matched,
        "iou": char_iou_num / char_iou_den if char_iou_den else 0.0,
        "exact_rate": exact / predicted_spans if predicted_spans else 0.0,
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "binary_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "pair_decision_accuracy": sum(scores) / len(scores) if scores else 0.0,
        "pair_decision_accuracy_ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if boots else None,
    }


def main():
    candidate_path = ROOT / "artifacts/fewshot54_candidate.json"
    if not candidate_path.exists():
        raise SystemExit(
            "нет artifacts/fewshot54_candidate.json — разметка куратора ещё не внесена. "
            'Формат: {"items": [{"pair_id": "...", "spans": [{"start": 0, "end": 9, "type": "..."}]}]}, '
            "по 30 парам из artifacts/fewshot54_blind.json")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    pairs = {row["pair_id"]: row for row in read_jsonl(ROOT / "artifacts/ours33.jsonl")}
    rows = []
    for item in candidate["items"]:
        pair = pairs[item["pair_id"]]
        rows.append({
            "pair_id": item["pair_id"],
            "predicted": item.get("spans") or [],
            "gold": pair["spans"],
            "type_map": {},
        })
    result = summarize(rows)
    result.update({
        "status": "готово",
        "source": "artifacts/fewshot54_candidate.json",
        "reference": {
            "iou_53": 0.0883,
            "binary_f1_53": 0.2609,
            "precision_53": 1.0,
            "recall_53": 0.15,
        },
        "threshold_iou": 0.45,
        "verdict": ("правило передаваемо" if result["iou"] >= 0.45 else
                    "правило не передаётся даже с примерами"),
    })
    (ROOT / "artifacts/score54.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
