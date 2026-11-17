"""ШАГ B: пилот авторазметки.

Задача — проверить, годится ли автоматический разметчик для расширения
обучения, ДО масштабирования. Сравнение идёт с человеческими метками на
парах, которых разметчик не видел.

Разметчик: обученная модель проекта (rubert-tiny2 + 5 лексических признаков,
веса V5, seed 0, внешний фолд 1 не участвовал в обучении этого сида), то есть
ровно тот сигнал, который потом будет обучаться. Промптовой LLM в песочнице нет
(внешняя сеть закрыта), поэтому «сильная модель-разметчик» недоступна — это
ограничение метода, а не выбора; в отчёте оно указывается явно.

Метрики согласия с человеческими спанами, как в договоре:
  * IoU совпавших спанов (жадное сопоставление по максимальному IoU);
  * exact (IoU == 1.0);
  * бинарный F1 по «есть нарушение / нет» на уровне пары;
  * согласие типа на совпавших спанах.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.predict_frozen38 import load_state
from tools.train_base33 import grouped_folds

GATE = 0.45
START = 0.50
EXTEND = 0.25
MAX_SPANS = 3


def iou(left, right):
    intersection = min(left["end"], right["end"]) - max(left["start"], right["start"]) + 1
    if intersection <= 0:
        return 0.0
    union = max(left["end"], right["end"]) - min(left["start"], right["start"]) + 1
    return intersection / union


def match(predicted, gold):
    """Жадное сопоставление: каждая предсказанная пара ищет лучший свободный золотой спан."""
    used = set()
    pairs = []
    for span in predicted:
        best, index = 0.0, None
        for position, truth in enumerate(gold):
            if position in used:
                continue
            value = iou(span, truth)
            if value > best:
                best, index = value, position
        if index is not None and best > 0.0:
            used.add(index)
            pairs.append((span, gold[index], best))
    return pairs


def metrics(rows):
    """Согласие авторазметчика с человеческими метками.

    Три независимых числа:
      * iou_mean — средний IoU по сопоставленным спанам (жадно, максимальный IoU);
      * exact_rate — доля предсказанных спанов, точно совпавших с человеческим;
      * binary_f1 — уровень ПАРЫ: «есть нарушение / нет». Это то, что просит
        договор; смешивать его со span-метриками нельзя.
    """
    predicted_spans = exact = matched = 0
    iou_sum = 0.0
    type_right = type_total = 0
    tp = fp = fn = tn = 0
    for row in rows:
        predicted, gold = row["predicted"], row["gold"]
        predicted_spans += len(predicted)
        for span, truth, value in match(predicted, gold):
            iou_sum += value
            matched += 1
            if value == 1.0:
                exact += 1
            type_total += 1
            if span.get("type") == truth.get("type"):
                type_right += 1
        has_gold = bool(gold)
        has_predicted = bool(predicted)
        if has_gold and has_predicted:
            tp += 1
        elif has_predicted:
            fp += 1
        elif has_gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pairs": len(rows),
        "predicted_spans": predicted_spans,
        "gold_spans": sum(len(row["gold"]) for row in rows),
        "matched_spans": matched,
        "iou_mean": iou_sum / matched if matched else 0.0,
        "exact_rate": exact / predicted_spans if predicted_spans else 0.0,
        "type_agreement": type_right / type_total if type_total else 0.0,
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "binary_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--out", default="artifacts/autolabel52_pilot.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    held = grouped_folds(pairs, 5)[0]
    chosen = held[: args.pairs]
    manifest = json.loads((ROOT / "weights/final/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", labels=2, seed=entry["seed"])
    model.base.config._attn_implementation = "eager"
    load_state(torch, model, ROOT / "weights/final" / f"seed_{entry['seed']}", entry)

    from spanver.decode import spans_from_flags
    from spanver.snap import snapped

    rows = []
    for pair in chosen:
        value = lexfeat.pair_probabilities(tokenizer, model, pair, top=4)
        pieces = value["pieces"] if value and value["pieces"] else []
        predicted = []
        if pieces and max(piece[2] for piece in pieces) >= GATE:
            flags = []
            active = False
            for _, _, score in pieces:
                active = score >= (EXTEND if active else START)
                flags.append(active)
            spans = spans_from_flags(pair["model_output"], pieces, flags, 2)
            predicted = sorted(snapped(spans, pair["model_output"], "expand"),
                               key=lambda item: (-item["risk"], item["start"]))[:MAX_SPANS]
        rows.append({"pair_id": pair["pair_id"], "predicted": predicted, "gold": pair["spans"]})

    result = metrics(rows)
    result.update({
        "status": "готово",
        "scope": f"пилот: {len(chosen)} позитивных пар внешнего фолда 1, разметчик не видел их при обучении",
        "labeler": "spanver tiny (rubert-tiny2 + 5 признаков), веса V5 seed 0, gate 0.45 start 0.50 extend 0.25 max_spans 3",
        "labeler_limit": "промптовой LLM в песочнице нет, поэтому разметчик — та же модель, а не внешняя сильная; согласие с ней измеряет воспроизводимость, а не независимое качество разметки",
        "iou_threshold_pass": 0.60,
        "iou_threshold_conditional": 0.45,
        "verdict": ("шаг C" if result["iou_mean"] >= 0.60 and result["binary_f1"] >= 0.75 else
                    "шаг C с весом 0.3 и половинным объёмом" if 0.45 <= result["iou_mean"] < 0.60 else
                    "стоп, не масштабировать"),
    })
    (ROOT / args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
