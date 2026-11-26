"""Разбор ошибок RusHallu val: какие границы теряет ранкер.

Только val (dev), held-out не читается. Для каждого gold-спана ищем
oracle-кандидата среди candidates_v4, считаем его ранг/логит по baseline,
сравниваем с текущим предсказанием и кодируем направление ошибки.
"""
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split
from spanver.module import document_context

ranker._WEIGHTS = json.loads((ROOT / "artifacts/ranker_weights_v3.json").read_text(encoding="utf-8"))


def category(gold, answer):
    length = gold["end"] - gold["start"] + 1
    if length >= 0.8 * max(1, len(answer)):
        return "whole"
    if length < 40:
        return "short"
    return "clause"


def main():
    val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")
    stats = collections.defaultdict(collections.Counter)
    ranks = collections.defaultdict(list)
    examples = collections.defaultdict(list)
    for pair in val:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        doc_tokens, _stems, doc_grams = document_context(document)
        pool = candidates_v4(answer)
        scored = []
        for start, end, text in pool:
            x = ranker.features(text, start, end, answer, document, doc_tokens, doc_grams)
            risk = ranker._logit(x, ranker._WEIGHTS["cand"], ranker.FEATURE_COUNT)
            scored.append({"start": start, "end": end, "text": text, "risk": risk})
        scored.sort(key=lambda row: -row["risk"])
        predicted = ranker.select_spans(answer, document)
        for gold in pair.get("spans") or []:
            kind = category(gold, answer)
            overlaps = [(iou(item["start"], item["end"], gold["start"], gold["end"]), index)
                        for index, item in enumerate(scored)]
            best_iou, best_index = max(overlaps)
            best = scored[best_index]
            stats[kind]["gold"] += 1
            ranks[kind].append(best_index + 1)
            stats[kind]["recovered"] += int(best_iou >= 0.5)
            pred_iou = max((iou(p["start"], p["end"], gold["start"], gold["end"]) for p in predicted), default=0.0)
            stats[kind]["hit_current"] += int(pred_iou >= 0.5)
            for low, high in ((0.3, 0.5), (0.5, 0.8), (0.8, 0.999)):
                stats[kind][f"near_{low}_{high}"] += sum(
                    1 for value, _ in overlaps if low <= value <= high)
            if pred_iou < 0.5:
                stats[kind]["miss"] += 1
                if best_iou >= 0.5:
                    stats[kind]["oracle_exists_but_not_selected"] += 1
                    if len(examples[kind]) < 5:
                        examples[kind].append({
                            "pair_id": pair["pair_id"],
                            "answer_chars": len(answer),
                            "gold": [gold["start"], gold["end"]],
                            "gold_text": answer[gold["start"]:gold["end"] + 1][:180],
                            "oracle_candidate": [best["start"], best["end"]],
                            "oracle_text": best["text"][:180],
                            "oracle_iou": round(best_iou, 3),
                            "oracle_rank": best_index + 1,
                            "oracle_risk": round(best["risk"], 4),
                            "top_risk": round(scored[0]["risk"], 4),
                            "predicted": [[p["start"], p["end"], p["risk"]] for p in predicted],
                        })
                else:
                    stats[kind]["oracle_missing"] += 1
            if pred_iou >= 0.5:
                chosen = max(predicted, key=lambda p: iou(p["start"], p["end"], gold["start"], gold["end"]))
                width_gold = gold["end"] - gold["start"] + 1
                width_pred = chosen["end"] - chosen["start"] + 1
                stats[kind]["wide"] += int(width_pred > width_gold * 1.2)
                stats[kind]["narrow"] += int(width_pred < width_gold * 0.8)
                stats[kind]["shifted"] += int(abs(chosen["start"] - gold["start"]) > 0.2 * width_gold)
    report = {
        "rule": "val_v3 only; baseline weights artifacts/ranker_weights_v3.json; oracle among candidates_v4",
        "stats": {kind: dict(counter) for kind, counter in stats.items()},
        "median_oracle_rank": {kind: statistics.median(values) if values else 0
                               for kind, values in ranks.items()},
        "examples": {kind: items for kind, items in examples.items()},
    }
    target = ROOT / "artifacts/rushallu_val_error_analysis_v10.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "stats": report["stats"],
        "median_oracle_rank": report["median_oracle_rank"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
