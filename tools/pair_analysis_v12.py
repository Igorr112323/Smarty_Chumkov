"""V12 шаг 1: разбор Pair-ошибок v11 hybrid на val.

Воспроизводит декодер v11 (v3 mode gate + boundary adjuster на global-пути),
считает Pair-матрицу TP/FP/FN/TN и сохраняет по каждому FP/FN признаки:
гейт пары, max/min риск кандидатов, gap top1-top2, число кандидатов выше порогов,
скоры local HGB, длину ответа, число предложений, document support ответа,
долю чисел/имён, какой декодер сработал. Отвечает на вопрос: FN или FP больше.
Held-out не читается.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import boundary_adjust_v11 as B
import v11_lab as L
from spanver.evaluate import iou, metrics, read_jsonl
from spanver.local_ranker import logistic, predict_trees


def candidate_stats(pair, params, cache):
    """Признаки пары для разбора ошибок и для pair-классификатора."""
    rows, gate = cache.get(pair)
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    base = [row["risk"] for row in rows]
    local = [predict_trees(params["local_hgb"], row["features"]) for row in rows]
    ordered = sorted(base, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    top5 = ordered[:5]
    support = L.support_features(answer, document)
    words = L.WORD_RE.findall(answer)
    digits = sum(1 for ch in answer if ch.isdigit())
    capitals = sum(1 for word in words if word[:1].isupper())
    from spanver import ranker
    mode_probability = logistic(ranker.mode_features(
        [{"start": row["start"], "end": row["end"], "text": row["text"], "risk": risk}
         for row, risk in zip(rows, base)], answer, document),
        params["mode"], ranker.MODE_FEATURE_COUNT)
    return {
        "gate": round(gate, 4),
        "base_top1": round(top1, 4),
        "base_top2": round(top2, 4),
        "base_gap": round(top1 - top2, 4),
        "base_top5_mean": round(float(np.mean(top5)), 4) if top5 else 0.0,
        "base_mean": round(float(np.mean(base)), 4) if base else 0.0,
        "n_cand": len(rows),
        "n_above_03": sum(1 for value in base if value > 0.3),
        "n_above_05": sum(1 for value in base if value > 0.5),
        "n_above_06": sum(1 for value in base if value > 0.6),
        "local_top1": round(max(local), 4) if local else 0.0,
        "local_top2_mean": round(float(np.mean(sorted(local, reverse=True)[:2])), 4) if local else 0.0,
        "mode_probability": round(mode_probability, 4),
        "answer_len": len(answer),
        "n_sentences": len(L.SENT_RE.findall(answer)),
        "n_clauses": len(L.clause_spans(answer)),
        "answer_cov": round(support["token_cov"], 4),
        "answer_cos": round(support["cos"], 4),
        "digit_share": round(digits / max(1, len(answer)), 4),
        "capital_share": round(capitals / max(1, len(words)), 4),
        "numbers_missing": round(support["numbers_missing"], 4),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/pair_report_v12.json")
    parser.add_argument("--cut", type=float, default=0.5)
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    cache = L.Cache(params=params)
    train, val, mfava = L.load_all()
    config = dict(params["config"])
    rows = B.collect(train, cache, config, "hybrid")
    model, scaler = B.train(rows)

    def decoder(pair):
        return B.adjust(pair, cache, config, model, scaler, args.cut, "hybrid")

    picked = {pair["pair_id"]: decoder(pair) for pair in val}
    result = metrics(val, lambda pair: picked[pair["pair_id"]])
    tp = fp = fn = 0
    negatives = 0
    negative_flagged = 0
    for pair in val:
        gold = pair.get("spans") or []
        predicted = picked[pair["pair_id"]]
        matched = any(iou(span["start"], span["end"], truth["start"], truth["end"]) >= 0.5
                      for span in predicted for truth in gold)
        if gold:
            tp += int(matched)
            fn += int(not matched)
        else:
            negatives += 1
            if predicted:
                fp += 1
                negative_flagged += 1
    tn = negatives - negative_flagged

    errors = {"fn": [], "fp": []}
    for pair in val:
        gold = pair.get("spans") or []
        predicted = picked[pair["pair_id"]]
        matched = any(iou(span["start"], span["end"], truth["start"], truth["end"]) >= 0.5
                      for span in predicted for truth in gold)
        stats = candidate_stats(pair, params, cache)
        stats["pair_id"] = pair["pair_id"]
        stats["gold_spans"] = len(gold)
        stats["predicted_spans"] = len(predicted)
        stats["decoder"] = "local" if B.is_local_mode(pair, params) else "global"
        stats["predicted"] = [[span["start"], span["end"]] for span in predicted]
        stats["gold"] = [[span["start"], span["end"]] for span in gold]
        if gold and not matched:
            errors["fn"].append(stats)
        elif not gold and predicted:
            errors["fp"].append(stats)

    def summary(items, keys):
        out = {}
        for key in keys:
            values = [item[key] for item in items if isinstance(item.get(key), (int, float))]
            if values:
                out[key] = {"mean": round(float(np.mean(values)), 4),
                            "p50": round(float(np.median(values)), 4),
                            "min": round(min(values), 4), "max": round(max(values), 4)}
        return out

    keys = ("gate", "base_top1", "base_top2", "base_gap", "base_top5_mean", "n_cand",
            "n_above_03", "n_above_05", "n_above_06", "local_top1", "mode_probability",
            "answer_len", "n_sentences", "n_clauses", "answer_cov", "answer_cos",
            "digit_share", "capital_share")
    tp_items, fp_items, fn_items = [], errors["fp"], errors["fn"]
    for pair in val:
        gold = pair.get("spans") or []
        predicted = picked[pair["pair_id"]]
        matched = any(iou(span["start"], span["end"], truth["start"], truth["end"]) >= 0.5
                      for span in predicted for truth in gold)
        if gold and matched:
            stats = candidate_stats(pair, params, cache)
            stats["decoder"] = "local" if B.is_local_mode(pair, params) else "global"
            tp_items.append(stats)

    report = {
        "decoder": "v11 hybrid (mode gate + boundary adjuster на global-пути)",
        "cut": args.cut,
        "pair_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "negatives": negatives, "positives": tp + fn},
        "pair_precision": round(tp / max(1, tp + fp), 4),
        "pair_recall": round(tp / max(1, tp + fn), 4),
        "pair_f1": rate(tp, fp, fn),
        "fpr": result["fpr"],
        "span": L.cell(result),
        "verdict": "FN больше FP" if fn > fp else ("FP больше FN" if fp > fn else "равно"),
        "fn_count": len(fn_items), "fp_count": len(fp_items), "tp_count": len(tp_items),
        "summary": {"tp": summary(tp_items, keys), "fn": summary(fn_items, keys),
                    "fp": summary(fp_items, keys)},
        "decoders": {
            "tp": {name: sum(1 for item in tp_items if item["decoder"] == name)
                   for name in ("global", "local")},
            "fn": {name: sum(1 for item in fn_items if item["decoder"] == name)
                   for name in ("global", "local")},
            "fp": {name: sum(1 for item in fp_items if item["decoder"] == name)
                   for name in ("global", "local")},
        },
        "fn": fn_items, "fp": fp_items,
    }
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("confusion", json.dumps(report["pair_confusion"], ensure_ascii=False), flush=True)
    print("P/R/F1", report["pair_precision"], report["pair_recall"], report["pair_f1"],
          "FPR", report["fpr"], flush=True)
    print("span", json.dumps(report["span"], ensure_ascii=False), flush=True)
    print("verdict", report["verdict"], "fn", report["fn_count"], "fp", report["fp_count"],
          flush=True)
    print("decoders", json.dumps(report["decoders"], ensure_ascii=False), flush=True)


def rate(tp, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0


if __name__ == "__main__":
    main()
