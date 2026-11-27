"""V11: быстрый стенд декодирования поверх обученного listwise-ранкера.

Кэширует (context, scores) для val+mFAVA в pickle и перебирает стратегии:
  topk        — top-k кандидатов выше порога (классический decode);
  hedge       — whole-ответ + top-k clause (не требует угадать гранулярность);
  whole_first — whole, если его скор не ниже лучшего clause;
  router      — режим из router_v11 (если передан).
Held-out не читается.
"""
import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import v11_lab as L
import v11_ranker as R
from spanver.evaluate import iou, metrics


def load_model(path):
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return payload, payload["columns"]


def build_cache(pairs, payload, cols, params, path):
    if Path(path).exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)
    cache = {}
    for pair in pairs:
        ctx = R.pair_context(pair, params)
        scored = []
        if ctx["rows"]:
            X = np.array([R.vector(ctx, row["start"], row["end"], row["text"]) for row in ctx["rows"]],
                         dtype=np.float64)
            scored = [(row, R.predict_raw(payload, list(row[cols]))) for row in X]
            scored = [(row, score) for row, score in zip(ctx["rows"], [s for _, s in scored])]
        cache[pair["pair_id"]] = (ctx, scored)
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)
    return cache


def emit(row, score, kind="clause"):
    return {"start": row["start"], "end": row["end"], "risk": round(float(score), 4),
            "type": "unsupported_addition", "confidence": 0.5 if kind == "clause" else 0.9}


def whole_span(ctx):
    answer = ctx["answer"]
    return {"start": 0, "end": len(answer) - 1, "risk": round(ctx["gate"], 4),
            "type": "unsupported_addition", "confidence": round(ctx["gate"], 4)}


def decode(strategy, ctx, scored, config):
    gate_cut = config["gate"]
    if ctx["gate"] < gate_cut or not ctx["answer"]:
        return []
    answer = ctx["answer"]
    whole = whole_span(ctx)
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]["start"]))
    clause_box = []
    for row, score in ranked:
        if score < config["threshold"]:
            break
        if row["start"] == 0 and row["end"] == len(answer) - 1:
            continue
        if any(iou(row["start"], row["end"], other["start"], other["end"]) > config["nms"]
               for other in clause_box):
            continue
        clause_box.append({"start": row["start"], "end": row["end"], "risk": round(float(score), 4),
                           "type": "unsupported_addition", "confidence": 0.5})
        if len(clause_box) >= config["max_clauses"]:
            break
    if strategy == "topk":
        out = []
        for row, score in ranked:
            if score < config["threshold"]:
                break
            if any(iou(row["start"], row["end"], other["start"], other["end"]) > config["nms"] for other in out):
                continue
            out.append(emit(row, score))
            if len(out) >= config["max_spans"]:
                break
        return out
    if strategy == "hedge":
        return [whole] + clause_box
    if strategy == "whole_first":
        best_clause = clause_box[0]["risk"] if clause_box else -99
        whole_score = next((score for row, score in ranked
                            if row["start"] == 0 and row["end"] == len(answer) - 1), -99)
        if whole_score >= best_clause:
            return [whole]
        return clause_box
    raise SystemExit(strategy)


def evaluate(cache, pairs, strategy, config):
    picked = {pair["pair_id"]: decode(strategy, *cache[pair["pair_id"]], config) for pair in pairs}
    return metrics(pairs, lambda pair: picked[pair["pair_id"]])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/listwise_whole_model_v11.json")
    parser.add_argument("--cache", default="artifacts/decode_cache_v11.pkl")
    parser.add_argument("--report", default="artifacts/decode_v11.json")
    args = parser.parse_args(argv)

    params = L.weights()
    payload, cols = load_model(args.model)
    _train, val, mfava = L.load_all()
    cache = build_cache(val + mfava, payload, cols, params, args.cache)

    rows = []
    for strategy in ("topk", "hedge", "whole_first"):
        for gate_cut in (0.3, 0.5, 0.6):
            for threshold, max_clauses, max_spans in itertools.product(
                    (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0), (1, 2, 3), (1, 2, 3)):
                config = {"gate": gate_cut, "threshold": threshold, "max_clauses": max_clauses,
                          "max_spans": max_spans, "nms": 0.1}
                combined = evaluate(cache, val + mfava, strategy, config)
                rows.append({
                    "strategy": strategy, "gate": gate_cut, "threshold": threshold,
                    "max_clauses": max_clauses, "max_spans": max_spans,
                    "combined": L.cell(combined),
                    "val": L.cell(evaluate(cache, val, strategy, config)),
                    "mfava": L.cell(evaluate(cache, mfava, strategy, config)),
                })
    rows.sort(key=lambda row: (row["val"]["span"] + row["mfava"]["span"]) / 2, reverse=True)
    passing = [row for row in rows if L.guard([row["val"], row["combined"]])]
    report = {
        "rule": "decode-стратегии на валидации+mFAVA; listwise-ранкер обучен на train-only; held не читается",
        "model": args.model, "best": rows[:15], "passing": passing[:10],
        "by_strategy": {name: max((row for row in rows if row["strategy"] == name),
                                  key=lambda row: row["val"]["span"], default=None)
                        for name in ("topk", "hedge", "whole_first")},
        "configs": rows,
    }
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in report["by_strategy"].items():
        if row:
            print(name, "val", row["val"], "mfava", row["mfava"], flush=True)
    print("best", json.dumps(rows[0], ensure_ascii=False), flush=True)
    print("guard", len(passing), flush=True)


if __name__ == "__main__":
    main()
