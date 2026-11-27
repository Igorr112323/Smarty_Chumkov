"""V11: сравнение scorer'ов кандидатов по top-1 точности и span F1.

Scorer'ы: baseline v3 global logit, v3 local HGB (predict_trees), document-support
дефицит, попарные fusion и обученный listwise. Политика «один спан на пару» —
именно она максимизирует precision при 1.17 gold-спанах на пару.
Held-out не читается.
"""
import argparse
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
from spanver.evaluate import iou, metrics, read_jsonl
from spanver.local_ranker import predict_trees


def kind(pair):
    answer = pair.get("model_output") or ""
    spans = pair.get("spans") or []
    if not spans:
        return "neg"
    if any(span["start"] == 0 and span["end"] == len(answer) - 1 for span in spans):
        return "whole"
    return "short" if max(span["end"] - span["start"] + 1 for span in spans) < 60 else "clause"


def build(pairs, params, model, cols):
    cache = {}
    for pair in pairs:
        ctx = R.pair_context(pair, params)
        rows = []
        for row in ctx["rows"]:
            vector = R.vector(ctx, row["start"], row["end"], row["text"])
            support = L.support_features(row["text"], ctx["document"])
            rows.append({
                "start": row["start"], "end": row["end"], "text": row["text"],
                "base": row["risk"],
                "local": predict_trees(params["local_hgb"], row["features"]),
                "support": 0.6 * support["token_cov"] + 0.4 * support["cos"],
                "listwise": R.predict_raw(model, [vector[index] for index in cols]) if model else 0.0,
                "is_whole": float(row["start"] == 0 and row["end"] == len(ctx["answer"]) - 1),
            })
        cache[pair["pair_id"]] = {"ctx": ctx, "rows": rows, "kind": kind(pair)}
    return cache


def scorer(name):
    if name == "base":
        return lambda row, ctx: row["base"]
    if name == "local":
        return lambda row, ctx: row["local"]
    if name == "support":
        return lambda row, ctx: -row["support"]
    if name == "listwise":
        return lambda row, ctx: row["listwise"]
    if name == "base_local":
        return lambda row, ctx: 0.5 * row["base"] + 0.5 * row["local"]
    if name == "base_support":
        return lambda row, ctx: 0.7 * row["base"] + 0.3 * (1 - row["support"])
    if name == "base_listwise":
        return lambda row, ctx: 0.5 * row["base"] + 0.5 * (1 / (1 + np.exp(-row["listwise"])))
    raise SystemExit(name)


def decode_pair(entry, score, gate_cut, max_spans, nms=0.1):
    ctx = entry["ctx"]
    if ctx["gate"] < gate_cut:
        return []
    ranked = sorted(entry["rows"], key=lambda row: (-score(row, ctx), row["start"]))
    chosen = []
    for row in ranked:
        if any(iou(row["start"], row["end"], other["start"], other["end"]) > nms for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= max_spans:
            break
    return [{"start": row["start"], "end": row["end"], "risk": round(float(score(row, ctx)), 4),
             "type": "unsupported_addition", "confidence": 0.5} for row in chosen]


def evaluate(cache, pairs, score, gate_cut, max_spans):
    picked = {pair["pair_id"]: decode_pair(cache[pair["pair_id"]], score, gate_cut, max_spans)
              for pair in pairs}
    return metrics(pairs, lambda pair: picked[pair["pair_id"]])


def top1_accuracy(cache, pairs, score):
    hits = {}
    for pair in pairs:
        entry = cache[pair["pair_id"]]
        if entry["kind"] == "neg" or not entry["rows"]:
            continue
        ctx = entry["ctx"]
        best = max(entry["rows"], key=lambda row: (score(row, ctx), -row["start"]))
        gold = pair.get("spans") or []
        ok = any(iou(best["start"], best["end"], span["start"], span["end"]) >= 0.5 for span in gold)
        hits.setdefault(entry["kind"], []).append(int(ok))
    return {key: {"hits": sum(value), "n": len(value),
                  "acc": round(sum(value) / len(value), 3)} for key, value in hits.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/listwise_whole_model_v11.json")
    parser.add_argument("--cache", default="artifacts/score_cache_v11.pkl")
    parser.add_argument("--report", default="artifacts/score_compare_v11.json")
    args = parser.parse_args(argv)

    params = L.weights()
    payload = json.loads((ROOT / args.model).read_text(encoding="utf-8"))
    cols = payload["columns"]
    _train, val, mfava = L.load_all()
    if Path(args.cache).exists():
        with open(args.cache, "rb") as handle:
            cache = pickle.load(handle)
    else:
        cache = build(val + mfava, params, payload, cols)
        with open(args.cache, "wb") as handle:
            pickle.dump(cache, handle)

    rows = []
    for name in ("base", "local", "support", "listwise", "base_local", "base_support", "base_listwise"):
        score = scorer(name)
        for gate_cut in (0.3, 0.5, 0.6):
            for max_spans in (1, 2, 3):
                rows.append({
                    "scorer": name, "gate": gate_cut, "max_spans": max_spans,
                    "val": L.cell(evaluate(cache, val, score, gate_cut, max_spans)),
                    "mfava": L.cell(evaluate(cache, mfava, score, gate_cut, max_spans)),
                    "combined": L.cell(evaluate(cache, val + mfava, score, gate_cut, max_spans)),
                    "val_top1": top1_accuracy(cache, val, score),
                })
    rows.sort(key=lambda row: (row["combined"]["span"], row["val"]["span"]), reverse=True)
    passing = [row for row in rows if L.guard([row["val"], row["combined"]])]
    report = {
        "rule": "scorer-сравнение на val+mFAVA; модели обучены на train-only; held не читается",
        "best": rows[:12], "passing": passing[:10], "configs": rows,
        "per_scorer_best": {
            name: max((row for row in rows if row["scorer"] == name),
                      key=lambda row: row["combined"]["span"], default=None)
            for name in sorted({row["scorer"] for row in rows})
        },
    }
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, row in report["per_scorer_best"].items():
        if row:
            print(name, "val", row["val"]["span"], row["val"]["token"], row["val"]["pair"],
                  "@", row["val"]["fpr"], "mfava", row["mfava"]["span"],
                  "top1", {k: v["acc"] for k, v in row["val_top1"].items()}, flush=True)
    print("guard", len(passing), flush=True)


if __name__ == "__main__":
    main()
