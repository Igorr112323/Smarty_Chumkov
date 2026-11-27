"""V11: dual decode — одновременно гипотеза whole (base) и clause (local).

Мотив: на val base находит whole (20/23 top-1), а local HGB — clause (9/18).
Вместо router'а (который по правилу плана отклонён: >15% ложных whole) можно
выдавать обе гипотезы, если каждая проходит свой порог уверенности. Цена —
precision, выигрыш — recall. Пороги подбираются на val+mFAVA, held не читается.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import v11_lab as L
from spanver.evaluate import iou, metrics


def pick(entry, config):
    ctx = entry["ctx"]
    rows = entry["rows"]
    answer = ctx["answer"]
    if ctx["gate"] < config["gate"] or not rows:
        return []
    whole = [row for row in rows if row["start"] == 0 and row["end"] == len(answer) - 1]
    clause = [row for row in rows if not (row["start"] == 0 and row["end"] == len(answer) - 1)]
    out = []
    if config["use_base"] and rows:
        top = max(rows, key=lambda row: (row["base"], -row["start"]))
        if top["base"] >= config["base_cut"]:
            out.append({"start": top["start"], "end": top["end"], "risk": round(top["base"], 4),
                        "type": "unsupported_addition", "confidence": 0.5})
    if config["use_local"] and clause:
        ranked = sorted(clause, key=lambda row: (-row["local"], row["start"]))
        taken = 0
        for row in ranked:
            if row["local"] < config["local_cut"]:
                break
            if any(iou(row["start"], row["end"], other["start"], other["end"]) > config["nms"]
                   for other in out):
                continue
            out.append({"start": row["start"], "end": row["end"], "risk": round(row["local"], 4),
                        "type": "unsupported_addition", "confidence": 0.5})
            taken += 1
            if taken >= config["max_clauses"]:
                break
    if config["whole_first"] and not any(item["start"] == 0 and item["end"] == len(answer) - 1
                                         for item in out) and whole:
        top = max(whole, key=lambda row: row["base"])
        if top["base"] >= config["whole_cut"]:
            out.append({"start": 0, "end": len(answer) - 1, "risk": round(top["base"], 4),
                        "type": "unsupported_addition", "confidence": 0.5})
    if config["max_spans"] and len(out) > config["max_spans"]:
        out = out[: config["max_spans"]]
    return out


def evaluate(cache, pairs, config):
    picked = {pair["pair_id"]: pick(cache[pair["pair_id"]], config) for pair in pairs}
    return metrics(pairs, lambda pair: picked[pair["pair_id"]])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="artifacts/score_cache_v11.pkl")
    parser.add_argument("--report", default="artifacts/dual_decode_v11.json")
    args = parser.parse_args(argv)

    _train, val, mfava = L.load_all()
    with open(ROOT / args.cache, "rb") as handle:
        cache = pickle.load(handle)

    rows = []
    for gate_cut in (0.3, 0.5, 0.6):
        for base_cut in (0.4, 0.6, 0.8, -1.0):
            for local_cut in (0.3, 0.5, 0.7, 0.9, 1.1):
                for max_clauses in (1, 2):
                    for max_spans in (2, 3):
                        config = {"gate": gate_cut, "use_base": True, "base_cut": base_cut,
                                  "use_local": True, "local_cut": local_cut,
                                  "max_clauses": max_clauses, "max_spans": max_spans,
                                  "whole_first": False, "whole_cut": 0.99, "nms": 0.1}
                        rows.append({"config": config,
                                     "val": L.cell(evaluate(cache, val, config)),
                                     "mfava": L.cell(evaluate(cache, mfava, config)),
                                     "combined": L.cell(evaluate(cache, val + mfava, config))})
    rows.sort(key=lambda row: (row["combined"]["span"], row["val"]["span"]), reverse=True)
    passing = [row for row in rows if L.guard([row["val"], row["combined"]])]
    report = {"rule": "dual decode base+local; пороги val+mFAVA; held не читается",
              "best": rows[:15], "passing": passing[:10], "configs": rows}
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("best", json.dumps(rows[0], ensure_ascii=False), flush=True)
    print("guard", len(passing), "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
