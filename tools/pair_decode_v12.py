"""V12 шаг 3: pair-гейт поверх штатного декада и локальный fallback (только val).

Разбор ошибок (шаг 1) показал: 27 FN = 12 «гейт молчит» + 15 span-промахов, 13 FP —
почти все global-whole. Поэтому здесь пара и спан развязаны:

  * спаны всегда считает штатный ranker.select_spans (hybrid mode decode v11),
    pair-решение их не переписывает — только разрешает/запрещает вывод;
  * pair-вероятность даёт отдельный pair-GBDT (tools/train_pair_classifier_v12.py),
    порог калибруется на val;
  * если runtime молчит или выдал только whole, а pair-гейт пропустил — допускается
    одна локальная гипотеза (local_hgb), т.е. ровно те пары, что закрывал v3-гейт.

Ограничения: FPR <= 0.40, span >= 0.3163, token >= 0.50, mFAVA span >= 0.65.
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
from train_pair_classifier_v12 import pair_vector, tree_probability
from spanver.evaluate import iou, metrics
from spanver.local_ranker import predict_trees
from spanver import ranker


def local_hypotheses(pair, params):
    """Локальные гипотезы (local_hgb) без whole-спана — как в local-режиме ranker."""
    from spanver.module import candidate_spans, document_context
    from spanver.ranker import features

    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    if not answer:
        return []
    context = document_context(document)
    scored = []
    for start, end, fragment in candidate_spans(answer):
        if start == 0 and end == len(answer) - 1:
            continue
        vector = features(fragment, start, end, answer, document, context[0], context[2])
        scored.append({"start": start, "end": end,
                       "score": predict_trees(params["local_hgb"], vector)})
    scored.sort(key=lambda item: (-item["score"], item["start"]))
    return scored[:3]


def build_cache(pairs, params, cache, model, config, adjuster):
    """Кэш по паре: GBDT-вероятность пары + спаны базового декодера v11.

    Базовые спаны = B.adjust(hybrid) с v3-гейтом, т.е. ровно то, что даёт baseline
    (span 0.3363 / token 0.6191 / pair 0.4872). pair-гейт их не переписывает: он
    либо разрешает вывод, либо запрещает. Дополнительно считается одна локальная
    гипотеза (local_hgb, без whole) — она нужна только как fallback там, где
    базовый декодер молчит, либо как clause-дополнение к whole-ответу.
    """
    rows = {}
    for pair in pairs:
        base_rows, gate = cache.get(pair)
        probability = tree_probability(model, pair_vector(pair, params, cache))
        base = B.adjust(pair, cache, config, adjuster[0], adjuster[1], 0.5, "hybrid")
        whole_only = (len(base) == 1 and base[0]["start"] == 0
                      and base[0]["end"] == len(pair.get("model_output") or "") - 1)
        local = []
        if not base or whole_only:
            local = local_hypotheses(pair, params)
        rows[pair["pair_id"]] = {
            "probability": probability, "gate_v3": gate,
            "base": base, "whole_only": whole_only, "local": local,
        }
    return rows


def compose(entry, config):
    """pair-гейт поверх базового декода: fallback при молчащем базе, clause к whole."""
    if entry["probability"] < config["cut"]:
        return []
    spans = [dict(span) for span in entry["base"]]
    local = entry["local"]
    if not spans:
        if config.get("fallback") and entry["probability"] >= config.get("fb_prob", 0.0):
            for item in local:
                if item["score"] < config["fb_cut"]:
                    break
                spans.append({"start": item["start"], "end": item["end"],
                              "risk": round(float(item["score"]), 4),
                              "type": "unsupported_addition", "confidence": 0.5})
                break
        return spans
    if config.get("add_clause") and entry["whole_only"]:
        for item in local:
            if item["score"] < config["add_cut"]:
                break
            spans.append({"start": item["start"], "end": item["end"],
                          "risk": round(float(item["score"]), 4),
                          "type": "unsupported_addition", "confidence": 0.5})
            break
    return spans


def evaluate(rows, pairs, config):
    picked = {pair["pair_id"]: compose(rows[pair["pair_id"]], config) for pair in pairs}
    return metrics(pairs, lambda pair: picked[pair["pair_id"]]), picked


def confusion(pairs, picked):
    tp = fp = fn = 0
    for pair in pairs:
        gold = pair.get("spans") or []
        predicted = picked[pair["pair_id"]]
        matched = any(iou(span["start"], span["end"], truth["start"], truth["end"]) >= 0.5
                      for span in predicted for truth in gold)
        if gold:
            tp += int(matched)
            fn += int(not matched)
        elif predicted:
            fp += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/pair_sweep_v12b.json")
    parser.add_argument("--chosen", default="artifacts/pair_chosen_v12.json")
    parser.add_argument("--model", default="artifacts/pair_classifier_model_v12.json")
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    cache = L.Cache(params=params)
    train, val, mfava = L.load_all()
    config = dict(params["config"])
    model = json.loads((ROOT / args.model).read_text(encoding="utf-8"))
    adjuster = B.train(B.collect(train, cache, config, "hybrid"))
    rows = build_cache(val + mfava, params, cache, model, config, adjuster)

    sweep = []
    grid = [(False, 1.1, 0.0, False, 1.1)] \
        + [(True, fb_cut, fb_prob, False, 1.1)
           for fb_prob in (0.0, 0.4, 0.6, 0.7) for fb_cut in (0.6, 0.8)] \
        + [(True, 0.8, 0.7, True, add_cut) for add_cut in (0.6, 0.75, 0.9)]
    for cut in (0.0, 0.10, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40):
        for fallback, fb_cut, fb_prob, add_clause, add_cut in grid:
            cfg = {"cut": cut, "fallback": fallback, "fb_cut": fb_cut,
                   "fb_prob": fb_prob, "add_clause": add_clause, "add_cut": add_cut}
            val_metrics, val_picked = evaluate(rows, val, cfg)
            mfava_metrics, _ = evaluate(rows, mfava, cfg)
            sweep.append({
                "cut": cut, "fallback": fallback, "fb_cut": fb_cut, "fb_prob": fb_prob,
                "add_clause": add_clause, "add_cut": add_cut,
                "pair": confusion(val, val_picked),
                "val": L.cell(val_metrics), "mfava": L.cell(mfava_metrics),
            })
            row = sweep[-1]
            print("cut", cut, "fb", fallback, fb_prob, fb_cut, "add", add_clause, add_cut,
                  "pair", row["pair"], "val", row["val"], "mfava", row["mfava"]["span"],
                  flush=True)

    ok = [row for row in sweep
          if row["pair"]["f1"] >= 0.5 and row["val"]["token"] >= 0.5
          and row["val"]["fpr"] <= 0.4 and row["val"]["span"] >= 0.3163
          and row["mfava"]["span"] >= 0.65]
    ok.sort(key=lambda row: (row["pair"]["f1"], row["val"]["span"]), reverse=True)
    band = [row for row in sweep if 0.30 <= row["val"]["fpr"] <= 0.40]
    band.sort(key=lambda row: row["pair"]["f1"], reverse=True)
    report = json.loads((ROOT / args.report).read_text(encoding="utf-8")) \
        if (ROOT / args.report).exists() else {}
    report["v12d"] = {
        "rule": "pair-GBDT гейт поверх базового декодера v11 (B.adjust hybrid); fallback на local_hgb "
                "только если базовый декодер молчит; калибровка на val; held не читается",
        "constraints": {"pair_f1": 0.5, "token": 0.5, "fpr_max": 0.4, "span_min": 0.3163,
                        "mfava_span_min": 0.65},
        "passing": ok[:15], "fpr_band_030_040": band[:10], "sweep": sweep,
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    chosen = ok[0] if ok else None
    if chosen is not None:
        cfg = {"cut": chosen["cut"], "fallback": chosen["fallback"],
               "fb_cut": chosen["fb_cut"], "fb_prob": chosen["fb_prob"],
               "add_clause": chosen["add_clause"], "add_cut": chosen["add_cut"]}
        val_metrics, val_picked = evaluate(rows, val, cfg)
        mfava_metrics, _ = evaluate(rows, mfava, cfg)
        base_cfg = {"cut": 0.0, "fallback": False, "add_clause": False}
        base_metrics, base_picked = evaluate(rows, val, base_cfg)
        errors = {"fn": [], "fp": []}
        for pair in val:
            gold = pair.get("spans") or []
            predicted = val_picked[pair["pair_id"]]
            matched = any(iou(span["start"], span["end"], truth["start"], truth["end"]) >= 0.5
                          for span in predicted for truth in gold)
            entry = rows[pair["pair_id"]]
            record = {
                "pair_id": pair["pair_id"],
                "probability": round(entry["probability"], 4),
                "gate_v3": round(float(entry["gate_v3"]), 4),
                "whole_only": entry["whole_only"],
                "base_spans": len(entry["base"]),
                "local_top1": round(entry["local"][0]["score"], 4) if entry["local"] else None,
                "predicted": [[span["start"], span["end"]] for span in predicted],
                "gold": [[truth["start"], truth["end"]] for truth in gold],
            }
            if gold and not matched:
                errors["fn"].append(record)
            elif not gold and predicted:
                errors["fp"].append(record)
        payload = {
            "rule": "лучший по pair F1 валидный конфиг val-калибровки (все констрейнты v12)",
            "config": cfg,
            "val": L.cell(val_metrics),
            "mfava": L.cell(mfava_metrics),
            "pair": confusion(val, val_picked),
            "baseline_v11": {"val": L.cell(base_metrics),
                             "pair": confusion(val, base_picked)},
            "errors": errors,
        }
        (ROOT / args.chosen).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        print("chosen", json.dumps(cfg, ensure_ascii=False),
              "pair", payload["pair"], "val", payload["val"],
              "mfava_span", payload["mfava"]["span"], flush=True)
    print("passing", len(ok), flush=True)


if __name__ == "__main__":
    main()
