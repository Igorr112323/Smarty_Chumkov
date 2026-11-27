"""V16 шаг 5: честная калибровка существующего декодера v11 в бюджете FPR ≤ 0.40.

Идея: признаки кандидата и все скоры v11 (cand logit, mode-вероятность, pair-гейт, local_hgb)
считаются ОДИН раз на пару, дальше любой набор порогов симулируется офлайн — это позволяет
перебрать тысячи конфигураций без повторного фичеринга.

Гейт режима (`mode`), pair-гейт и local_hgb **не сбрасываются** — крутим только калибровочные
пороги (`mode_cut`, `gate`, `threshold`, `max_spans`, `local_config`), что и предписывает шаг 5.

Таблицы сохраняются в `artifacts/cache_v16/*_tables.json`, чтобы LM-скоры v15 можно было
примешать к `risk_cand` без пересчёта 37 признаков.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import v11_lab as L
from spanver import ranker
from spanver.evaluate import metrics
from spanver.local_ranker import logistic, predict_trees
from spanver.module import candidate_spans, confidence, document_context, predict_type


def pair_table(pair, weights):
    """Все скоры v11 по кандидатам пары (без применения порогов)."""
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    doc_tokens, doc_grams = document_context(document)[0], document_context(document)[2]
    rows = []
    for start, end, fragment in candidate_spans(answer):
        vector = ranker.features(fragment, start, end, answer, document, doc_tokens, doc_grams)
        rows.append({"start": start, "end": end, "text": fragment, "features": vector})
    if not rows:
        return {"pair_id": pair["pair_id"], "rows": [], "mode_prob": 0.0, "pair_prob": 0.0,
                "whole_skip": False}
    for row in rows:
        row["risk_cand"] = ranker._logit(row["features"], weights["cand"], ranker.FEATURE_COUNT)
        row["risk_local"] = predict_trees(weights["local_hgb"], row["features"])
        row["risk"] = row["risk_cand"]
    mode_prob = logistic(ranker.mode_features(rows, answer, document), weights["mode"],
                         ranker.MODE_FEATURE_COUNT)
    pair_prob = ranker._logit(ranker.pair_features([row["features"] for row in rows],
                                                   [row["risk_cand"] for row in rows], answer),
                              weights["pair"], ranker.PAIR_FEATURE_COUNT)
    return {"pair_id": pair["pair_id"], "rows": rows, "mode_prob": mode_prob,
            "pair_prob": pair_prob}


def build_tables(pairs, weights, path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    tables = []
    for index, pair in enumerate(pairs):
        tables.append(pair_table(pair, weights))
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(pairs)} пар", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")
    return tables


def decode(table, config, risk_key="risk_cand"):
    """Симуляция `ranker.select_spans` с заданными порогами."""
    rows = table["rows"]
    if not rows:
        return []
    answer_len = max(row["end"] for row in rows)  # не используется, оставлено для ясности
    del answer_len
    local = table["mode_prob"] >= config["mode_cut"]
    if local:
        threshold = config["local_threshold"]
        max_spans = config["local_max_spans"]
        pool = [row for row in rows if not (row["start"] == 0 and row["end"] == rows[0]["end"])]
        key = "risk_local"
        gated = True
    else:
        if table["pair_prob"] < config["gate"]:
            return []
        threshold = config["threshold"]
        max_spans = config["max_spans"]
        pool = rows
        key = risk_key
        gated = True
    del gated
    order = sorted(range(len(pool)), key=lambda index: (-pool[index][key], pool[index]["start"]))
    accepted = []
    for index in order:
        if pool[index][key] < threshold:
            break
        candidate = pool[index]
        if any(ranker.iou(candidate["start"], candidate["end"], other["start"], other["end"])
               > config["nms_iou"] for other in accepted):
            continue
        accepted.append(candidate)
        if len(accepted) >= max_spans:
            break
    return [{"start": row["start"], "end": row["end"], "risk": round(row[key], 4),
             "type": predict_type(row["text"], 0.0), "confidence": confidence(row[key])}
            for row in accepted]


def evaluate(pairs, tables, config, risk_key="risk_cand"):
    picked = {table["pair_id"]: decode(table, config, risk_key)
              for table in tables}
    return metrics(pairs, lambda pair: picked[pair["pair_id"]])


def grid():
    for mode_cut, gate, threshold, max_spans, local_threshold, local_max_spans in itertools.product(
            (0.3, 0.5, 0.6), (0.2, 0.3, 0.4, 0.5), (0.2, 0.3, 0.4, 0.5, 0.6),
            (1, 2, 3, 5), (0.2, 0.3, 0.4, 0.5), (1, 3, 5)):
        yield {"mode_cut": mode_cut, "gate": gate, "threshold": threshold,
               "max_spans": max_spans, "local_threshold": local_threshold,
               "local_max_spans": local_max_spans, "nms_iou": 0.1}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/calibration_v16.json")
    parser.add_argument("--risk-key", default="risk_cand")
    parser.add_argument("--fpr-max", type=float, default=0.40)
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    train, val, mfava = L.load_all()
    cache = ROOT / "artifacts/cache_v16"
    tables = {name: build_tables(pairs, params, cache / f"{name}_tables.json")
              for name, pairs in (("val", val), ("mfava", mfava))}
    print("таблицы:", {name: len(rows) for name, rows in tables.items()}, flush=True)

    rows = []
    for config in grid():
        result = evaluate(val, tables["val"], config, args.risk_key)
        cell = L.cell(result)
        rows.append({**cell, "config": config})
    rows.sort(key=lambda row: (-row["span"], row["fpr"]))

    feasible = [row for row in rows if row["fpr"] <= args.fpr_max]
    best_span = feasible[0] if feasible else None
    best_pair = max(feasible, key=lambda row: row["pair"]) if feasible else None
    report = {
        "rule": "калибровка только порогов существующего v11-декодера (mode-гейт, pair-гейт и "
                "local_hgb сохранены), бюджет FPR ≤ 0.40, выбор по val",
        "baseline_v11": {"span": 0.3363, "token": 0.6191, "pair": 0.4872, "fpr": 0.125},
        "configs_tested": len(rows),
        "best_span": best_span,
        "best_pair": best_pair,
        "top": rows[:10],
        "top_fpr": sorted(feasible, key=lambda row: row["fpr"])[:5] if feasible else [],
    }
    if best_span:
        mfava_cell = L.cell(evaluate(mfava, tables["mfava"], best_span["config"], args.risk_key))
        report["mfava_at_best_span"] = mfava_cell
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best_span", {k: round(best_span[k], 4) for k in ("span", "token", "pair", "fpr")},
          best_span["config"], flush=True)
    print("best_pair", {k: round(best_pair[k], 4) for k in ("span", "token", "pair", "fpr")},
          best_pair["config"], flush=True)


if __name__ == "__main__":
    main()
