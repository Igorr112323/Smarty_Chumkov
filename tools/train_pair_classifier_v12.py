"""V12 шаг 2–3: отдельный pair-классификатор «есть ли ошибка в паре».

Развязывает pair-решение и span-выбор: раньше гейт v3 (и forced 1.0 в local-режиме)
одновременно решал, будет ли пара помечена, и влиял на порог спанов. Теперь:

  pair-решение  — GBDT на inference-признаках обеих голов (train-only);
  span-выбор    — прежний v11 hybrid (mode gate + boundary adjuster), без гейта.

Экспорт деревьев в чистый JSON (как v8/v9), рантайм без sklearn.
Порог калибруется на val (FPR 0.30–0.35 — целевая зона), held-out не читается.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import boundary_adjust_v11 as B
import v11_lab as L
from pair_analysis_v12 import candidate_stats
from spanver import ranker
from spanver.evaluate import iou, metrics
from spanver.local_ranker import logistic, predict_trees

FEATURES = (
    "gate", "mode_probability", "base_top1", "base_top2", "base_gap", "base_top5_mean",
    "base_mean", "log_n_cand", "n_above_03", "n_above_05", "n_above_06",
    "local_top1", "local_top2_mean", "local_above_05", "local_above_08",
    "log_answer_len", "n_sentences", "n_clauses", "answer_cov", "answer_cos",
    "digit_share", "capital_share", "numbers_missing",
)


def pair_vector(pair, params, cache):
    stats = candidate_stats(pair, params, cache)
    rows, _ = cache.get(pair)
    local = [predict_trees(params["local_hgb"], row["features"]) for row in rows]
    values = dict(stats)
    values["log_n_cand"] = np.log1p(stats["n_cand"])
    values["local_above_05"] = sum(1 for value in local if value > 0.5)
    values["local_above_08"] = sum(1 for value in local if value > 0.8)
    values["log_answer_len"] = np.log1p(stats["answer_len"])
    return [float(values[name]) for name in FEATURES]


def export_trees(classifier):
    trees = []
    for iteration in classifier._predictors:
        nodes = []
        for row in iteration[0].nodes:
            nodes.append({
                "leaf": bool(row["is_leaf"]), "f": int(row["feature_idx"]),
                "t": float(row["num_threshold"]), "l": int(row["left"]),
                "r": int(row["right"]), "v": float(row["value"]),
            })
        trees.append(nodes)
    return {"baseline": float(classifier._baseline_prediction[0, 0]), "trees": trees,
            "n_features": len(FEATURES), "feature_names": list(FEATURES)}


def tree_probability(model, values):
    total = model["baseline"]
    for tree in model["trees"]:
        index = 0
        while True:
            node = tree[index]
            if node["leaf"]:
                total += node["v"]
                break
            index = node["l"] if values[node["f"]] <= node["t"] else node["r"]
    return 1 / (1 + np.exp(-max(-30.0, min(30.0, total))))


def pick_spans(pair, cache, config, params):
    """Span-выбор v11 без pair-гейта: путь решает mode gate, порог — из конфига."""
    rows, _ = cache.get(pair)
    answer = pair.get("model_output") or ""
    if not rows or not answer:
        return []
    local_mode = B.is_local_mode(pair, params)
    if local_mode:
        local_config = params["local_config"]
        candidates = [
            {"start": row["start"], "end": row["end"], "text": row["text"],
             "risk": predict_trees(params["local_hgb"], row["features"])}
            for row in rows if not (row["start"] == 0 and row["end"] == len(answer) - 1)
        ]
        threshold = local_config.get("threshold", 0.5)
        max_spans = local_config.get("max_spans", 1)
        nms = local_config.get("nms_iou", 0.1)
    else:
        candidates = [{"start": row["start"], "end": row["end"], "text": row["text"],
                       "risk": row["risk"]} for row in rows]
        threshold = config["threshold"]
        max_spans = config["max_spans"]
        nms = config["nms_iou"]
    chosen = []
    for row in sorted(candidates, key=lambda item: (-item["risk"], item["start"])):
        if row["risk"] < threshold:
            break
        if any(iou(row["start"], row["end"], other["start"], other["end"]) > nms for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= max_spans:
            break
    return chosen


def decode(pair, cache, config, params, probability, cut, adjuster):
    if probability < cut:
        return []
    chosen = pick_spans(pair, cache, config, params)
    if not chosen:
        return []
    if adjuster is not None and not B.is_local_mode(pair, params):
        model, scaler = adjuster
        return B.adjust(pair, cache, config, model, scaler, 0.5, "global")
    return [{"start": row["start"], "end": row["end"], "risk": row["risk"] if "risk" in row else 0.5,
             "type": "unsupported_addition", "confidence": 0.5} for row in chosen]


def pair_counts(pairs, picked):
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
    return tp, fp, fn


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/pair_classifier_v12.json")
    parser.add_argument("--model", default="artifacts/pair_classifier_model_v12.json")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--leaves", type=int, default=15)
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    cache = L.Cache(params=params)
    train, val, mfava = L.load_all()
    config = dict(params["config"])

    # boundary adjuster из v11 (global-путь)
    adjuster = B.train(B.collect(train, cache, config, "hybrid"))

    X = np.array([pair_vector(pair, params, cache) for pair in train + mfava], dtype=np.float64)
    y = np.array([int(bool(pair.get("spans"))) for pair in train + mfava], dtype=int)
    weights = np.where(y == 1, 1.0, 1.6)
    classifier = HistGradientBoostingClassifier(
        max_iter=args.trees, learning_rate=0.06, max_leaf_nodes=args.leaves,
        min_samples_leaf=8, l2_regularization=1.0, random_state=12,
    ).fit(X, y, sample_weight=weights)
    model = export_trees(classifier)
    sample = X[:: max(1, len(X) // 100)]
    expected = classifier.predict_proba(sample)[:, 1]
    actual = np.array([tree_probability(model, list(row)) for row in sample])
    delta = float(np.max(np.abs(expected - actual))) if len(expected) else 0.0

    probabilities = {}
    for pair in val + mfava:
        probabilities[pair["pair_id"]] = tree_probability(
            model, pair_vector(pair, params, cache))

    val_auc = roc_auc_score([int(bool(p.get("spans"))) for p in val],
                            [probabilities[p["pair_id"]] for p in val])
    sweep = []
    for cut in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        picked = {pair["pair_id"]: decode(pair, cache, config, params,
                                          probabilities[pair["pair_id"]], cut, adjuster)
                  for pair in val + mfava}
        val_metrics = metrics(val, lambda pair: picked[pair["pair_id"]])
        mfava_metrics = metrics(mfava, lambda pair: picked[pair["pair_id"]])
        tp, fp, fn = pair_counts(val, picked)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        negatives = sum(1 for pair in val if not pair.get("spans"))
        sweep.append({
            "cut": cut,
            "pair": {"tp": tp, "fp": fp, "fn": fn,
                     "precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(f1, 4)},
            "val": L.cell(val_metrics),
            "mfava": L.cell(mfava_metrics),
            "negatives_flagged": round(val_metrics["fpr"] * negatives),
            "negatives": negatives,
        })
    passing = [row for row in sweep
               if row["val"]["pair"] >= 0.5 and row["val"]["token"] >= 0.5
               and row["val"]["fpr"] <= 0.4 and row["mfava"]["span"] >= 0.65]

    model_path = ROOT / args.model
    model_path.write_text(json.dumps(model), encoding="utf-8")
    report = {
        "rule": "pair-GBDT на inference-признаках обеих голов; span-выбор отдельно (v11 hybrid без гейта)",
        "features": list(FEATURES), "train": {"pairs": len(y), "positives": int(y.sum())},
        "export_max_abs_delta": round(delta, 10),
        "val_pair_auc": round(float(val_auc), 4),
        "sweep": sweep, "passing": passing,
        "model": args.model,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("auc", report["val_pair_auc"], "export_delta", delta, flush=True)
    for row in sweep:
        print("cut", row["cut"], "pair_f1", row["pair"]["f1"], "P", row["pair"]["precision"],
              "R", row["pair"]["recall"], "span", row["val"]["span"], "token", row["val"]["token"],
              "fpr", row["val"]["fpr"], "mfava_span", row["mfava"]["span"], flush=True)
    print("passing", len(passing), [row["cut"] for row in passing], flush=True)


if __name__ == "__main__":
    main()
