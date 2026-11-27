"""V12 шаг 4: 3-fold OOF по документам (RusHallu train + mFAVA, только реальные данные).

Проверяет, что выигрыш pair-гейта не перефит: pair-GBDT обучается на 2/3 документов,
а оценивается на отложенной трети. Все пороги (cut, fb_prob, fb_cut, add_cut) взяты
из калибровки на val и внутри фолдов НЕ подбираются.

Базовый декодер (v11 hybrid + boundary adjuster) остаётся общим для всех фолдов —
OOF проверяет именно pair-решение, а не span-слой.

Held-out не читается.
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
import pair_decode_v12 as P
import v11_lab as L
from spanver.evaluate import iou, metrics
from train_pair_classifier_v12 import export_trees, pair_vector, tree_probability

CONFIG = {"cut": 0.25, "fallback": True, "fb_prob": 0.7, "fb_cut": 0.75,
          "add_clause": True, "add_cut": 0.75}

ABLATION = {
    "base": {"cut": 0.0, "fallback": False, "add_clause": False},
    "gate_0.25": {"cut": 0.25, "fallback": False, "add_clause": False},
    "gate_0.16": {"cut": 0.16, "fallback": False, "add_clause": False},
    "gate_fallback": {"cut": 0.25, "fallback": True, "fb_prob": 0.7, "fb_cut": 0.75,
                      "add_clause": False},
    "gate_fallback_clause": CONFIG,
}


def fold_of(pair, folds=3):
    digest = hashlib.sha256((pair.get("document_text") or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def fit_model(pairs, vectors, trees, leaves):
    X = np.array([vectors[pair["pair_id"]] for pair in pairs], dtype=np.float64)
    y = np.array([int(bool(pair.get("spans"))) for pair in pairs], dtype=int)
    weights = np.where(y == 1, 1.0, 1.6)
    classifier = HistGradientBoostingClassifier(
        max_iter=trees, learning_rate=0.06, max_leaf_nodes=leaves,
        min_samples_leaf=8, l2_regularization=1.0, random_state=12,
    ).fit(X, y, sample_weight=weights)
    return export_trees(classifier)


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
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/pair_oof_v12.json")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--leaves", type=int, default=15)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    cache = L.Cache(params=params)
    train, val, mfava = L.load_all()
    config = dict(params["config"])
    adjuster = B.train(B.collect(train, cache, config, "hybrid"))
    full_model = json.loads((ROOT / "artifacts/pair_classifier_model_v12.json")
                            .read_text(encoding="utf-8"))

    # строки декодера (base/whole_only/local) от модели не зависят
    rows = P.build_cache(train + mfava, params, cache, full_model, config, adjuster)
    vectors = {pair["pair_id"]: pair_vector(pair, params, cache) for pair in train + mfava}
    for pair in train + mfava:
        rows[pair["pair_id"]]["probability_full"] = tree_probability(
            full_model, vectors[pair["pair_id"]])

    per_fold = []
    for fold in range(args.folds):
        held = [pair for pair in train + mfava if fold_of(pair, args.folds) == fold]
        rest = [pair for pair in train + mfava if fold_of(pair, args.folds) != fold]
        model = fit_model(rest, vectors, args.trees, args.leaves)
        for pair in held:
            rows[pair["pair_id"]]["probability"] = tree_probability(
                model, vectors[pair["pair_id"]])
        held_val = [pair for pair in held if pair.get("split") == "train"]
        held_mfava = [pair for pair in held if pair.get("split") == "train_extra"]
        labels = [int(bool(pair.get("spans"))) for pair in held_val]
        scores = [rows[pair["pair_id"]]["probability"] for pair in held_val]
        auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else None
        variants = {}
        for name, cfg in ABLATION.items():
            picked = {pair["pair_id"]: P.compose(rows[pair["pair_id"]], cfg)
                      for pair in held}
            variants[name] = {
                "pair": pair_counts(held_val, picked),
                "real": L.cell(metrics(held_val, lambda pair: picked[pair["pair_id"]])),
                "mfava": L.cell(metrics(held_mfava, lambda pair: picked[pair["pair_id"]]))
                if held_mfava else None,
            }
        picked = {pair["pair_id"]: P.compose(rows[pair["pair_id"]], CONFIG) for pair in held}
        entry = {
            "fold": fold,
            "held_pairs": len(held),
            "held_real": len(held_val),
            "held_mfava": len(held_mfava),
            "pair": pair_counts(held_val, picked),
            "real": L.cell(metrics(held_val, lambda pair: picked[pair["pair_id"]])),
            "mfava": L.cell(metrics(held_mfava, lambda pair: picked[pair["pair_id"]]))
            if held_mfava else None,
            "auc": None if auc is None else round(float(auc), 4),
            "variants": variants,
            "base_pair": variants["base"]["pair"],
            "base_real": variants["base"]["real"],
            "base_mfava": variants["base"]["mfava"],
        }
        per_fold.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    def mean(key, sub=None):
        values = []
        for entry in per_fold:
            value = entry[key] if sub is None else (entry[key] or {}).get(sub)
            if value is not None:
                values.append(value)
        return round(float(np.mean(values)), 4) if values else None

    report = {
        "rule": "3-fold OOF по документам (sha256 документа % folds); pair-GBDT только на "
                "2/3 фолдов; пороги взяты из калибровки на val и не подбирались; "
                "базовый декодер v11 общий",
        "config": CONFIG,
        "folds": per_fold,
        "mean": {
            "pair_f1": mean("pair", "f1"),
            "pair_precision": mean("pair", "precision"),
            "pair_recall": mean("pair", "recall"),
            "span": mean("real", "span"),
            "token": mean("real", "token"),
            "fpr": mean("real", "fpr"),
            "mfava_span": mean("mfava", "span"),
            "auc": mean("auc"),
            "base_pair_f1": mean("base_pair", "f1"),
            "base_span": mean("base_real", "span"),
            "base_token": mean("base_real", "token"),
            "base_fpr": mean("base_real", "fpr"),
            "base_mfava_span": mean("base_mfava", "span"),
        },
        "ablation_mean": {
            name: {
                "pair_f1": round(float(np.mean([entry["variants"][name]["pair"]["f1"]
                                                for entry in per_fold])), 4),
                "span": round(float(np.mean([entry["variants"][name]["real"]["span"]
                                             for entry in per_fold])), 4),
                "token": round(float(np.mean([entry["variants"][name]["real"]["token"]
                                              for entry in per_fold])), 4),
                "fpr": round(float(np.mean([entry["variants"][name]["real"]["fpr"]
                                            for entry in per_fold])), 4),
                "mfava_span": round(float(np.mean([entry["variants"][name]["mfava"]["span"]
                                                   for entry in per_fold])), 4),
            }
            for name in ABLATION
        },
        "min_pair_f1": min(entry["pair"]["f1"] for entry in per_fold),
        "delta_vs_base": {
            "pair_f1": round((mean("pair", "f1") or 0) - (mean("base_pair", "f1") or 0), 4),
            "span": round((mean("real", "span") or 0) - (mean("base_real", "span") or 0), 4),
            "fpr": round((mean("real", "fpr") or 0) - (mean("base_real", "fpr") or 0), 4),
            "mfava_span": round((mean("mfava", "span") or 0)
                                - (mean("base_mfava", "span") or 0), 4),
        },
        "constraints": {"pair_f1": 0.5, "span_min": 0.3163, "token_min": 0.5,
                        "fpr_max": 0.4, "mfava_span_min": 0.65},
    }
    report["pass"] = bool(
        report["mean"]["pair_f1"] >= 0.5 and report["min_pair_f1"] >= 0.45
        and report["mean"]["span"] >= 0.3163 and report["mean"]["token"] >= 0.5
        and report["mean"]["fpr"] <= 0.4 and report["mean"]["mfava_span"] >= 0.65)
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report["mean"], ensure_ascii=False),
          "min_pair_f1", report["min_pair_f1"], "pass", report["pass"], flush=True)


if __name__ == "__main__":
    main()
