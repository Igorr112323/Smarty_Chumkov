"""V14 шаг 3–4 (сфокусированный): дообучение локального span-каскада без сброса v11.

Что НЕ меняется: признаки (37 штатных), mode-гейт v3, boundary adjuster, пороги из
`ranker_weights_v3.json` (`config`/`local_config`). Меняется только `local_hgb` —
HistGB на кандидатах local-режима, по рецепту `tools/train_local_ranker.py`
(lr 0.1, depth 4, 31 лист, 150 итераций, random_state 0), но с добавкой данных:

  A real                       — gold_v3 train + mfava_train_v3 (контроль рецепта v3);
  B real+synth                 — + rushallu_synth_v11 (3 563 пары, точные границы);
  C real+synth+pseudo          — + clause_pseudo_v11 (84 пары, weak supervision);
  D real+synth+pseudo+ragtruth — + ragtruth_proj_v14 (958 пар, en-проекция стиля).

В local-каскад идут пары, у которых gold не whole и есть clause-спан
(len(spans) > 1 ИЛИ доля спана < 0.8 ответа): в v3 брали только multi-span, из-за
чего одиночные clause-пары (а это и есть наш тип ошибки whole→clause) в обучение
не попадали.

Оценка — полный hybrid-декод v11 (adjuster включён), затем val + mFAVA.
OOF: 3 фолда по документам реального train; из extra-данных выкидываются пары,
чей документ попал в отложенный фолд. Held-out не читается.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import boundary_adjust_v11 as B
import v11_lab as L
from spanver import ranker
from spanver.evaluate import iou, metrics
from spanver.module import candidate_spans, document_context
from train_pair_classifier_v12 import export_trees, tree_probability

GUARD = {"span": 0.50, "token": 0.50, "pair": 0.50, "fpr_max": 0.40, "mfava_span": 0.65}
BASE = {"span": 0.3363, "token": 0.6191, "pair": 0.4872, "fpr": 0.125, "mfava_span": 0.6895}
TREES_CONFIG = {"learning_rate": 0.1, "max_depth": 4, "max_leaf_nodes": 31,
                "max_iter": 150, "random_state": 0}


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def doc_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def is_clause_pair(pair):
    answer = pair.get("model_output") or ""
    gold = pair.get("spans") or []
    if not gold or not answer:
        return False
    if any(span["start"] == 0 and span["end"] == len(answer) - 1 for span in gold):
        return False
    if len(gold) > 1:
        return True
    span = gold[0]
    return (span["end"] - span["start"]) / max(1, len(answer)) < 0.8


def local_candidates(pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    if not answer:
        return []
    context = document_context(document)
    gold = pair.get("spans") or []
    rows = []
    for start, end, fragment in candidate_spans(answer):
        if start == 0 and end == len(answer) - 1:
            continue
        vector = ranker.features(fragment, start, end, answer, document,
                                 context[0], context[2])
        hit = any(iou(start, end, span["start"], span["end"]) >= 0.5 for span in gold)
        rows.append((vector, int(hit)))
    return rows


def build_matrix(pairs, cache_path):
    """X, y и хэш документа на каждую строку-кандидата (для OOF-маски)."""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["docs"])
    X, y, docs = [], [], []
    started = time.time()
    for pair in pairs:
        key = doc_hash(pair.get("document_text"))
        for vector, label in local_candidates(pair):
            X.append(vector)
            y.append(label)
            docs.append(key)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    np.savez_compressed(cache_path, X=X, y=y, docs=np.asarray(docs))
    print("built", cache_path.name, X.shape, "pos", int(y.sum()),
          f"{time.time() - started:.1f}s", flush=True)
    return X, y, docs


def fit_local(X, y):
    classifier = HistGradientBoostingClassifier(**TREES_CONFIG).fit(X, y)
    model = export_trees(classifier)
    sample = X[:: max(1, len(X) // 64)] if len(X) else X
    delta = 0.0
    if len(sample):
        expected = classifier.predict_proba(sample)[:, 1]
        actual = np.asarray([tree_probability(model, list(row)) for row in sample])
        delta = float(np.max(np.abs(expected - actual)))
    return model, delta


def fold_of(pair, folds=3):
    return int(doc_hash(pair.get("document_text"))[:8], 16) % folds


def evaluate(pairs, cache, adjuster, config):
    return metrics(pairs, lambda pair: B.adjust(pair, cache, config, adjuster[0],
                                                adjuster[1], 0.5, "hybrid"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/span_oof_v14.json")
    parser.add_argument("--model", default="artifacts/local_hgb_v14.json")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args(argv)

    base_params = L.weights()
    L.set_weights(base_params)
    config = dict(base_params["config"])
    cache = L.Cache(params=base_params)
    train, val, mfava = L.load_all()
    sources = {
        "real": train + mfava,
        "synth": read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl"),
        "pseudo": read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl"),
        "ragtruth": read_jsonl(ROOT / "artifacts/ragtruth_proj_v14.jsonl"),
    }
    adjuster = B.train(B.collect(train, cache, config, "hybrid"))

    cache_dir = ROOT / "artifacts/cache_v14"
    cache_dir.mkdir(exist_ok=True)
    matrices = {}
    for name, rows in sources.items():
        keep = [pair for pair in rows if is_clause_pair(pair)]
        print(name, "пар", len(rows), "clause-пар", len(keep), flush=True)
        matrices[name] = build_matrix(keep, cache_dir / f"local_{name}.npz")

    variant_parts = {
        "real": ["real"],
        "real+synth": ["real", "synth"],
        "real+synth+pseudo": ["real", "synth", "pseudo"],
        "real+synth+pseudo+ragtruth": ["real", "synth", "pseudo", "ragtruth"],
    }

    def stack(parts, drop_docs=None):
        X_blocks, y_blocks = [], []
        for part in parts:
            block_X, block_y, docs = matrices[part]
            mask = np.ones(len(block_y), dtype=bool)
            if drop_docs:
                mask = np.asarray([doc not in drop_docs for doc in docs])
            X_blocks.append(block_X[mask])
            y_blocks.append(block_y[mask])
        return np.vstack(X_blocks), np.concatenate(y_blocks)

    report = {
        "rule": "дообучается только local_hgb (те же 37 признаков и рецепт train_local_ranker); "
                "mode-гейт, boundary adjuster и пороги v11 не меняются и заново не подбираются",
        "baseline": BASE, "guard": GUARD, "trees_config": TREES_CONFIG,
        "data": {name: {"pairs": len(keep)} for name, keep in
                 ((n, [p for p in rows if is_clause_pair(p)]) for n, rows in sources.items())},
        "variants": {},
    }

    best = None
    best_model = None
    for name, parts in variant_parts.items():
        X, y = stack(parts)
        started = time.time()
        model, delta = fit_local(X, y)
        fit_seconds = time.time() - started
        params = dict(base_params)
        params["local_hgb"] = model
        L.set_weights(params)
        try:
            val_metrics = L.cell(evaluate(val, cache, adjuster, config))
            mfava_metrics = L.cell(evaluate(mfava, cache, adjuster, config))
        finally:
            L.set_weights(base_params)
        entry = {"rows": int(X.shape[0]), "positives": int(y.sum()),
                 "fit_seconds": round(fit_seconds, 1),
                 "nodes": sum(len(tree) for tree in model["trees"]),
                 "json_bytes": len(json.dumps(model)), "export_delta": delta,
                 "val": val_metrics, "mfava": mfava_metrics,
                 "guard_pass": bool(val_metrics["span"] >= GUARD["span"]
                                    and val_metrics["token"] >= GUARD["token"]
                                    and val_metrics["pair"] >= GUARD["pair"]
                                    and val_metrics["fpr"] <= GUARD["fpr_max"]
                                    and mfava_metrics["span"] >= GUARD["mfava_span"])}
        report["variants"][name] = entry
        print(name, val_metrics, "mfava", mfava_metrics["span"], "guard",
              entry["guard_pass"], flush=True)
        if best is None or val_metrics["span"] > report["variants"][best]["val"]["span"]:
            best, best_model = name, model

    # 3-fold OOF лучшего варианта: обучение на 2/3 документов реального train,
    # extra-пары с документами из отложенного фолда выкидываются
    parts = variant_parts[best]
    fold_rows = []
    for fold in range(args.folds):
        held = [pair for pair in train if fold_of(pair, args.folds) == fold]
        held_docs = {doc_hash(pair.get("document_text")) for pair in held}
        X, y = stack(parts, drop_docs=held_docs)
        model, delta = fit_local(X, y)
        params = dict(base_params)
        params["local_hgb"] = model
        L.set_weights(params)
        try:
            held_metrics = L.cell(evaluate(held, cache, adjuster, config))
            fold_val = L.cell(evaluate(val, cache, adjuster, config))
        finally:
            L.set_weights(base_params)
        L.set_weights(base_params)
        base_held = L.cell(metrics(held, lambda pair: ranker.select_spans(
            pair["model_output"], pair["document_text"])))
        fold_rows.append({"fold": fold, "held_pairs": len(held),
                          "train_rows": int(X.shape[0]), "export_delta": delta,
                          "held": held_metrics, "held_baseline_select_spans": base_held,
                          "val_with_fold_model": fold_val})
        print("fold", fold, held_metrics, "val", fold_val["span"], flush=True)

    report["oof"] = {
        "variant": best,
        "folds": fold_rows,
        "mean_held_span": round(float(np.mean([row["held"]["span"] for row in fold_rows])), 4),
        "mean_held_token": round(float(np.mean([row["held"]["token"] for row in fold_rows])), 4),
        "mean_held_pair": round(float(np.mean([row["held"]["pair"] for row in fold_rows])), 4),
        "mean_held_fpr": round(float(np.mean([row["held"]["fpr"] for row in fold_rows])), 4),
        "mean_val_span": round(float(np.mean([row["val_with_fold_model"]["span"]
                                              for row in fold_rows])), 4),
        "mean_val_mfava": None,
    }
    L.set_weights(base_params)
    params = dict(base_params)
    params["local_hgb"] = best_model
    L.set_weights(params)
    try:
        mfava_full = L.cell(evaluate(mfava, cache, adjuster, config))
    finally:
        L.set_weights(base_params)
    report["best"] = {"variant": best, "val": report["variants"][best]["val"],
                      "mfava": mfava_full,
                      "model_bytes": len(json.dumps(best_model))}
    report["oof"]["mean_val_mfava"] = mfava_full["span"]
    (ROOT / args.model).write_text(json.dumps(best_model, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best", best, report["best"]["val"], "mfava", mfava_full["span"],
          "oof span", report["oof"]["mean_held_span"], flush=True)


if __name__ == "__main__":
    main()
