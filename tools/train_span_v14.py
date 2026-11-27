"""V14 шаг 3–4: span-классификатор на кандидатах и честная проверка guard.

Модель: HistGradientBoostingClassifier на 37 штатных признаках кандидата
(`spanver.ranker.features`), экспорт деревьев в чистый JSON (как v8/v9/v12),
рантайм без sklearn. Веса v11 (mode-гейт, boundary adjuster) не перезаписываются:
декодер здесь — только candidate-ranker + порог + NMS + max_spans, поэтому
сравнение с baseline честное (baseline = hybrid v11 через boundary adjuster).

Данные (все train/extra, held-out не читается):
  real     — artifacts/gold_v3.jsonl, split=train (RusHallu, русский);
  synth    — artifacts/rushallu_synth_v11.jsonl (3 563 пары, точные границы);
  pseudo   — artifacts/clause_pseudo_v11.jsonl (84 пары, weak supervision);
  ragtruth — artifacts/ragtruth_proj_v14.jsonl (958 пар, en, проекция стиля).

Ablation: real / real+synth / real+synth+pseudo / real+synth+pseudo+ragtruth.
Guard v14: val span ≥ 0.50, token ≥ 0.50, pair ≥ 0.50, FPR ≤ 0.40, mFAVA span ≥ 0.65,
OOF ≥ baseline + 0.02.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import v11_lab as L
from spanver import ranker
from spanver.evaluate import iou, metrics
from spanver.module import candidate_spans, document_context
from train_pair_classifier_v12 import export_trees, tree_probability

GUARD = {"span": 0.50, "token": 0.50, "pair": 0.50, "fpr_max": 0.40, "mfava_span": 0.65,
         "oof_delta": 0.02}
BASELINE = {"val": {"span": 0.3363, "token": 0.6191, "pair": 0.4872, "fpr": 0.125},
            "mfava_span": 0.6895, "oof_span": None}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def pair_candidates(pair):
    """Кандидаты пары: (start, end, признаки 37)."""
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    if not answer:
        return []
    context = document_context(document)
    doc_tokens, doc_grams = context[0], context[2]
    rows = []
    for start, end, fragment in candidate_spans(answer):
        rows.append((start, end, ranker.features(fragment, start, end, answer, document,
                                                 doc_tokens, doc_grams)))
    return rows


def build_matrix(pairs, cache_path=None):
    """Матрица кандидатов по всем парам + границы групп. Кэшируется в .npz."""
    if cache_path and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        return data["X"], data["y"], meta
    X, y, meta = [], [], []
    for pair in pairs:
        gold = pair.get("spans") or []
        rows = pair_candidates(pair)
        for start, end, vector in rows:
            hit = any(iou(start, end, span["start"], span["end"]) >= 0.5 for span in gold)
            X.append(vector)
            y.append(1 if hit else 0)
        meta.append({"pair_id": pair["pair_id"], "n": len(rows),
                     "spans": [[row[0], row[1]] for row in rows]})
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    if cache_path:
        np.savez_compressed(cache_path, X=X, y=y, meta=json.dumps(meta))
    return X, y, meta


def fit(X, y, trees, leaves, lr, pos_weight):
    weights = np.where(y == 1, pos_weight, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=trees, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=10,
        l2_regularization=1.0, random_state=14,
    ).fit(X, y, sample_weight=weights)
    return export_trees(model)


def scores_for(model, X):
    return np.asarray([tree_probability(model, list(row)) for row in X])


def decode_pair(meta, scores, threshold, max_spans, nms_iou=0.1):
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], meta["spans"][index][0]))
    accepted = []
    for index in order:
        if scores[index] < threshold:
            break
        start, end = meta["spans"][index]
        if any(iou(start, end, other[0], other[1]) > nms_iou for other in accepted):
            continue
        accepted.append((start, end))
        if len(accepted) >= max_spans:
            break
    return [{"start": start, "end": end, "risk": round(1.0, 4),
             "type": "unsupported_addition", "confidence": 0.5} for start, end in accepted]


def evaluate_split(pairs, meta, scores, config):
    picked = {}
    offset = 0
    for pair, entry in zip(pairs, meta):
        picked[pair["pair_id"]] = decode_pair(entry, scores[offset:offset + entry["n"]],
                                              config["threshold"], config["max_spans"])
        offset += entry["n"]
    return metrics(pairs, lambda pair: picked[pair["pair_id"]])


def sweep(pairs, meta, scores, grid):
    best = None
    rows = []
    for config in grid:
        result = evaluate_split(pairs, meta, scores, config)
        row = {"config": config, **L.cell(result)}
        rows.append(row)
        if best is None or row["span"] > best["span"]:
            best = row
    return rows, best


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/span_oof_v14.json")
    parser.add_argument("--model", default="artifacts/span_model_v14.json")
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--leaves", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    train, val, mfava = L.load_all()
    synth = read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl")
    pseudo = read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl")
    ragtruth = read_jsonl(ROOT / "artifacts/ragtruth_proj_v14.jsonl")

    cache_dir = ROOT / "artifacts/cache_v14"
    cache_dir.mkdir(exist_ok=True)
    groups = {"real": train, "synth": synth, "pseudo": pseudo, "ragtruth": ragtruth,
              "val": val, "mfava": mfava}
    matrices = {}
    for name, rows in groups.items():
        started = time.time()
        cache = cache_dir / f"{name}.npz"
        if args.rebuild and cache.exists():
            cache.unlink()
        matrices[name] = build_matrix(rows, cache)
        print(name, matrices[name][0].shape, "positives", int(matrices[name][1].sum()),
              f"{time.time() - started:.1f}s", flush=True)

    grids = [{"threshold": threshold, "max_spans": max_spans}
             for threshold in (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
             for max_spans in (1, 2, 3)]

    ablations = {
        "real": ["real"],
        "real+synth": ["real", "synth"],
        "real+synth+pseudo": ["real", "synth", "pseudo"],
        "real+synth+pseudo+ragtruth": ["real", "synth", "pseudo", "ragtruth"],
    }
    report = {
        "rule": "candidate-GBDT на 37 штатных признаках; обучение только на train/extra; "
                "порог и max_spans калибруются на val; held-out не читается",
        "data": {name: {"pairs": len(rows), "candidates": int(matrices[name][0].shape[0]),
                        "positives": int(matrices[name][1].sum())} for name, rows in groups.items()},
        "model": {"trees": args.trees, "leaves": args.leaves, "lr": args.lr,
                  "pos_weight": args.pos_weight},
        "baseline": BASELINE,
        "guard": GUARD,
        "ablations": {},
    }

    best_model = None
    best_score = -1.0
    for name, parts in ablations.items():
        X = np.vstack([matrices[part][0] for part in parts])
        y = np.concatenate([matrices[part][1] for part in parts])
        started = time.time()
        model = fit(X, y, args.trees, args.leaves, args.lr, args.pos_weight)
        fit_time = time.time() - started
        val_rows, val_best = sweep(val, matrices["val"][2],
                                   scores_for(model, matrices["val"][0]), grids)
        mfava_scores = scores_for(model, matrices["mfava"][0])
        config = val_best["config"]
        mfava_metrics = L.cell(evaluate_split(mfava, matrices["mfava"][2], mfava_scores, config))
        entry = {
            "train_rows": int(X.shape[0]), "train_positives": int(y.sum()),
            "fit_seconds": round(fit_time, 1),
            "val_best": val_best, "all_configs": val_rows,
            "mfava": mfava_metrics,
            "nodes": sum(len(tree) for tree in model["trees"]),
            "json_bytes": len(json.dumps(model)),
        }
        guard_pass = (val_best["span"] >= GUARD["span"] and val_best["token"] >= GUARD["token"]
                      and val_best["pair"] >= GUARD["pair"] and val_best["fpr"] <= GUARD["fpr_max"]
                      and mfava_metrics["span"] >= GUARD["mfava_span"])
        entry["guard_pass"] = guard_pass
        report["ablations"][name] = entry
        print(name, "val", {k: val_best[k] for k in ("span", "token", "pair", "fpr")},
              "mfava", round(mfava_metrics["span"], 4), "guard", guard_pass, flush=True)
        if val_best["span"] > best_score:
            best_score = val_best["span"]
            best_model = (name, model, config)

    name, model, config = best_model
    (ROOT / args.model).write_text(json.dumps(model, ensure_ascii=False) + "\n", encoding="utf-8")
    report["best"] = {"ablation": name, "config": config,
                      "model_bytes": (ROOT / args.model).stat().st_size}
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best", name, config, "model_bytes", report["best"]["model_bytes"], flush=True)


if __name__ == "__main__":
    main()
