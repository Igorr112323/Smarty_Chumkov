"""V11 блок 2.1/2.2: listwise-ранкер границ (LightGBM lambdarank) + ablation.

Группа = одна пара: позитив = gold-граница (clause-стиль), негативы = сдвиги ±1..3
слова, расширение/сжатие на clause, соседние clause, топ-1 baseline и случайные
кандидаты. Обучение — listwise (lambdarank), не бинарное.

Обучение: synth v11 + pseudo v11 + mFAVA train + gold train clause/short.
Whole-пары в ранкер не входят: выбор «весь ответ» — задача router (блок 3).
Оценка: val (150) и mFAVA train (148) — dev; held-out не читается.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import lightgbm as lgb

import v11_lab as L
import v11_ranker as R
from spanver.evaluate import iou, read_jsonl

SUPPORT_COLUMNS = [R.FEATURES.index(name) for name in (
    "token_cov", "cos", "sentence_cov", "verbatim", "numbers_missing", "names_missing",
    "answer_cov", "answer_cos", "min_clause_cov", "mean_clause_cov", "n_low_cov",
    "deficit", "cov_gap", "cov_rank", "is_min_cov",
)]
BOUNDARY_COLUMNS = [R.FEATURES.index(name) for name in (
    "left_ok", "right_ok", "is_clause", "sentence_aligned",
)]


def columns(feature_set):
    if feature_set == "full":
        return list(range(R.N_FEATURES))
    if feature_set == "support":
        return list(SUPPORT_COLUMNS)
    if feature_set == "no_support":
        return [index for index in range(R.N_FEATURES) if index not in set(SUPPORT_COLUMNS)]
    if feature_set == "boundary":
        return list(BOUNDARY_COLUMNS)
    raise SystemExit(f"unknown feature set {feature_set}")


def is_whole_pair(pair):
    answer = pair.get("model_output") or ""
    return any(span["start"] == 0 and span["end"] == len(answer) - 1
               for span in (pair.get("spans") or []))


def training_pairs(with_pseudo=False, with_whole=True):
    gold = read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
    train = [p for p in gold if p.get("split") == "train" and p.get("spans")]
    if not with_whole:
        train = [p for p in train if not is_whole_pair(p)]
    pseudo = read_jsonl(str(ROOT / "artifacts/clause_pseudo_v11.jsonl")) if with_pseudo else []
    synth = read_jsonl(str(ROOT / "artifacts/rushallu_synth_v11.jsonl"))
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    pairs = train + pseudo + synth + mfava
    train_ids = {id(pair) for pair in train}
    pseudo_ids = {id(pair) for pair in pseudo}
    synth_ids = {id(pair) for pair in synth}
    for pair in pairs:
        if id(pair) in train_ids:
            pair["_source"] = "gold_train"
        elif id(pair) in pseudo_ids:
            pair["_source"] = "pseudo"
        elif id(pair) in synth_ids:
            pair["_source"] = "synth"
        else:
            pair["_source"] = "mfava"
    return pairs


def build_groups(pairs, params, seed=11, max_negatives=12):
    rng = random.Random(seed)
    X, y, group, meta = [], [], [], []
    for pair in pairs:
        ctx = R.pair_context(pair, params)
        rows = R.make_group(pair, ctx, pair["spans"], rng, max_negatives)
        if not any(row["label"] for row in rows):
            continue
        group.append(len(rows))
        meta.append({"pair_id": pair["pair_id"], "source": pair.get("_source"),
                     "rows": len(rows), "positives": sum(row["label"] for row in rows)})
        for row in rows:
            X.append(row["vector"])
            y.append(row["label"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), group, meta


def train_model(X, y, group, cols, trees=300, lr=0.05, leaves=15):
    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=trees, learning_rate=lr, num_leaves=leaves,
        min_child_samples=10, subsample=0.9, colsample_bytree=0.9, random_state=11,
        label_gain=[0, 1], verbose=-1, n_jobs=2,
    )
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[cols] = True
    model.fit(X[:, mask], y, group=group)
    return model, mask


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def score_pair(model, mask, pair, params):
    ctx = R.pair_context(pair, params)
    rows = ctx["rows"] if ctx["rows"] else []
    if not rows:
        return ctx, []
    X = np.array([R.vector(ctx, row["start"], row["end"], row["text"]) for row in rows], dtype=np.float32)
    scores = model.predict(X[:, mask])
    return ctx, list(zip(rows, scores))


def decode(scored, threshold, max_spans, nms_iou, gate, gate_cut):
    if gate < gate_cut:
        return []
    chosen = []
    for row, score in sorted(scored, key=lambda item: (-item[1], item[0]["start"])):
        if score < threshold:
            break
        if any(iou(row["start"], row["end"], other["start"], other["end"]) > nms_iou
               for other, _ in chosen):
            continue
        chosen.append((row, score))
        if len(chosen) >= max_spans:
            break
    return chosen


def evaluate(scored_all, pairs, threshold, max_spans, nms_iou, gate_cut):
    picked = {}
    for pair in pairs:
        ctx, scored = scored_all[pair["pair_id"]]
        picked[pair["pair_id"]] = [
            {"start": row["start"], "end": row["end"], "risk": round(float(score), 4),
             "type": "unsupported_addition", "confidence": round(float(sigmoid(score)), 4)}
            for row, score in decode(scored, threshold, max_spans, nms_iou, ctx["gate"], gate_cut)
        ]

    def predictor(pair):
        return picked[pair["pair_id"]]

    from spanver.evaluate import metrics
    return metrics(pairs, predictor)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="full",
                        choices=["full", "support", "no_support", "boundary"])
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--leaves", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--with-pseudo", action="store_true")
    parser.add_argument("--no-whole", action="store_true")
    parser.add_argument("--report", default="artifacts/listwise_v11.json")
    parser.add_argument("--model", default="artifacts/listwise_model_v11.json")
    args = parser.parse_args(argv)

    params = L.weights()
    started = time.time()
    tag = f"{int(args.with_pseudo)}{int(not args.no_whole)}"
    cache_path = ROOT / f"artifacts/listwise_groups_v11_{tag}.npz"
    meta_path = ROOT / f"artifacts/listwise_groups_meta_v11_{tag}.json"
    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=True)
        X, y, group = stored["X"], stored["y"], list(stored["group"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print("groups from cache", X.shape, flush=True)
    else:
        pairs = training_pairs(args.with_pseudo, not args.no_whole)
        X, y, group, meta = build_groups(pairs, params)
        np.savez_compressed(cache_path, X=X, y=y, group=np.array(group))
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    cols = columns(args.features)
    model, mask = train_model(X, y, group, cols, args.trees, args.lr, args.leaves)
    exported = R.export_lightgbm(model.booster_)
    exported["feature_set"] = args.features
    exported["columns"] = cols
    exported["feature_names"] = list(R.FEATURES)
    sample = X[:: max(1, len(X) // 200)]
    expected = model.predict(sample[:, mask])
    actual = np.array([R.predict_raw(exported, list(np.asarray(row)[cols])) for row in sample])
    export_delta = float(np.max(np.abs(expected - actual))) if len(expected) else 0.0
    print("export max|delta|", round(export_delta, 8), flush=True)

    train_pairs, val, mfava = L.load_all()
    del train_pairs
    scored_all = {}
    for pair in val + mfava:
        scored_all[pair["pair_id"]] = score_pair(model, mask, pair, params)
    # baseline v3 на тех же парах — контрольная точка ablation
    cache = L.Cache(params=params)

    def baseline(pair):
        rows, gate = cache.get(pair)
        if gate < params["config"]["gate"]:
            return []
        return [{"start": row["start"], "end": row["end"], "risk": row["risk"],
                 "type": "unsupported_addition", "confidence": row["risk"]}
                for row in L.nms_pick(rows, params["config"]["threshold"],
                                      params["config"]["max_spans"],
                                      params["config"]["nms_iou"])]

    from spanver.evaluate import metrics
    baseline_cells = {"combined": L.cell(metrics(val + mfava, baseline)),
                      "val": L.cell(metrics(val, baseline)),
                      "mfava": L.cell(metrics(mfava, baseline))}
    print("baseline", baseline_cells, flush=True)

    rows_out = []
    for threshold in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        for max_spans in (1, 2, 3):
            for gate_cut in (0.3, 0.5, 0.6):
                combined = evaluate(scored_all, val + mfava, threshold, max_spans, 0.1, gate_cut)
                rows_out.append({
                    "threshold": threshold, "max_spans": max_spans, "gate": gate_cut,
                    "combined": L.cell(combined),
                    "val": L.cell(evaluate(scored_all, val, threshold, max_spans, 0.1, gate_cut)),
                    "mfava": L.cell(evaluate(scored_all, mfava, threshold, max_spans, 0.1, gate_cut)),
                })
    rows_out.sort(key=lambda row: (row["combined"]["span"], row["combined"]["token"]), reverse=True)
    best = rows_out[0]
    passing = [row for row in rows_out if L.guard([row["combined"], row["val"]])]
    report = {
        "rule": "listwise lambdarank на train-only данных; пороги по val+mFAVA; held не читается",
        "features": args.features, "columns": cols, "trees": args.trees,
        "with_pseudo": args.with_pseudo, "with_whole": not args.no_whole,
        "leaves": args.leaves, "lr": args.lr,
        "train": {"pairs": len(meta), "rows": int(len(y)), "positives": int(y.sum()),
                  "by_source": {key: sum(1 for item in meta if item["source"] == key)
                                for key in ("gold_train", "pseudo", "synth", "mfava")}},
        "export": {"max_abs_delta": round(export_delta, 10),
                   "sha256": hashlib.sha256(
                       json.dumps(exported, sort_keys=True).encode()).hexdigest()},
        "baseline": baseline_cells, "best": best, "top": rows_out[:15],
        "passing": passing[:10], "configs": rows_out,
        "elapsed_sec": round(time.time() - started, 1),
    }
    model_path = ROOT / args.model
    model_path.write_text(json.dumps(exported), encoding="utf-8")
    report["model_file"] = args.model
    report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("best", best, flush=True)
    print("passing", len(passing), flush=True)
    print("report_sha", hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
