"""V11 блок 2: stacker — listwise-ранкер поверх скоров двух голов.

Мотивация из ablation (artifacts/score_compare_v11.json): baseline-голова находит
whole (20/23 top-1 на val), local HGB находит clause (9/18), но каждая по
отдельности плоха на чужом типе. Stacker получает скоры обеих голов как признаки
и учится выбирать между гипотезами на РЕАЛЬНЫХ gold-метках train+mFAVA
(синтетика и pseudo — только в ablation, по умолчанию выключены).

Декодирование: top-k выше порога при гейте; max_spans=1 оптимизирует precision.
Held-out не читается.
"""
import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import lightgbm as lgb
import numpy as np

import v11_lab as L
import v11_ranker as R
from spanver.evaluate import iou, metrics, read_jsonl

STACK = (
    "base", "local", "listwise", "base_rank", "local_rank", "listwise_rank",
    "base_gap", "local_gap", "listwise_gap", "is_whole", "log_len", "rel_len", "position",
    "token_cov", "cos", "left_ok", "right_ok", "is_clause", "cov_gap", "is_min_cov",
    "answer_cov", "answer_cos", "min_clause_cov", "mean_clause_cov", "cov_spread",
    "n_clauses", "gate", "support_rank", "iou_base_top1", "iou_local_top1",
)
N_STACK = len(STACK)
SUPPORT_IDX = [STACK.index(name) for name in (
    "token_cov", "cos", "cov_gap", "is_min_cov", "answer_cov", "answer_cos",
    "min_clause_cov", "mean_clause_cov", "cov_spread", "support_rank")]


def kind(pair):
    answer = pair.get("model_output") or ""
    spans = pair.get("spans") or []
    if not spans:
        return "neg"
    if any(span["start"] == 0 and span["end"] == len(answer) - 1 for span in spans):
        return "whole"
    return "short" if max(span["end"] - span["start"] + 1 for span in spans) < 60 else "clause"


def row_vector(entry, row, params):
    ctx = entry["ctx"]
    answer = ctx["answer"]
    text = answer[row["start"]:row["end"] + 1]
    feature = R.vector(ctx, row["start"], row["end"], text)
    mapping = dict(zip(R.FEATURES, feature))
    covs = ctx["clause_covs"] or [0.0]
    mapping["cov_spread"] = max(covs) - min(covs)
    total = max(1, len(entry["rows"]))
    base_rank = sorted(entry["rows"], key=lambda item: (-item["base"], item["start"])).index(row) / total
    local_rank = sorted(entry["rows"], key=lambda item: (-item["local"], item["start"])).index(row) / total
    list_rank = sorted(entry["rows"], key=lambda item: (-item["listwise"], item["start"])).index(row) / total
    base_top1 = max(entry["rows"], key=lambda item: (item["base"], -item["start"]))
    local_top1 = max(entry["rows"], key=lambda item: (item["local"], -item["start"]))
    values = [
        row["base"], row["local"], row["listwise"], base_rank, local_rank, list_rank,
        row["base"] - max(item["base"] for item in entry["rows"]),
        row["local"] - max(item["local"] for item in entry["rows"]),
        row["listwise"] - max(item["listwise"] for item in entry["rows"]),
        float(row["start"] == 0 and row["end"] == len(answer) - 1),
        np.log1p(len(text)), len(text) / max(1, len(answer)), row["start"] / max(1, len(answer)),
        mapping["token_cov"], mapping["cos"], mapping["left_ok"], mapping["right_ok"],
        mapping["is_clause"], mapping["cov_gap"], mapping["is_min_cov"],
        mapping["answer_cov"], mapping["answer_cos"], mapping["min_clause_cov"],
        mapping["mean_clause_cov"], mapping["cov_spread"], mapping["n_clauses"],
        entry["ctx"]["gate"], mapping["support_rank"],
        iou(row["start"], row["end"], base_top1["start"], base_top1["end"]),
        iou(row["start"], row["end"], local_top1["start"], local_top1["end"]),
    ]
    del params
    return values


def negatives(entry, pair, positives, rng, max_negatives=12):
    ctx = entry["ctx"]
    answer = ctx["answer"]
    rows = entry["rows"]
    seen = {(item["start"], item["end"]) for item in positives}
    out = []

    def add(start, end):
        if start is None or end is None or start < 0 or end >= len(answer) or start >= end:
            return
        if (start, end) in seen:
            return
        if any(iou(start, end, item["start"], item["end"]) >= 0.5 for item in positives):
            return
        seen.add((start, end))
        match = next((row for row in rows if row["start"] == start and row["end"] == end), None)
        if match:
            out.append(match)

    for item in list(positives):
        for step in (-3, -2, -1, 1, 2, 3):
            shift = R.snap_to_words(answer, item["start"], item["end"], step)
            if shift:
                add(*shift)
        for _, left, right in R.neighbour_clauses(ctx, item["start"], item["end"]):
            add(left, right)
    base_top1 = max(rows, key=lambda row: (row["base"], -row["start"]))
    local_top1 = max(rows, key=lambda row: (row["local"], -row["start"]))
    add(base_top1["start"], base_top1["end"])
    add(local_top1["start"], local_top1["end"])
    pool = list(rows)
    rng.shuffle(pool)
    for row in pool:
        if len(out) >= max_negatives:
            break
        add(row["start"], row["end"])
    return out[:max_negatives]


def build_dataset(pairs, cache, params, with_synth_pairs, seed=11, whole_weight=1.0):
    rng = random.Random(seed)
    X, y, group, meta, weights = [], [], [], [], []
    for pair in pairs:
        entry = cache[pair["pair_id"]]
        if not entry["rows"]:
            continue
        gold = pair.get("spans") or []
        positives = []
        for span in gold:
            best, best_iou = None, 0.0
            for row in entry["rows"]:
                value = iou(row["start"], row["end"], span["start"], span["end"])
                if value > best_iou:
                    best, best_iou = row, value
            if best is not None and best_iou >= 0.5:
                positives.append(best)
        if not positives:
            continue
        members = [(row, 1) for row in positives] + [(row, 0) for row in negatives(entry, pair, positives, rng)]
        weight = whole_weight if kind(pair) == "whole" else 1.0
        weights.extend([weight] * len(members))
        group.append(len(members))
        meta.append({"pair_id": pair["pair_id"], "kind": kind(pair), "rows": len(members),
                     "positives": len(positives),
                     "synthetic": pair["pair_id"] in with_synth_pairs})
        for row, label in members:
            X.append(row_vector(entry, row, params))
            y.append(label)
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), group, meta,
            np.array(weights, dtype=np.float32))


def decode(entry, config):
    ctx = entry["ctx"]
    answer = ctx["answer"]
    if ctx["gate"] < config["gate"] or not entry["rows"]:
        return []
    ordered = sorted(entry["rows"], key=lambda row: (-row["stack"], row["start"]))
    chosen = []
    for row in ordered:
        if row["stack"] < config["threshold"]:
            break
        if any(iou(row["start"], row["end"], other["start"], other["end"]) > 0.1 for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= config["max_spans"]:
            break
    return [{"start": row["start"], "end": row["end"], "risk": round(float(row["stack"]), 4),
             "type": "unsupported_addition", "confidence": 0.5} for row in chosen]


def evaluate(cache, pairs, config):
    picked = {pair["pair_id"]: decode(cache[pair["pair_id"]], config) for pair in pairs}
    return metrics(pairs, lambda pair: picked[pair["pair_id"]])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/listwise_whole_model_v11.json")
    parser.add_argument("--cache", default="artifacts/router2_cache_v11.pkl")
    parser.add_argument("--with-synth", action="store_true")
    parser.add_argument("--with-pseudo", action="store_true")
    parser.add_argument("--whole-weight", type=float, default=1.0)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--leaves", type=int, default=15)
    parser.add_argument("--report", default="artifacts/stacker_v11.json")
    parser.add_argument("--export", default="artifacts/stacker_model_v11.json")
    args = parser.parse_args(argv)

    started = time.time()
    params = L.weights()
    payload = json.loads((ROOT / args.model).read_text(encoding="utf-8"))
    cols = payload["columns"]
    gold = read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
    train_pos = [p for p in gold if p.get("split") == "train" and p.get("spans")]
    val = [p for p in gold if p.get("split") == "val"]
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))

    if Path(args.cache).exists():
        with open(args.cache, "rb") as handle:
            cache = pickle.load(handle)
    else:
        raise SystemExit("нет кэша router2; запустите tools/router_v2_v11.py")

    synth_ids, pseudo_ids = set(), set()
    training = list(train_pos)
    if args.with_synth:
        synth = read_jsonl(str(ROOT / "artifacts/rushallu_synth_v11.jsonl"))
        missing = [p for p in synth if p["pair_id"] not in cache]
        if missing:
            from router_v2_v11 import build_cache
            cache.update(build_cache(missing, params, payload, cols))
            with open(args.cache, "wb") as handle:
                pickle.dump(cache, handle)
        synth_ids = {p["pair_id"] for p in synth}
        training += synth
    if args.with_pseudo:
        pseudo = read_jsonl(str(ROOT / "artifacts/clause_pseudo_v11.jsonl"))
        missing = [p for p in pseudo if p["pair_id"] not in cache]
        if missing:
            from router_v2_v11 import build_cache
            cache.update(build_cache(missing, params, payload, cols))
        pseudo_ids = {p["pair_id"] for p in pseudo}
        training += pseudo
    training += mfava

    X, y, group, meta, weights = build_dataset(training, cache, params, synth_ids | pseudo_ids,
                                               whole_weight=args.whole_weight)
    for pair in val + mfava:
        entry = cache[pair["pair_id"]]
        for row in entry["rows"]:
            row["vector"] = np.array(row_vector(entry, row, params), dtype=np.float32)

    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=args.trees, learning_rate=0.05,
                           num_leaves=args.leaves, min_child_samples=10, subsample=0.9,
                           colsample_bytree=0.9, random_state=11, label_gain=[0, 1],
                           verbose=-1, n_jobs=2)
    model.fit(X, y, group=group, sample_weight=weights)
    exported = R.export_lightgbm(model.booster_)
    exported["whole_weight"] = args.whole_weight
    exported["feature_names"] = list(STACK)
    exported["support_idx"] = SUPPORT_IDX
    sample = X[:: max(1, len(X) // 200)]
    expected = model.predict(sample)
    actual = np.array([R.predict_raw(exported, list(row)) for row in sample])
    export_delta = float(np.max(np.abs(expected - actual))) if len(expected) else 0.0

    for pair in val + mfava:
        entry = cache[pair["pair_id"]]
        for row in entry["rows"]:
            row["stack"] = R.predict_raw(exported, list(row["vector"]))

    rows_out = []
    for threshold in (-6.0, -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0):
        for max_spans in (1, 2, 3):
            for gate_cut in (0.3, 0.4, 0.5, 0.6):
                config = {"threshold": threshold, "max_spans": max_spans, "gate": gate_cut}
                rows_out.append({
                    "threshold": threshold, "max_spans": max_spans, "gate": gate_cut,
                    "val": L.cell(evaluate(cache, val, config)),
                    "mfava": L.cell(evaluate(cache, mfava, config)),
                    "combined": L.cell(evaluate(cache, val + mfava, config)),
                })
    rows_out.sort(key=lambda row: (row["combined"]["span"], row["val"]["span"]), reverse=True)
    passing = [row for row in rows_out if L.guard([row["val"], row["combined"]])]
    by_kind = {}
    for pair in val:
        entry = cache[pair["pair_id"]]
        if entry["kind"] == "neg" or not entry["rows"]:
            continue
        top = max(entry["rows"], key=lambda row: (row["stack"], -row["start"]))
        ok = any(iou(top["start"], top["end"], span["start"], span["end"]) >= 0.5
                 for span in (pair.get("spans") or []))
        by_kind.setdefault(entry["kind"], []).append(int(ok))
    top1 = {key: {"hits": sum(value), "n": len(value), "acc": round(sum(value) / len(value), 3)}
            for key, value in by_kind.items()}

    export_path = ROOT / args.export
    export_path.write_text(json.dumps(exported), encoding="utf-8")
    report = {
        "rule": "stacker lambdarank на реальных gold train+mFAVA; пороги val+mFAVA; held не читается",
        "with_synth": args.with_synth, "with_pseudo": args.with_pseudo,
        "whole_weight": args.whole_weight,
        "train": {"pairs": len(meta), "rows": int(len(y)), "positives": int(y.sum()),
                  "by_kind": {key: sum(1 for item in meta if item["kind"] == key)
                              for key in ("whole", "clause", "short")},
                  "synthetic_pairs": sum(1 for item in meta if item["synthetic"])},
        "export": {"max_abs_delta": round(export_delta, 10),
                   "sha256": hashlib.sha256(export_path.read_bytes()).hexdigest()},
        "val_top1_by_kind": top1, "best": rows_out[0], "top": rows_out[:15],
        "passing": passing[:10], "configs": rows_out,
        "elapsed_sec": round(time.time() - started, 1),
    }
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("train", json.dumps(report["train"], ensure_ascii=False), flush=True)
    print("top1", json.dumps(top1, ensure_ascii=False), flush=True)
    print("best", json.dumps(rows_out[0], ensure_ascii=False), flush=True)
    print("export_delta", export_delta, "guard", len(passing), flush=True)


if __name__ == "__main__":
    main()
