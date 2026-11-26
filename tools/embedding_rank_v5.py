"""Émbedдинг-диагностика и условный locator v5.

Отличия от отклонённых v4-подходов:
  1) новый признак локальной подтверждённости (answer-token ↔ doc-token cosine);
  2) pair gate замораживается: new scorer работает только внутри пар, где
     baseline уже поставил флаг, поэтому pair FPR не меняется;
  3) обучение только на train, пороги/максимум выбираются на val+mFAVA-train.
"""
import argparse
import collections
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ranking_analysis_v5 as ra
from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl

W = ranker._model()


def emb_features(rows, emb_cache):
    by_pair = collections.defaultdict(list)
    for row in rows:
        by_pair[row["pid"]].append(row)
    result = [None] * len(rows)
    index = {id(row): i for i, row in enumerate(rows)}
    for pid, items in by_pair.items():
        data = emb_cache.get(pid)
        if not data:
            continue
        offsets = data["offsets"]
        sims = np.asarray(data["sim"], dtype=np.float32)
        centers = np.array([(s + e - 1) / 2.0 for s, e in offsets], dtype=np.float64)
        order = np.argsort(centers)
        centers = centers[order]
        sims_sorted = sims[order]
        for row in items:
            left = np.searchsorted(centers, row["start"], side="left")
            right = np.searchsorted(centers, row["end"], side="right")
            values = sims_sorted[left:right]
            if len(values) == 0:
                feature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                feature = [
                    float(values.mean()), float(values.min()), float(values.max()),
                    float(values.std()), float(np.mean(values < 0.4)), float(len(values)),
                ]
            result[index[id(row)]] = feature
    return np.array([f if f is not None else [0.0] * 6 for f in result], dtype=np.float32)


def rank_of(rows, key):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["pid"]].append(row)
    ranks = []
    for row in rows:
        if not row["matched"]:
            continue
        ranks.append(1 + sum(other[key] > row[key] for other in grouped[row["pid"]]))
    ranks.sort()
    return {
        "gold": len(ranks),
        "top1": sum(r == 1 for r in ranks),
        "top10": sum(r <= 10 for r in ranks),
        "median": ranks[len(ranks) // 2] if ranks else 0,
        "mrr": round(sum(1 / r for r in ranks) / len(ranks), 4) if ranks else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="artifacts/candidate_cache_v5.pkl")
    parser.add_argument("--emb", default="artifacts/embedding_v5_cache.pkl")
    parser.add_argument("--report", default="artifacts/embedding_rank_v5.json")
    parser.add_argument("--flags", default="artifacts/baseline_flags_v5.pkl")
    args = parser.parse_args()
    started = time.time()
    pairs = (
        load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
        + load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
        + read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    )
    ra.PAIRS = pairs
    byid = {p["pair_id"]: p for p in pairs}
    ids = {
        "train": {p["pair_id"] for p in pairs if p.get("split") == "train"},
        "val": {p["pair_id"] for p in pairs if p.get("split") == "val"},
        "mfava": {p["pair_id"] for p in pairs if p["pair_id"].startswith("mfava")},
    }
    rows = ra.load_or_build(pairs, args.cache)
    ra.scores_for(rows)
    with open(ROOT / args.emb, "rb") as handle:
        emb_cache = pickle.load(handle)["pairs"]
    ef = emb_features(rows, emb_cache)
    for row, values in zip(rows, ef):
        row["emb_mean"] = float(values[0])
        row["emb_low"] = float(values[4])
    print("embedding features", round(time.time() - started, 1), flush=True)

    report = {
        "rule": "Conditional locator: frozen baseline pair gate; только train/val/mFAVA-train",
        "emb_model_sha": hashlib.sha256((ROOT / "models/rubert-tiny2/pytorch_model.bin").read_bytes()).hexdigest(),
        "splits": {},
    }
    for name, pids in ids.items():
        subset = [r for r in rows if r["pid"] in pids]
        short_rows = [r for r in subset if r["end"] - r["start"] + 1 < 40]
        y = np.array([r["matched"] for r in short_rows], dtype=int)
        auc = {}
        for key in ("base", "local", "emb_mean", "emb_low"):
            if y.sum() and (1 - y).sum():
                auc[key] = round(roc_auc_score(y, [r[key] for r in short_rows]), 4)
        report["splits"][name] = {
            "pairs": len(pids), "short_candidates": len(short_rows),
            "short_gold": int(y.sum()), "short_auc": auc,
            "rank_base_short": rank_of(short_rows, "base"),
            "rank_local_short": rank_of(short_rows, "local"),
        }
    print("auc", {k: v.get("short_auc") for k, v in report["splits"].items()}, flush=True)

    flags_path = ROOT / args.flags
    if flags_path.exists():
        flags = pickle.load(open(flags_path, "rb"))
    else:
        flags = {}
        for p in pairs:
            flags[p["pair_id"]] = bool(ranker.select_spans(p.get("model_output") or "", p.get("document_text") or ""))
        pickle.dump(flags, open(flags_path, "wb"))
    flagged = {pid for pid, value in flags.items() if value}
    print("flagged", {name: sum(flags[pid] for pid in pids) for name, pids in ids.items()}, flush=True)

    def vector(row):
        length = row["end"] - row["start"] + 1
        return [
            row["base"], row["local"], row["emb_mean"], row["emb_low"],
            float(length), np.log1p(length), float(row["emb_mean"] < 0.35),
        ]

    train_rows = [r for r in rows if r["pid"] in ids["train"] and r["pid"] in flagged]
    X_train = np.array([vector(r) for r in train_rows], dtype=np.float32)
    y_train = np.array([r["matched"] for r in train_rows], dtype=int)
    buckets = collections.Counter(ra.bucket(r["end"] - r["start"] + 1) for r in train_rows if r["matched"])
    max_bucket = max(buckets.values()) if buckets else 1
    weights = np.array([
        20.0 * max_bucket / buckets[ra.bucket(r["end"] - r["start"] + 1)] if r["matched"] else 1.0
        for r in train_rows
    ])
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight=None, random_state=0)
    clf.fit(scaler.transform(X_train), y_train, sample_weight=weights)
    X_all = scaler.transform(np.array([vector(r) for r in rows], dtype=np.float32))
    scores = clf.predict_proba(X_all)[:, 1]
    by_pair = collections.defaultdict(list)
    for row, score in zip(rows, scores):
        row["score"] = float(score)
        by_pair[row["pid"]].append(row)

    def predict(pair):
        pid = pair["pair_id"]
        if not flags.get(pid):
            return []
        answer = pair.get("model_output") or ""
        mode = by_pair[pid][0]["mode"] if by_pair[pid] else 0.0
        local_mode = mode >= W.get("mode_cut", 0.6)
        items = [r for r in by_pair[pid] if not (local_mode and r["start"] == 0 and r["end"] == len(answer) - 1)]
        items.sort(key=lambda r: -r["score"])
        out = []
        for row in items:
            if row["score"] < report["chosen_threshold"]:
                break
            if any(iou(row["start"], row["end"], o["start"], o["end"]) > 0.1 for o in out):
                continue
            out.append({
                "start": row["start"], "end": row["end"],
                "risk": row["score"], "type": "unsupported_addition", "confidence": 1.0,
            })
            if len(out) >= report["chosen_max_spans"]:
                break
        return out

    val_pairs = [byid[pid] for pid in ids["val"]]
    ex_pairs = [byid[pid] for pid in ids["mfava"]]
    sweep = {}
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for mx in (1, 2, 3, 5):
            report["chosen_threshold"] = th
            report["chosen_max_spans"] = mx
            mv = metrics(val_pairs, predict)
            me = metrics(ex_pairs, predict)
            sweep[f"th{th}_max{mx}"] = {"val": mv, "mfava": me}
    report.pop("chosen_threshold")
    report.pop("chosen_max_spans")
    report["sweep"] = sweep
    best = sorted(sweep.items(), key=lambda kv: (
        kv[1]["val"]["span_level"]["f1"] * 6 + kv[1]["mfava"]["span_level"]["f1"]
    ), reverse=True)
    report["best_by_weighted_span"] = {"config": best[0][0], "val": best[0][1]["val"], "mfava": best[0][1]["mfava"]}
    print("best", best[0][0], best[0][1]["val"]["span_level"]["f1"], best[0][1]["mfava"]["span_level"]["f1"], flush=True)
    with open(ROOT / "artifacts/conditional_scores_v5.pkl", "wb") as handle:
        pickle.dump({
            "flags": flags,
            "scores": {pid: [(r["start"], r["end"], r["score"], r["mode"]) for r in items]
                       for pid, items in by_pair.items()},
        }, handle)
    target = ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, target.stat().st_size, "sec", round(time.time() - started, 1))


if __name__ == "__main__":
    main()
