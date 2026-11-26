"""Ranking analysis v5 для oracle-достижимых gold spans.

Только train/val/mFAVA-train. Held-out не читается.
Считает:
  - ранг правильного кандидата у base/local/active scorer;
  - long/short precision/recall/FPR на уровне кандидатов при порогах;
  - pair-level метрики политики «только короткие кандидаты».

Результат: artifacts/ranking_analysis_v5.json.
"""
import argparse
import collections
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.local_ranker import logistic, predict_trees
from spanver.module import document_context

W = ranker._model()


def bucket(length):
    if length < 20:
        return "<20"
    if length < 40:
        return "20-40"
    if length < 80:
        return "40-80"
    return "80+"


def short(length):
    return length < 40


def build_rows(pairs):
    rows = []
    started = time.time()
    for pair in pairs:
        pid = pair["pair_id"]
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        dtok, dstems, dgrams = document_context(document)
        for start, end, fragment in candidates_v4(answer):
            x = ranker.features(fragment, start, end, answer, document, dtok, dgrams)
            matched = int(any(
                iou(start, end, gold["start"], gold["end"]) >= 0.5
                for gold in (pair.get("spans") or [])
            ))
            rows.append({
                "pid": pid,
                "start": start,
                "end": end,
                "text": fragment,
                "features": x,
                "matched": matched,
            })
    print("rows", len(rows), "pos", sum(r["matched"] for r in rows),
          "sec", round(time.time() - started, 1), flush=True)
    return rows


def load_or_build(pairs, cache):
    if cache and Path(cache).exists():
        with open(cache, "rb") as handle:
            raw = pickle.load(handle)
        if isinstance(raw, tuple) and raw and isinstance(raw[0], list):
            raw = [item for part in raw for item in part]
        pids = {p["pair_id"] for p in pairs}
        rows = []
        for item in raw:
            if item[0] not in pids:
                continue
            rows.append({
                "pid": item[0], "start": item[1], "end": item[2],
                "text": item[3], "features": item[4], "matched": item[5],
            })
        print("loaded cache", len(rows), "rows", flush=True)
        return rows
    return build_rows(pairs)


def scores_for(rows):
    base = [ranker._logit(r["features"][:37], W["cand"], 37) for r in rows]
    local = [predict_trees(W["local_hgb"], r["features"][:37]) for r in rows]
    lengths = {p["pair_id"]: len(p.get("model_output") or "") for p in PAIRS}
    by_pair = collections.defaultdict(list)
    for row, b, l in zip(rows, base, local):
        row["base"] = b
        row["local"] = l
        by_pair[row["pid"]].append(row)
    cut = W.get("mode_cut", 0.6)
    for pid, items in by_pair.items():
        answer_len = lengths.get(pid, 0)
        scored = [{"start": r["start"], "end": r["end"], "risk": r["base"]} for r in items]
        mode = logistic(ranker.mode_features(scored, "x" * answer_len, ""), W["mode"], 11)
        is_local = mode >= cut
        for row in items:
            whole = row["start"] == 0 and row["end"] == answer_len - 1
            row["mode"] = mode
            row["active"] = row["local"] if is_local and not whole else row["base"]
    return by_pair


def rank_stats(rows, key):
    buckets = collections.defaultdict(list)
    for row in rows:
        if not row["matched"]:
            continue
        length = row["end"] - row["start"] + 1
        buckets["all"].append(row)
        buckets[bucket(length)].append(row)
        buckets["short" if short(length) else "long"].append(row)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["pid"]].append(row)
    result = {}
    for name, items in buckets.items():
        ranks = []
        for row in items:
            better = sum(1 for other in grouped[row["pid"]] if other[key] > row[key])
            ranks.append(better + 1)
        ranks.sort()
        result[name] = {
            "gold_candidates": len(ranks),
            "top1": sum(r == 1 for r in ranks),
            "top3": sum(r <= 3 for r in ranks),
            "top5": sum(r <= 5 for r in ranks),
            "top10": sum(r <= 10 for r in ranks),
            "top20": sum(r <= 20 for r in ranks),
            "median_rank": ranks[len(ranks) // 2] if ranks else 0,
            "mrr": round(sum(1 / r for r in ranks) / len(ranks), 4) if ranks else 0.0,
        }
    return result


def threshold_stats(rows, key):
    buckets = {
        "all": rows,
        "short": [r for r in rows if short(r["end"] - r["start"] + 1)],
        "long": [r for r in rows if not short(r["end"] - r["start"] + 1)],
        "<20": [r for r in rows if r["end"] - r["start"] + 1 < 20],
        "20-40": [r for r in rows if 20 <= r["end"] - r["start"] + 1 < 40],
    }
    result = {}
    for name, items in buckets.items():
        total_pos = sum(r["matched"] for r in items)
        total_neg = len(items) - total_pos
        per = {}
        for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            tp = sum(r["matched"] and r[key] >= th for r in items)
            fp = sum(not r["matched"] and r[key] >= th for r in items)
            fn = total_pos - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / total_pos if total_pos else 0.0
            per[str(th)] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "candidate_fpr": round(fp / total_neg, 4) if total_neg else 0.0,
            }
        result[name] = {"candidates": len(items), "gold_candidates": total_pos, "thresholds": per}
    return result


def short_policy(by_pair, pid, th, max_spans):
    items = [r for r in by_pair[pid] if short(r["end"] - r["start"] + 1)]
    items.sort(key=lambda r: -r["active"])
    out = []
    for row in items:
        if row["active"] < th:
            break
        if any(iou(row["start"], row["end"], o["start"], o["end"]) > 0.1 for o in out):
            continue
        out.append({
            "start": row["start"], "end": row["end"],
            "risk": row["active"], "type": "unsupported_addition", "confidence": 1.0,
        })
        if len(out) >= max_spans:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="")
    parser.add_argument("--report", default="artifacts/ranking_analysis_v5.json")
    args = parser.parse_args()
    global PAIRS
    PAIRS = (
        load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
        + load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
        + read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    )
    splits = {
        "train": [p for p in PAIRS if p.get("split") == "train"],
        "val": [p for p in PAIRS if p.get("split") == "val"],
        "mfava_train": [p for p in PAIRS if p["pair_id"].startswith("mfava")],
    }
    rows = load_or_build(PAIRS, args.cache)
    by_pair = scores_for(rows)
    report = {
        "rule": "IoU>=0.5; только train/val/mFAVA-train; active score = baseline mode-selector",
        "splits": {},
        "manifest": {
            "gold_jsonl_sha256": hashlib.sha256((ROOT / "artifacts/gold.jsonl").read_bytes()).hexdigest(),
            "mfava_train_v2_sha256": hashlib.sha256((ROOT / "artifacts/mfava_train_v2.jsonl").read_bytes()).hexdigest(),
            "ranker_weights_sha256": hashlib.sha256((ROOT / "spanver/ranker_weights.json").read_bytes()).hexdigest(),
        },
    }
    for name, pairs in splits.items():
        pids = {p["pair_id"] for p in pairs}
        subset = [r for r in rows if r["pid"] in pids]
        if not subset:
            continue
        report["splits"][name] = {
            "pairs": len(pairs),
            "gold_spans": sum(len(p.get("spans") or []) for p in pairs),
            "candidates": len(subset),
            "rank_base": rank_stats(subset, "base"),
            "rank_local": rank_stats(subset, "local"),
            "rank_active": rank_stats(subset, "active"),
            "thresholds_active": threshold_stats(subset, "active"),
        }
    val_pairs = splits["val"]
    ex_pairs = splits["mfava_train"]
    policies = {}
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        for max_spans in (1, 2, 3):
            def predict(pair):
                return short_policy(by_pair, pair["pair_id"], th, max_spans)
            mv = metrics(val_pairs, predict)
            me = metrics(ex_pairs, predict)
            policies[f"th{th}_max{max_spans}"] = {
                "val": mv, "mfava_train": me,
            }
    report["short_only_policies"] = policies
    target = ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, target.stat().st_size)


if __name__ == "__main__":
    main()
