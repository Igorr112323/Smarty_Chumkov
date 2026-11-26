"""Hard-negative global head v5 (итерация после embedding-rank).

Отличия от отклонённых подходов:
  - обучаются только базовые кандидаты (не все n-граммы);
  - negatives с 0.2 <= IoU < 0.5 (boundary near-miss) получают повышенный вес;
  - pair gate и mode selector замораживаются, local-ветка не трогается;
  - пороги выбираются на val+mFAVA-train, held-out не используется.
"""
import collections
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans, document_context

W = ranker._model()
CUT = W.get("mode_cut", 0.6)


def build(rows_pairs, mode_map):
    rows = []
    started = time.time()
    for pair in rows_pairs:
        pid = pair["pair_id"]
        answer = pair.get("model_output") or ""
        doc = pair.get("document_text") or ""
        dtok, dstems, dgrams = document_context(doc)
        for start, end, text in candidate_spans(answer):
            x = ranker.features(text, start, end, answer, doc, dtok, dgrams)
            overlaps = [iou(start, end, g["start"], g["end"]) for g in (pair.get("spans") or [])]
            best = max(overlaps) if overlaps else 0.0
            rows.append({
                "pid": pid, "start": start, "end": end, "features": x,
                "y": int(best >= 0.5), "hard": int(0.2 <= best < 0.5),
                "mode": mode_map.get(pid, 0.0),
            })
    print("rows", len(rows), "pos", sum(r["y"] for r in rows),
          "hard", sum(r["hard"] for r in rows), "sec", round(time.time() - started, 1), flush=True)
    return rows


def main():
    extra = pickle.load(open(ROOT / "artifacts/conditional_scores_v5.pkl", "rb"))
    flags = extra["flags"]
    mode_map = {}
    for pid, items in extra["scores"].items():
        if items:
            mode_map[pid] = items[0][3]
    train = load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    byid = {p["pair_id"]: p for p in train + val + ex}
    ids = {
        "train": {p["pair_id"] for p in train},
        "val": {p["pair_id"] for p in val},
        "mfava": {p["pair_id"] for p in ex},
    }
    rows = build(train + val + ex, mode_map)
    by_pair = collections.defaultdict(list)
    for row in rows:
        by_pair[row["pid"]].append(row)

    tr_rows = [r for r in rows if r["pid"] in ids["train"]]
    X = np.array([r["features"] for r in tr_rows], dtype=np.float32)
    y = np.array([r["y"] for r in tr_rows], dtype=int)
    weights = np.array([
        20.0 if r["y"] else (6.0 if r["hard"] else 1.0)
        for r in tr_rows
    ], dtype=np.float32)
    model = HistGradientBoostingClassifier(
        max_depth=4, max_leaf_nodes=31, max_iter=150,
        learning_rate=0.08, random_state=0, early_stopping=False,
    ).fit(X, y, sample_weight=weights)
    for pid, items in by_pair.items():
        probs = model.predict_proba(np.array([r["features"] for r in items], dtype=np.float32))[:, 1]
        for row, probability in zip(items, probs):
            row["score"] = float(probability)

    baseline = {}
    for p in train + val + ex:
        baseline[p["pair_id"]] = ranker.select_spans(p.get("model_output") or "", p.get("document_text") or "")

    def predict(pair, th, max_spans):
        pid = pair["pair_id"]
        if not flags.get(pid):
            return []
        if mode_map.get(pid, 0.0) >= CUT:
            return baseline[pid]
        items = by_pair.get(pid, [])
        items = sorted(items, key=lambda r: -r["score"])
        out = []
        for row in items:
            if row["score"] < th:
                break
            if any(iou(row["start"], row["end"], o["start"], o["end"]) > 0.1 for o in out):
                continue
            out.append({
                "start": row["start"], "end": row["end"],
                "risk": row["score"], "type": "unsupported_addition", "confidence": 1.0,
            })
            if len(out) >= max_spans:
                break
        return out

    sweep = {}
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        for mx in (1, 2, 3):
            sweep[f"th{th}_max{mx}"] = {
                "val": metrics(val, lambda p: predict(p, th, mx)),
                "mfava": metrics(ex, lambda p: predict(p, th, mx)),
            }
    report = {
        "rule": "hard negatives IoU 0.2-0.5, frozen gate/mode, val+mFAVA selection",
        "train_rows": len(tr_rows), "hard_negatives": sum(r["hard"] for r in tr_rows),
        "baseline": {
            "val": metrics(val, lambda p: baseline[p["pair_id"]]),
            "mfava": metrics(ex, lambda p: baseline[p["pair_id"]]),
        },
        "sweep": sweep,
    }
    best = sorted(sweep.items(), key=lambda kv: (
        kv[1]["val"]["span_level"]["f1"] * 6 + kv[1]["mfava"]["span_level"]["f1"]
    ), reverse=True)
    report["best"] = [{"config": k, **v} for k, v in best[:5]]
    for key, item in best[:5]:
        print(key, item["val"]["span_level"]["f1"], item["mfava"]["span_level"]["f1"], item["val"]["fpr"])
    target = ROOT / "artifacts/hard_negative_v5.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, target.stat().st_size)


if __name__ == "__main__":
    main()
