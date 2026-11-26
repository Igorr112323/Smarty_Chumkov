"""Оценка pairwise ranker v10 на dev (val + mFAVA-train)."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import document_context
from train_pairwise_v10 import W, feature_vector, extra_features, candidate_rows

PAIR_W = json.loads((ROOT / "artifacts/pairwise_weights_v10.json").read_text(encoding="utf-8"))
ranker._WEIGHTS = W


def score(pair, cache):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    context = document_context(document)
    pool = candidates_v4(answer)
    scored = []
    for start, end, text in pool:
        vector = feature_vector(pair, start, end, cache)
        normalized = (vector - np.array(PAIR_W["mu"])) / np.array(PAIR_W["sd"])
        logit = float(np.dot(normalized, PAIR_W["w"]) + PAIR_W["b"])
        scored.append({"start": start, "end": end, "text": text, "risk": 1 / (1 + math.exp(-logit))})
    base = candidate_rows(pair, {} if False else {})
    baseline = []
    for _score, start, end in base:
        x = ranker.features(answer[start:end + 1], start, end, answer, document, context[0], context[2])
        baseline.append({"start": start, "end": end, "features": x,
                         "risk": ranker._logit(x, W["cand"], ranker.FEATURE_COUNT)})
    gate = ranker._logit(
        ranker.pair_features(baseline, [item["risk"] for item in baseline], answer),
        W["pair"], ranker.PAIR_FEATURE_COUNT,
    )
    return {"ranked": sorted(scored, key=lambda row: -row["risk"]),
            "baseline": sorted(baseline, key=lambda row: -row["risk"]), "gate": gate}


def decode(pair, data, mode, threshold, gate_cut, max_spans, nms_iou=0.10):
    if data["gate"] < gate_cut:
        return []
    scored = data[mode]
    chosen = []
    for item in scored:
        if item["risk"] < threshold or len(chosen) >= max_spans:
            break
        if any(iou(item["start"], item["end"], other["start"], other["end"]) > nms_iou for other in chosen):
            continue
        chosen.append(item)
    return [{"start": item["start"], "end": item["end"], "risk": 1.0,
             "type": "unsupported_addition", "confidence": 1.0} for item in chosen]


def cell(result):
    return {"span": result["span_level"]["f1"], "token": result["token_level"]["f1"],
            "pair": result["pair_level"]["f1"], "fpr": result["fpr"]}


def main():
    val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    cache = {}
    data = {pair["pair_id"]: score(pair, cache) for pair in val + mfava}
    rows = []
    for mode in ("ranked", "baseline"):
        for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for gate_cut in (0.4, 0.5, 0.6):
                for max_spans in (1, 2, 3, 5):
                    def predictor(pair):
                        return decode(pair, data[pair["pair_id"]], mode, threshold, gate_cut, max_spans)
                    combined = metrics(val + mfava, predictor)
                    rows.append({
                        "mode": mode, "threshold": threshold, "gate": gate_cut, "max_spans": max_spans,
                        "combined": cell(combined), "val": cell(metrics(val, predictor)),
                        "mfava": cell(metrics(mfava, predictor)),
                    })
    rows.sort(key=lambda row: (row["combined"]["span"], row["combined"]["token"]), reverse=True)
    passing = [row for row in rows if row["combined"]["span"] >= 0.5 and row["combined"]["token"] >= 0.5
               and row["combined"]["pair"] >= 0.5 and row["combined"]["fpr"] <= 0.4]
    report = {
        "rule": "pairwise v10 weights; pair gate from v3; val+mFAVA dev only",
        "weights_sha256": hashlib.sha256((ROOT / "artifacts/pairwise_weights_v10.json").read_bytes()).hexdigest(),
        "best": rows[:20], "passing": passing[:20], "configs": rows,
    }
    target = ROOT / "artifacts/pairwise_eval_v10.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("best", rows[0], "passing", len(passing), flush=True)
    for row in rows[:5]:
        print("cfg", row["mode"], row["threshold"], row["gate"], row["max_spans"],
              "combined", row["combined"], "val", row["val"], flush=True)


if __name__ == "__main__":
    main()
