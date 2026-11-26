"""V10: whole-answer gate, обученный на train-метках вместо gold-style proxy.

Старый mode gate учился предсказывать `is_local_pair` и на val ошибался во всех
9 cases; whole-кандидат при этом удалялся. Новый gate.predicts has_whole по
признакам ответа/документа, доступным на inference; порог и decoding tunes
только на val+mFAVA. Held-out не читается.
"""
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans, document_context

W = json.loads((ROOT / "artifacts/ranker_weights_v3.json").read_text(encoding="utf-8"))
ranker._WEIGHTS = W
CUE = re.compile(r"(в контексте|в предоставлен|в документ|нет информации|нет прямого ответа|"
                 r"согласно|по данным|например|следующ|перечисл|список)", re.I)


def features(pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    context = document_context(document)
    pool = list(candidate_spans(answer))
    vectors = [ranker.features(text, start, end, answer, document, context[0], context[2])
               for start, end, text in pool]
    risks = [ranker._logit(x, W["cand"], ranker.FEATURE_COUNT) for x in vectors]
    whole = [risk for (start, end, _), risk in zip(pool, risks)
             if start == 0 and end == len(answer) - 1]
    gate = ranker._logit(
        ranker.pair_features([{"features": x} for x in vectors], risks, answer),
        W["pair"], ranker.PAIR_FEATURE_COUNT,
    )
    words = [w for w in answer.split() if len(w) > 2]
    supported = sum(1 for w in words[:60] if w.lower() in context[0]) / max(1, len(words[:60]))
    return [
        float(len(answer)), float(len(document)), float(len(pool)),
        max(risks) if risks else 0.0, float(np.mean(risks)) if risks else 0.0,
        max(whole) if whole else 0.0, float(bool(whole)), gate,
        float(answer.count("\n")), float(len(re.findall(r"[*•]|^\d+\.", answer, re.M))),
        float(len(re.findall(r"\d", answer))), float(len(CUE.findall(answer))),
        supported, float(answer.strip().lower().startswith(("в контексте", "в предоставлен"))),
    ]


def label(pair):
    answer = pair.get("model_output") or ""
    return int(any(g["start"] == 0 and g["end"] == len(answer) - 1 for g in (pair.get("spans") or [])))


def train_gate(train_pairs):
    X = np.array([features(p) for p in train_pairs], dtype=np.float32)
    y = np.array([label(p) for p in train_pairs], dtype=int)
    scale = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=10).fit(scale.transform(X), y)
    return {
        "w": [round(float(v), 8) for v in clf.coef_[0]],
        "b": round(float(clf.intercept_[0]), 8),
        "mu": [round(float(v), 8) for v in scale.mean_],
        "sd": [round(float(v), 8) for v in scale.scale_],
        "n_features": len(X[0]),
        "positive_pairs": int(y.sum()),
        "train_pairs": len(y),
    }


PAIRS = {}
GATE_CACHE = {}


@lru_cache(maxsize=None)
def select_cached(pid, threshold, max_spans):
    pair = PAIRS[pid]
    return ranker.select_spans(pair.get("model_output") or "", pair.get("document_text") or "",
                               threshold=threshold, max_spans=max_spans, nms_iou=0.10)


def gate_prob(pair, params):
    if pair["pair_id"] in GATE_CACHE:
        return GATE_CACHE[pair["pair_id"]]
    x = np.array(features(pair), dtype=np.float64)
    norm = (x - np.array(params["mu"])) / np.array(params["sd"])
    value = float(np.dot(norm, params["w"]) + params["b"])
    result = 1 / (1 + np.exp(-max(-30, min(30, value))))
    GATE_CACHE[pair["pair_id"]] = result
    return result


def decode(pair, params, cut, whole_threshold, strategy, fallback_threshold, max_spans):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    if gate_prob(pair, params) >= cut:
        if strategy == "force":
            return [{"start": 0, "end": len(answer) - 1, "risk": 1.0,
                     "type": "unsupported_addition", "confidence": 1.0}]
        return select_cached(pair["pair_id"], whole_threshold, 1)
    return select_cached(pair["pair_id"], fallback_threshold, max_spans)


def cell(result):
    return {"span": result["span_level"]["f1"], "token": result["token_level"]["f1"],
            "pair": result["pair_level"]["f1"], "fpr": result["fpr"]}


def main():
    train = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "train")
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")
    params = train_gate(train + mfava)
    for pair in val + mfava:
        PAIRS[pair["pair_id"]] = pair
        gate_prob(pair, params)
    rows = []
    for strategy in ("force", "global"):
        for cut in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for whole_threshold in ((0.4,) if strategy == "force" else (0.3, 0.4, 0.5)):
                for fallback_threshold in (0.4, 0.5, 0.6):
                    for max_spans in (1, 3):
                        def predictor(pair):
                            return decode(pair, params, cut, whole_threshold, strategy,
                                          fallback_threshold, max_spans)
                        combined = metrics(val + mfava, predictor)
                        rows.append({
                            "strategy": strategy, "cut": cut, "whole_threshold": whole_threshold,
                            "fallback_threshold": fallback_threshold, "max_spans": max_spans,
                            "combined": cell(combined), "val": cell(metrics(val, predictor)),
                            "mfava": cell(metrics(mfava, predictor)),
                        })
    rows.sort(key=lambda row: (row["combined"]["span"], row["combined"]["token"]), reverse=True)
    passing = [row for row in rows if row["combined"]["span"] >= 0.5 and row["combined"]["token"] >= 0.5
               and row["combined"]["pair"] >= 0.5 and row["combined"]["fpr"] <= 0.4]
    report = {
        "rule": "whole gate trained on RusHallu+mFAVA train labels; dev sweeps only; held not read",
        "params": params, "best": rows[:20], "passing": passing[:20], "configs": rows,
    }
    target = ROOT / "artifacts/whole_gate_v10.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("params positive", params["positive_pairs"], "best", rows[0], "passing", len(passing), flush=True)
    for row in rows[:8]:
        print("cfg", row["strategy"], row["cut"], row["whole_threshold"], row["fallback_threshold"],
              row["max_spans"], "combined", row["combined"], "val", row["val"], flush=True)


if __name__ == "__main__":
    main()
