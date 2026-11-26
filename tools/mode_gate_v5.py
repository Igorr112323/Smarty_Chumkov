"""Performance-labelled mode gate v5.

Отличается от cycle2: метка обучается как argmax per-pair F1 между уже
обученными global/local heads, а не из эвристики «несколько span / не whole».
Все остальные веса и пороги замораживаются; выбор порога — на val+mFAVA-train.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.local_ranker import logistic, predict_trees
from spanver.module import candidate_spans, document_context

W = ranker._model()
LOCAL_CFG = W["local_config"]


def features(pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    dtok, dstems, dgrams = document_context(document)
    scored = []
    for start, end, text in candidate_spans(answer):
        x = ranker.features(text, start, end, answer, document, dtok, dgrams)
        scored.append({"start": start, "end": end, "risk": ranker._logit(x[:37], W["cand"], 37)})
    return ranker.mode_features(scored, answer, document)


def global_pred(pair):
    return ranker.select_spans(
        pair.get("model_output") or "", pair.get("document_text") or "",
        gate=0.5, threshold=0.6, max_spans=1, nms_iou=0.1,
    )


def local_pred(pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    dtok, dstems, dgrams = document_context(document)
    items = []
    for start, end, text in candidate_spans(answer):
        if start == 0 and end == len(answer) - 1:
            continue
        x = ranker.features(text, start, end, answer, document, dtok, dgrams)
        items.append({"start": start, "end": end, "risk": predict_trees(W["local_hgb"], x)})
    out = []
    for item in sorted(items, key=lambda z: -z["risk"]):
        if item["risk"] < LOCAL_CFG["threshold"]:
            break
        if any(iou(item["start"], item["end"], o["start"], o["end"]) > LOCAL_CFG["nms_iou"] for o in out):
            continue
        out.append({
            "start": item["start"], "end": item["end"],
            "risk": item["risk"], "type": "unsupported_addition", "confidence": 1.0,
        })
        if len(out) >= LOCAL_CFG["max_spans"]:
            break
    return out


def pair_score(predicted, gold):
    matched, used, _ = __import__("spanver.evaluate", fromlist=["match_spans"]).match_spans(predicted, gold)
    tp, fp, fn = matched, len(predicted) - matched, len(gold) - matched
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    train = load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    X_train = np.array([features(p) for p in train], dtype=np.float32)
    y_train = []
    for pair in train:
        gold = pair.get("spans") or []
        g = pair_score(global_pred(pair), gold)
        l = pair_score(local_pred(pair), gold)
        y_train.append(int(l > g + 1e-9))
    y_train = np.array(y_train, dtype=int)
    print("train labels", y_train.sum(), "/", len(y_train), flush=True)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0)
    clf.fit(scaler.transform(X_train), y_train)
    train_prob = clf.predict_proba(scaler.transform(X_train))[:, 1]
    val_prob = clf.predict_proba(scaler.transform(np.array([features(p) for p in val], dtype=np.float32)))[:, 1]
    ex_prob = clf.predict_proba(scaler.transform(np.array([features(p) for p in ex], dtype=np.float32)))[:, 1]
    gv, lv = [global_pred(p) for p in val], [local_pred(p) for p in val]
    ge, le = [global_pred(p) for p in ex], [local_pred(p) for p in ex]
    cut_map = {p["pair_id"]: v for p, v in zip(val, val_prob)}
    cut_map.update({p["pair_id"]: v for p, v in zip(ex, ex_prob)})
    sweep = {}
    for cut in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        mv = metrics(val, lambda p: lv[val.index(p)] if cut_map[p["pair_id"]] >= cut else gv[val.index(p)])
        me = metrics(ex, lambda p: le[ex.index(p)] if cut_map[p["pair_id"]] >= cut else ge[ex.index(p)])
        sweep[str(cut)] = {"val": mv, "mfava": me}
    report = {
        "rule": "mode gate trained on per-pair F1 argmax; global/local heads frozen",
        "train_local_labels": int(y_train.sum()),
        "train_prob_auc": None,
        "sweep": sweep,
    }
    best = sorted(sweep.items(), key=lambda kv: (
        kv[1]["val"]["span_level"]["f1"] * 6 + kv[1]["mfava"]["span_level"]["f1"],
        kv[1]["val"]["token_level"]["f1"] * 6 + kv[1]["mfava"]["token_level"]["f1"],
    ), reverse=True)
    report["best"] = [{"cut": k, **v} for k, v in best[:5]]
    for cut, item in best[:5]:
        print(cut, item["val"]["span_level"]["f1"], item["mfava"]["span_level"]["f1"], item["val"]["fpr"])
    target = ROOT / "artifacts/mode_gate_v5.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, target.stat().st_size)


if __name__ == "__main__":
    main()
