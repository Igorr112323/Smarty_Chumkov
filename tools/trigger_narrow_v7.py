"""Trigger-narrowed decoding v7.

Новая гипотеза: synthetic-обученный trigger-детектор не предсказывает span сам,
а только сужает пространство кандидатов. Внутри узкого окна используется
уже обученный baseline-скоринг. Pair gate заморожен.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.local_ranker import logistic, predict_trees
from spanver.module import candidate_spans, document_context
from train_local_synth_v7 import trigger_positions, local_candidates, extra_features

W = ranker._model()


def mode_probability(pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    dtok, dstems, dgrams = document_context(document)
    scored = []
    for start, end, text in candidate_spans(answer):
        x = ranker.features(text, start, end, answer, document, dtok, dgrams)
        scored.append({"start": start, "end": end, "risk": ranker._logit(x[:37], W["cand"], 37)})
    return logistic(ranker.mode_features(scored, answer, document), W["mode"], 11)


def main():
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    flags = pickle.load(open(ROOT / "artifacts/baseline_flags_v5.pkl", "rb"))
    modes = {p["pair_id"]: mode_probability(p) for p in val + ex}
    scores = {}
    for pair in val + ex:
        pid = pair["pair_id"]
        answer = pair.get("model_output") or ""
        doc = pair.get("document_text") or ""
        dtok, dstems, _ = document_context(doc)
        focus = trigger_positions(answer, doc, dtok, dstems)
        cands = local_candidates(answer, doc, dtok, dstems, focus)
        if not cands:
            scores[pid] = []
            continue
        feats = np.array([
            ranker.features(f, s, e, answer, doc, dtok, dstems)
            for s, e, f in cands
        ], dtype=np.float32)
        if modes[pid] >= W.get("mode_cut", 0.6):
            probs = np.array([predict_trees(W["local_hgb"], row[:37]) for row in feats])
        else:
            probs = np.array([ranker._logit(x[:37], W["cand"], 37) for x in feats])
        scores[pid] = list(zip([(s, e) for s, e, _ in cands], [float(p) for p in probs]))

    def predict(pair, th, mx, nms):
        pid = pair["pair_id"]
        if not flags.get(pid):
            return []
        out = []
        for (s, e), score in sorted(scores[pid], key=lambda x: -x[1]):
            if score < th:
                break
            if any(iou(s, e, o["start"], o["end"]) > nms for o in out):
                continue
            out.append({"start": s, "end": e, "risk": score,
                        "type": "unsupported_addition", "confidence": 1.0})
            if len(out) >= mx:
                break
        return out

    sweep = {}
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        for mx in (1, 2, 3, 5):
            for nms in (0.1, 0.3):
                mv = metrics(val, lambda p: predict(p, th, mx, nms))
                me = metrics(ex, lambda p: predict(p, th, mx, nms))
                sweep[f"th{th}_n{mx}_nms{nms}"] = {"val": mv, "mfava": me}
    best = sorted(sweep.items(), key=lambda kv: (
        kv[1]["val"]["span_level"]["f1"] * 6 + kv[1]["mfava"]["span_level"]["f1"],
        kv[1]["val"]["token_level"]["f1"] * 6 + kv[1]["mfava"]["token_level"]["f1"],
    ), reverse=True)
    report = {
        "rule": "trigger-narrowed v4 candidates + frozen baseline scoring/gate",
        "sweep": sweep,
        "best": [{"config": k, **v} for k, v in best[:10]],
    }
    for key, item in best[:5]:
        print(key, item["val"]["span_level"]["f1"], item["mfava"]["span_level"]["f1"],
              item["val"]["token_level"]["f1"], item["val"]["fpr"], flush=True)
    target = ROOT / "artifacts/trigger_narrow_v7.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, target.stat().st_size)


if __name__ == "__main__":
    main()
