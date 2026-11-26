"""Constrained decoding для QA start/end модели v6 (val/mFAVA-train только).

Отличается от наивного top-start×top-end: length prior из train, явное
ограничение start<=end и пороги по start/end отдельно. Pair gate заморожен.
"""
import collections
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from train_qa_span_v6 import QASpan, MODEL_DIR, OUT, predict_pair, WINDOW, STEP, DOC_LIMIT, MAX_LEN


def length_prior(pairs):
    lengths = [g["end"] - g["start"] + 1 for p in pairs for g in (p.get("spans") or [])]
    buckets = collections.Counter(
        "s" if L < 20 else "m" if L < 40 else "l" if L < 80 else "xl"
        for L in lengths
    )
    total = sum(buckets.values())
    return {k: (v + 1) / (total + 4) for k, v in buckets.items()}, lengths


def bucket(L):
    return "s" if L < 20 else "m" if L < 40 else "l" if L < 80 else "xl"


def decode(pair, data, flags, prior, start_th, end_th, top, alpha, max_len, nms, max_spans):
    if not flags.get(pair["pair_id"]):
        return []
    starts, ends = data[pair["pair_id"]]
    starts = [x for x in starts if x[2] >= start_th]
    ends = [x for x in ends if x[2] >= end_th]
    starts.sort(key=lambda x: -x[2])
    ends.sort(key=lambda x: -x[2])
    starts = starts[:top]
    ends = ends[:top]
    out = []
    for a, _, ps in starts:
        ranked = []
        for b, _, pe in ends:
            L = b - a + 1
            if L < 5 or L > max_len:
                continue
            score = ps * pe * (prior[bucket(L)] ** alpha)
            ranked.append((score, a, b))
        ranked.sort(reverse=True)
        for score, aa, bb in ranked[:3]:
            if any(iou(aa, bb, o["start"], o["end"]) > nms for o in out):
                continue
            out.append({
                "start": aa, "end": bb, "risk": score,
                "type": "unsupported_addition", "confidence": 1.0,
            })
            if len(out) >= max_spans:
                break
        if len(out) >= max_spans:
            break
    return out


def main():
    train = load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    flags = pickle.load(open(ROOT / "artifacts/baseline_flags_v5.pkl", "rb"))
    prior, lengths = length_prior(train)
    device = torch.device("cpu")
    head = torch.load(str(OUT / "head.pt"), map_location=device)
    model = QASpan(AutoModel.from_pretrained(str(MODEL_DIR))).to(device)
    model.load_state_dict(head, strict=False)
    data = {}
    for p in val + ex:
        data[p["pair_id"]] = predict_pair(model, p, device)
    print("predicted", len(data), "median train len", int(np.median(lengths)), flush=True)
    sweep = {}
    for start_th in (0.02, 0.05, 0.1):
        for end_th in (0.02, 0.05, 0.1):
            for top in (10, 30):
                for alpha in (0, 1, 3):
                    for max_len in (200, 800):
                        for max_spans in (1, 3, 5):
                            def predict(pair):
                                return decode(pair, data, flags, prior, start_th,
                                              end_th, top, alpha, max_len, 0.1, max_spans)
                            mv = metrics(val, predict)
                            me = metrics(ex, predict)
                            key = f"st{start_th}_et{end_th}_top{top}_a{alpha}_len{max_len}_n{max_spans}"
                            sweep[key] = {"val": mv, "mfava": me}
    best = sorted(sweep.items(), key=lambda kv: (
        kv[1]["val"]["span_level"]["f1"] * 6 + kv[1]["mfava"]["span_level"]["f1"],
        kv[1]["val"]["token_level"]["f1"] * 6 + kv[1]["mfava"]["token_level"]["f1"],
    ), reverse=True)
    report = {
        "rule": "QA constrained decoding; frozen gate; val+mFAVA selection only",
        "checkpoint": str(OUT / "head.pt"),
        "prior": prior,
        "best": [{"config": k, **v} for k, v in best[:10]],
        "all": sweep,
    }
    for key, item in best[:5]:
        print(key, item["val"]["span_level"]["f1"], item["mfava"]["span_level"]["f1"],
              item["val"]["fpr"], flush=True)
    target = ROOT / "artifacts/qa_span_v6/decode_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, target.stat().st_size)


if __name__ == "__main__":
    main()
