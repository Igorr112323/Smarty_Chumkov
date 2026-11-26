"""V10: pairwise ranker с hard negatives по границам спанов.

Позитив — oracle-совпадающий кандидат; негативы — те же границы, сдвинутые
на 1–3 токена, расширенные/обрезанные в том же предложении, а также лучшие по
baseline кандидаты с IoU<0.5. Обучение только на train RusHallu + mFAVA v3.
Вearly stopping/подборе порогов участвуют val/mFAVA, held-out не читается.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans, document_context, stem, tokens

W = json.loads((ROOT / "artifacts/ranker_weights_v3.json").read_text(encoding="utf-8"))
ranker._WEIGHTS = W
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-’'][0-9A-Za-zА-Яа-яЁё]+)*|[^\s]", re.U)
CUE_RE = re.compile(r"(не|нет|ни|без|никогда|отрица|противореч|опроверг|отверг)", re.I)


def extra_features(text, start, end, answer, document, doc_tokens):
    length = end - start + 1
    before = answer[max(0, start - 30):start]
    after = answer[end + 1:end + 31]
    words = [w.lower() for w in tokens(text) if len(w) > 2]
    support = sum(1 for w in words if w in doc_tokens or stem(w) in doc_tokens) / max(1, len(words))
    first = text[:1]
    last = text[-1:]
    return [
        support,
        float(bool(before and before[-1] in "\n:;—–.-–")),
        float(bool(after and after[0] in "\n:;—–,.-–")),
        float(bool(re.match(r"^\s*[*•\-\d]", text))),
        float(start / max(1, len(answer))),
        float((len(answer) - end - 1) / max(1, len(answer))),
        float(length / max(1, len(answer))),
        float(len(words) / max(1, length)),
        float(bool(CUE_RE.search(text))),
        float(sum(1 for w in words if w not in doc_tokens and not any(ch.isdigit() for ch in w)) / max(1, len(words))),
        float(bool(re.search(r"\d", text))),
        float(len(before) / 30.0),
        float(len(after) / 30.0),
    ]


def feature_vector(pair, start, end, cache):
    key = (id(pair), start, end)
    if key in cache:
        return cache[key]
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    context = document_context(document)
    text = answer[start:end + 1]
    base = ranker.features(text, start, end, answer, document, context[0], context[2])
    value = np.array(base + extra_features(text, start, end, answer, document, context[0]), dtype=np.float32)
    cache[key] = value
    return value


def variants(pair, gold):
    answer = pair.get("model_output") or ""
    positions = [m for m in TOKEN_RE.finditer(answer)]
    if not positions:
        return []
    out = set()
    for delta in (-3, -2, -1, 1, 2, 3):
        left = [i for i, m in enumerate(positions) if m.start() <= gold["start"] <= m.end() - 1]
        right = [i for i, m in enumerate(positions) if m.start() <= gold["end"] <= m.end() - 1]
        li = left[0] if left else 0
        ri = right[0] if right else len(positions) - 1
        a = positions[max(0, li + delta)].start()
        b = positions[min(len(positions) - 1, ri + delta)].end() - 1
        if 0 <= a <= b < len(answer):
            out.add((a, b))
    width = gold["end"] - gold["start"] + 1
    for scale in (0.5, 0.75, 1.25, 1.5, 2.0):
        new_width = max(1, min(len(answer) - 1, int(width * scale)))
        a = max(0, gold["start"] - (new_width - width) // 2)
        b = min(len(answer) - 1, a + new_width - 1)
        out.add((a, b))
    return [(a, b) for a, b in out if iou(a, b, gold["start"], gold["end"]) < 0.999]


def candidate_rows(pair, cache):
    answer = pair.get("model_output") or ""
    context = document_context(pair.get("document_text") or "")
    pool = list(candidate_spans(answer))
    rows = []
    for start, end, text in pool:
        x = ranker.features(text, start, end, answer, pair.get("document_text") or "", context[0], context[2])
        rows.append((ranker._logit(x, W["cand"], ranker.FEATURE_COUNT), start, end))
    rows.sort(reverse=True)
    return rows[:25]


def build_pairs(pairs, cache):
    diffs = []
    labels = []
    started = time.time()
    positives = negatives = 0
    for index, pair in enumerate(pairs):
        gold_spans = pair.get("spans") or []
        if not gold_spans:
            continue
        hard = []
        for gold in gold_spans:
            hard.extend(variants(pair, gold))
        for score, start, end in candidate_rows(pair, cache):
            if not any(iou(start, end, g["start"], g["end"]) >= 0.5 for g in gold_spans):
                hard.append((start, end))
        hard = list(dict.fromkeys(hard))[:60]
        if not hard:
            continue
        for gold in gold_spans:
            pos_vector = feature_vector(pair, gold["start"], gold["end"], cache)
            for start, end in hard:
                if iou(start, end, gold["start"], gold["end"]) >= 0.5:
                    continue
                neg_vector = feature_vector(pair, start, end, cache)
                diffs.append(pos_vector - neg_vector)
                labels.append(1)
                diffs.append(neg_vector - pos_vector)
                labels.append(0)
                positives += 1
        if (index + 1) % 200 == 0:
            print("pairs", index + 1, "diff rows", len(diffs), "sec", round(time.time() - started, 1), flush=True)
    print("positives", positives, "negatives", len(diffs) - positives, flush=True)
    return np.array(diffs, dtype=np.float32), np.array(labels, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/pairwise_weights_v10.json")
    parser.add_argument("--report", default="artifacts/pairwise_train_v10.json")
    args = parser.parse_args()
    train = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "train")
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    cache = {}
    X, y = build_pairs(train + mfava, cache)
    scale = StandardScaler().fit(X)
    model = LogisticRegression(C=0.05, max_iter=2000, random_state=10, solver="lbfgs").fit(scale.transform(X), y)
    weights = {
        "w": [round(float(value), 8) for value in model.coef_[0]],
        "b": round(float(model.intercept_[0]), 8),
        "mu": [round(float(value), 8) for value in scale.mean_],
        "sd": [round(float(value), 8) for value in scale.scale_],
        "feature_count": len(X[0]),
        "seed": 10,
    }
    target = ROOT / args.out
    target.write_text(json.dumps(weights, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "rule": "pairwise logistic on (gold - hard_negative) feature differences",
        "train_pairs": len(train) + len(mfava),
        "diff_rows": len(X), "positive_rows": int(y.sum()),
        "weights_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
