"""BIOES + Viterbi token tagger v6 (stdlib/sklearn, без RuBERT).

Отличается от v3 token-level: BIOES-метки, Viterbi с переходной матрицей,
замороженный pair gate, подбор max_spans/NMS на val+mFAVA-train.
Веса и отчёт сохраняются в artifacts/bioes_v6/.
"""
import collections
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import document_context, stem

OUT = ROOT / "artifacts/bioes_v6"
TOKEN_RE = re.compile(r"\S+")
LABELS = ["O", "B", "I", "E", "S"]
TRANSITIONS = {("O", "O"), ("O", "B"), ("O", "S"), ("B", "I"), ("B", "E"),
               ("I", "I"), ("I", "E"), ("E", "O"), ("E", "B"), ("E", "S"),
               ("S", "O"), ("S", "B"), ("S", "S")}


def token_features(index, tokens, answer, doc_tokens, doc_stems):
    token = tokens[index]
    low = token.lower()
    stemmed = stem(low)
    before = tokens[index - 1].lower() if index > 0 else ""
    after = tokens[index + 1].lower() if index + 1 < len(tokens) else ""
    supported = low in doc_tokens or stemmed in doc_stems
    before_supported = before in doc_tokens or stem(before) in doc_stems
    after_supported = after in doc_tokens or stem(after) in doc_stems
    return {
        "word": low,
        "prefix": low[:3],
        "suffix": low[-3:],
        "shape": "".join("D" if c.isdigit() else "A" if c.isalpha() else "P" for c in token)[:12],
        "len": min(len(token), 20),
        "position": index / max(1, len(tokens)),
        "supported": int(supported),
        "before_supported": int(before_supported),
        "after_supported": int(after_supported),
        "digits": int(any(c.isdigit() for c in token)),
        "caps": int(token[:1].isupper() and index > 0),
        "punct_before": int(bool(re.search(r"[,;:()—–-]$", before))),
        "punct_after": int(bool(re.search(r"^[,;:()—–-]", after))),
        "neg": int(low in {"не", "нет", "ни", "без", "отрица", "отвергнуты"}),
    }


def gold_labels(pair, tokens, spans):
    labels = ["O"] * len(tokens)
    positions = [(m.start(), m.end() - 1) for m in TOKEN_RE.finditer(pair.get("model_output") or "")]
    for gold in spans:
        cover = [i for i, (s, e) in enumerate(positions) if s <= gold["end"] and e >= gold["start"]]
        if not cover:
            continue
        first, last = cover[0], cover[-1]
        if first == last:
            labels[first] = "S"
        else:
            labels[first] = "B"
            for i in cover[1:-1]:
                labels[i] = "I"
            labels[last] = "E"
    return labels


def build_dataset(pairs):
    rows = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        tokens = TOKEN_RE.findall(answer)
        if not tokens:
            continue
        doc_tokens, doc_stems, _ = document_context(pair.get("document_text") or "")
        labels = gold_labels(pair, tokens, pair.get("spans") or [])
        for index, token in enumerate(tokens):
            rows.append((pair["pair_id"], index, token_features(index, tokens, answer, doc_tokens, doc_stems), labels[index]))
    return rows


def viterbi(scores, transitions, smoothing=0.1):
    n = len(scores)
    if not n:
        return []
    dp = np.full((n, len(LABELS)), -1e9)
    back = np.zeros((n, len(LABELS)), dtype=int)
    dp[0] = scores[0]
    for i in range(1, n):
        for current in range(len(LABELS)):
            best = -1e9
            best_prev = 0
            for previous in range(len(LABELS)):
                trans = transitions.get((LABELS[previous], LABELS[current]), -20.0)
                value = dp[i - 1, previous] + trans + scores[i, current]
                if value > best:
                    best, best_prev = value, previous
            dp[i, current] = best
            back[i, current] = best_prev
    path = [int(np.argmax(dp[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return [LABELS[i] for i in reversed(path)]


def labels_to_spans(pair, labels, min_len, max_len):
    answer = pair.get("model_output") or ""
    positions = [(m.start(), m.end() - 1) for m in TOKEN_RE.finditer(answer)]
    spans = []
    i = 0
    while i < len(labels):
        if labels[i] == "S":
            spans.append((positions[i][0], positions[i][1])); i += 1
        elif labels[i] == "B":
            j = i + 1
            while j < len(labels) and labels[j] == "I":
                j += 1
            if j < len(labels) and labels[j] == "E":
                spans.append((positions[i][0], positions[j][1]))
            i = j + 1
        else:
            i += 1
    return [
        {"start": s, "end": e, "risk": 1.0, "type": "unsupported_addition", "confidence": 1.0}
        for s, e in spans if min_len <= e - s + 1 <= max_len
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    flags = pickle.load(open(ROOT / "artifacts/baseline_flags_v5.pkl", "rb"))
    rows = build_dataset(train)
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform([r[2] for r in rows])
    y = [r[3] for r in rows]
    weights = [8.0 if label != "O" else 1.0 for label in y]
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight=None, random_state=0)
    clf.fit(X, y, sample_weight=weights)
    transitions = collections.Counter()
    for pair_id in {r[0] for r in rows}:
        seq = [r[3] for r in rows if r[0] == pair_id]
        for left, right in zip(seq, seq[1:]):
            transitions[(left, right)] += 1
    counts = {label: sum(1 for r in rows if r[3] == label) for label in LABELS}
    smoothed = {}
    for a in LABELS:
        for b in LABELS:
            if (a, b) in TRANSITIONS:
                smoothed[(a, b)] = math.log((transitions[(a, b)] + 0.5) / (counts[a] + 2.5))
    by_pair = collections.defaultdict(list)
    for row in rows:
        by_pair[row[0]].append(row)
    eval_rows = build_dataset(val + ex)
    eval_by_pair = collections.defaultdict(list)
    for row in eval_rows:
        eval_by_pair[row[0]].append(row)
    proba = clf.predict_proba(vectorizer.transform([r[2] for r in eval_rows]))
    class_index = {name: i for i, name in enumerate(clf.classes_)}
    score_map = {}
    cursor = 0
    for pid, items in eval_by_pair.items():
        matrix = np.array([
            [proba[cursor + i, class_index[label]] if label in class_index else 0.0 for label in LABELS]
            for i in range(len(items))
        ])
        cursor += len(items)
        score_map[pid] = viterbi(np.log(matrix + 1e-9), smoothed)
    sweep = {}
    for min_len in (3, 10):
        for max_len in (80, 400, 1000):
            for max_spans in (1, 3, 5):
                def predict(pair):
                    pid = pair["pair_id"]
                    if not flags.get(pid):
                        return []
                    spans = labels_to_spans(pair, score_map[pid], min_len, max_len)
                    spans.sort(key=lambda s: -s["risk"])
                    return spans[:max_spans]
                mv = metrics(val, predict)
                me = metrics(ex, predict)
                sweep[f"min{min_len}_max{max_len}_n{max_spans}"] = {"val": mv, "mfava": me}
    best = sorted(sweep.items(), key=lambda kv: (
        kv[1]["val"]["span_level"]["f1"] * 6 + kv[1]["mfava"]["span_level"]["f1"]
    ), reverse=True)
    report = {
        "rule": "BIOES+Viterbi; frozen gate; val+mFAVA only",
        "class_counts": counts,
        "best": [{"config": k, **v} for k, v in best[:5]],
        "all": sweep,
    }
    for key, item in best[:5]:
        print(key, item["val"]["span_level"]["f1"], item["mfava"]["span_level"]["f1"], item["val"]["fpr"])
    with open(OUT / "tagger.pkl", "wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "classifier": clf, "transitions": smoothed}, handle)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", OUT / "report.json")


if __name__ == "__main__":
    import math
    main()
