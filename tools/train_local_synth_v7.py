"""Обучение локального boundary-скоринга на синтетических мутациях v7.

Новая гипотеза: не ранжировать все 250–800 кандидатов, а учиться на
синтетических парах с известной мутацией, где позитивный кандидат — именно
вставленный/заменённый короткий фрагмент. На выводе кандидаты генерируются
только вокруг trigger-токенов (числа/даты/сущности/отрицания/глаголы/низкий
doc support). Pair gate baseline замораживается.
"""
import argparse
import collections
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import STOPWORDS, document_context, stem, tokens

OUT = ROOT / "artifacts/local_synth_v7"
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-’'][0-9A-Za-zА-Яа-яЁё]+)*|[^\s]", re.U)
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
CAP_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b")
STARTERS = {"как", "в", "это", "при", "по", "на", "так", "но", "и", "о", "с", "для", "если", "такой", "также", "однако", "например"}
VERB_WORDS = {"увеличилась", "увеличился", "вырос", "выросла", "поддержал", "поддержала", "подтвердил", "помог", "признал", "соответствует", "имеет", "является", "начал", "началась"}
NEG_WORDS = {"не", "нет", "ни", "никогда", "без"}


def extra_features(text, start, end, answer, doc, dtok, dstems):
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    unique = {t for t in toks if t not in STOPWORDS}
    support = sum(1 for t in unique if t in dtok or stem(t) in dstems) / max(1, len(unique))
    context = answer[max(0, start - 60):min(len(answer), end + 61)].lower()
    ctx_tokens = [t.lower() for t in TOKEN_RE.findall(context)]
    ctx_support = sum(1 for t in ctx_tokens if t in dtok or stem(t) in dstems) / max(1, len(ctx_tokens))
    has_num = bool(NUM_RE.search(text))
    nums = NUM_RE.findall(text)
    num_unsupported = sum(1 for n in nums if n not in doc) / max(1, len(nums))
    caps = [t for t in TOKEN_RE.findall(text) if t[:1].isupper()]
    entity_unsupported = sum(1 for t in caps if t.lower() not in STARTERS and t.lower() not in doc.lower()) / max(1, len(caps))
    return [
        support, ctx_support, float(has_num), num_unsupported, entity_unsupported,
        float(any(t in NEG_WORDS for t in toks)),
        float(any(t in VERB_WORDS for t in toks)),
        float(text.strip().lower() in (doc or "").lower()),
        np.log1p(end - start + 1),
        (start + end) / max(1, 2 * len(answer)),
    ]


def trigger_positions(answer, doc, dtok, dstems, limit=24):
    scored = []
    for match in TOKEN_RE.finditer(answer or ""):
        token = match.group(0)
        low = token.lower()
        score = 0.0
        if YEAR_RE.fullmatch(token) or NUM_RE.fullmatch(token):
            score += 2.0
        if low in VERB_WORDS:
            score += 1.5
        if low in NEG_WORDS:
            score += 1.0
        if CAP_RE.fullmatch(token) and match.start() > 0 and low not in STARTERS:
            score += 1.0
        if low not in dtok and stem(low) not in dstems and len(token) > 3:
            score += 0.8
        if score > 0:
            scored.append((score, match.start(), match.end() - 1))
    scored.sort(reverse=True)
    return [(s, e) for _, s, e in scored[:limit]]


def local_candidates(answer, doc, dtok, dstems, focus, rng=None):
    out = {}
    matches = list(TOKEN_RE.finditer(answer or ""))
    all_cands = candidates_v4(answer)
    for center_start, center_end in focus:
        center = (center_start + center_end) // 2
        for index, match in enumerate(matches):
            if match.start() > center + 120 or match.end() < center - 120:
                continue
            for size in range(1, 6):
                if index + size > len(matches):
                    continue
                start = match.start()
                end = matches[index + size - 1].end() - 1
                if end < center - 80 or start > center + 80:
                    continue
                fragment = answer[start:end + 1].strip()
                if not (2 <= len(fragment) <= 100):
                    continue
                out[(start, end)] = fragment
        for start, end, fragment in all_cands:
            if start <= center <= end and end - start + 1 <= 160:
                out[(start, end)] = fragment
    if rng and len(out) < 8:
        for start, end, fragment in rng.sample(all_cands, min(8, len(all_cands))):
            out[(start, end)] = fragment
    return [(s, e, f) for (s, e), f in sorted(out.items())]


def build_rows(pairs, rng):
    rows = []
    started = time.time()
    for pair in pairs:
        answer = pair.get("model_output") or ""
        doc = pair.get("document_text") or ""
        dtok, dstems, _ = document_context(doc)
        gold = pair.get("spans") or []
        focus = []
        for g in gold:
            focus.append((max(0, g["start"] - 40), min(len(answer) - 1, g["end"] + 40)))
        if not focus:
            focus = trigger_positions(answer, doc, dtok, dstems, 4)
        if not focus:
            continue
        for start, end, fragment in local_candidates(answer, doc, dtok, dstems, focus, rng):
            x = ranker.features(fragment, start, end, answer, doc, dtok, dstems) + extra_features(
                fragment, start, end, answer, doc, dtok, dstems)
            y = int(any(iou(start, end, g["start"], g["end"]) >= 0.5 for g in gold))
            rows.append((pair["pair_id"], start, end, x, y))
    print("rows", len(rows), "pos", sum(r[4] for r in rows), "sec", round(time.time() - started, 1), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/local_synth_v7/report.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    synth = read_jsonl(str(ROOT / "artifacts/synthetic_short_v7.jsonl"))
    real_train = load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
    train = [p for p in synth if p.get("spans")] + real_train
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    rows = build_rows(train, rng)
    X = np.array([r[3] for r in rows], dtype=np.float32)
    y = np.array([r[4] for r in rows], dtype=int)
    model = HistGradientBoostingClassifier(
        max_depth=4, max_leaf_nodes=31, max_iter=180, learning_rate=0.08,
        random_state=7, early_stopping=False,
    ).fit(X, y, sample_weight=np.where(y == 1, 20.0, 1.0))
    with open(OUT / "model.pkl", "wb") as handle:
        pickle.dump(model, handle)
    flags = pickle.load(open(ROOT / "artifacts/baseline_flags_v5.pkl", "rb"))
    W = ranker._model()
    by_pair_data = {}
    for pair in val + ex:
        answer = pair.get("model_output") or ""
        doc = pair.get("document_text") or ""
        dtok, dstems, _ = document_context(doc)
        focus = trigger_positions(answer, doc, dtok, dstems)
        cands = local_candidates(answer, doc, dtok, dstems, focus)
        feats = np.array([
            ranker.features(f, s, e, answer, doc, dtok, dstems) + extra_features(f, s, e, answer, doc, dtok, dstems)
            for s, e, f in cands
        ], dtype=np.float32) if cands else np.zeros((0, X.shape[1]), dtype=np.float32)
        probs = model.predict_proba(feats)[:, 1] if len(feats) else []
        by_pair_data[pair["pair_id"]] = list(zip(
            [(s, e) for s, e, _ in cands], [float(p) for p in probs]
        ))
    def predict(pair, th, max_spans, nms):
        pid = pair["pair_id"]
        if not flags.get(pid):
            return []
        items = sorted(by_pair_data.get(pid, []), key=lambda x: -x[1])
        out = []
        for (s, e), score in items:
            if score < th:
                break
            if any(iou(s, e, o["start"], o["end"]) > nms for o in out):
                continue
            out.append({"start": s, "end": e, "risk": score,
                        "type": "unsupported_addition", "confidence": 1.0})
            if len(out) >= max_spans:
                break
        return out
    sweep = {}
    for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
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
        "rule": "synthetic mutation supervision + trigger-local candidates + frozen gate",
        "train_rows": len(rows), "train_pos": int(y.sum()),
        "synthetic_sha": __import__("hashlib").sha256((ROOT / "artifacts/synthetic_short_v7.jsonl").read_bytes()).hexdigest(),
        "manifest_sha": __import__("hashlib").sha256((ROOT / "artifacts/synthetic_short_v7_manifest.json").read_bytes()).hexdigest(),
        "sweep": sweep,
        "best": [{"config": k, **v} for k, v in best[:10]],
    }
    for key, item in best[:5]:
        print(key, item["val"]["span_level"]["f1"], item["mfava"]["span_level"]["f1"],
              item["val"]["token_level"]["f1"], item["val"]["fpr"], flush=True)
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", args.report)


if __name__ == "__main__":
    main()
