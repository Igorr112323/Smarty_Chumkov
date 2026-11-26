"""V9b: GBDT-скоринг кандидатов с реальными фразовыми спанами RAGTruth_ru.

Отличается от v7 синтетики: RAGTruth даёт настоящие ошибки в настоящем
контексте с автоматическими точными границами, а не тривиальные подмены
токенов. Held-out документы исключены. Runtime остаётся stdlib: экспорт
деревьев в JSON совместим со spanver/local_ranker.predict_trees.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import document_context
from train_local_ranker import export_trees, predict_trees

NEG_PER_PAIR = 20
ranker._WEIGHTS = json.loads((ROOT / "artifacts/ranker_weights_v3.json").read_text(encoding="utf-8"))


def rows_from(pairs, label):
    rows = []
    started = time.time()
    for index, pair in enumerate(pairs):
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        doc_tokens, _stems, doc_grams = document_context(document)
        gold = pair.get("spans") or []
        pool = candidates_v4(answer)
        # Дешёвый shortlist: support + sentence-кандидаты + точные gold offsets.
        scored_cheap = []
        for start, end, text in pool:
            words = [w.lower() for w in text.split() if len(w) > 2][:20]
            support = sum(w in doc_tokens for w in words) / max(1, len(words))
            scored_cheap.append((support, start, end, text))
        scored_cheap.sort(reverse=True)
        chosen = {(start, end): text for _score, start, end, text in scored_cheap[:60]}
        for start, end, text in pool:
            if (start, end) in chosen and (start, end, text) not in pool:
                continue
        from spanver.module import candidate_spans
        for start, end, text in candidate_spans(answer):
            chosen.setdefault((start, end), text)
        for span in gold:
            start, end = int(span["start"]), int(span["end"])
            chosen[(start, end)] = answer[start:end + 1]
        scored = []
        for (start, end), text in chosen.items():
            x = ranker.features(text, start, end, answer, document, doc_tokens, doc_grams)
            hit = any(iou(start, end, g["start"], g["end"]) >= 0.5 for g in gold)
            score = 10.0 if hit else ranker._logit(x, ranker._WEIGHTS["cand"], ranker.FEATURE_COUNT)
            scored.append((score, start, end, text, hit, x))
        positives = [row for row in scored if row[4]]
        negatives = sorted((row for row in scored if not row[4]), key=lambda row: -row[0])[:NEG_PER_PAIR]
        for row in positives + negatives:
            rows.append({"pid": pair.get("pair_id", f"{label}-{index}"), "source": label,
                         "start": row[1], "end": row[2], "vector": row[5], "label": int(row[4])})
        if (index + 1) % 200 == 0:
            print(label, index + 1, "rows", len(rows), "sec", round(time.time() - started, 1), flush=True)
    return rows


def mode_features_for(pair, scored):
    answer = pair.get("model_output") or ""
    return ranker.mode_features(scored, answer, pair.get("document_text") or "")


def score_candidates(pairs, model):
    result = {}
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        doc_tokens, _stems, doc_grams = document_context(document)
        pool = candidates_v4(answer)
        vectors = [ranker.features(text, start, end, answer, document, doc_tokens, doc_grams)
                   for start, end, text in pool]
        risks = model.predict_proba(np.array(vectors, dtype=np.float32))[:, 1]
        scored = [{"start": start, "end": end, "text": text,
                   "features": x, "risk": float(risk)}
                  for (start, end, text), x, risk in zip(pool, vectors, risks)]
        result[id(pair)] = scored
    return result


def decode(scored, pair, threshold, max_spans):
    answer = pair.get("model_output") or ""
    chosen = []
    for item in sorted(scored, key=lambda row: -row["risk"]):
        if item["risk"] < threshold or len(chosen) >= max_spans:
            break
        if any(iou(item["start"], item["end"], other["start"], other["end"]) > 0.10 for other in chosen):
            continue
        chosen.append(item)
    return [{"start": item["start"], "end": item["end"], "risk": 1.0,
             "type": "unsupported_addition", "confidence": 1.0} for item in chosen]


def cell(result):
    return {key: result[key]["f1"] for key in ("span_level", "token_level", "pair_level")} | {"fpr": result["fpr"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-pairs", type=int, default=1200)
    parser.add_argument("--out", default="artifacts/ranker_weights_rag_aug_v9.json")
    parser.add_argument("--report", default="artifacts/rag_aug_v9_report.json")
    args = parser.parse_args()

    held = [p for p in read_jsonl(str(ROOT / "artifacts/gold1200_v3.jsonl"))
            if p.get("split") in ("test", "extra")]
    held_docs = {hashlib.sha256((p.get("document_text") or "").encode()).hexdigest() for p in held}
    rag = [p for p in read_jsonl(str(ROOT / "Dataset/RAGTruth_ru/ragtruth_ru.jsonl"))
           if hashlib.sha256((p.get("document_text") or "").encode()).hexdigest() not in held_docs]
    train = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "train")
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")
    print("pairs rag", len(rag[:args.rag_pairs]), "train", len(train), "mfava", len(mfava), flush=True)

    rows = rows_from(rag[:args.rag_pairs], "ragtruth") + rows_from(train + mfava, "v3")
    X = np.array([row["vector"] for row in rows], dtype=np.float32)
    y = np.array([row["label"] for row in rows], dtype=int)
    print("rows", len(X), "positive", int(y.sum()), flush=True)
    model = HistGradientBoostingClassifier(
        learning_rate=0.1, max_depth=4, max_leaf_nodes=31, max_iter=150,
        random_state=9, early_stopping=False,
    ).fit(X, y, sample_weight=np.where(y == 1, 6.0, 1.0))
    exported = export_trees(model)
    sample = X[:: max(1, len(X) // 128)]
    expected = model.predict_proba(sample)[:, 1]
    actual = np.array([predict_trees(exported, list(row)) for row in sample])
    diff = float(np.max(np.abs(expected - actual)))
    if diff > 1e-6:
        raise SystemExit(f"export mismatch {diff}")

    weights = dict(ranker._WEIGHTS)
    weights["local_hgb"] = exported
    scores = score_candidates(val + mfava, model)
    best = None
    grid = []
    for threshold in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        for max_spans in (1, 2, 3, 5):
            def predictor(pair):
                return decode(scores[id(pair)], pair, threshold, max_spans)
            combined = metrics(val + mfava, predictor)
            row = {"threshold": threshold, "max_spans": max_spans,
                   "combined": cell(combined), "val": cell(metrics(val, predictor)),
                   "mfava": cell(metrics(mfava, predictor))}
            grid.append(row)
            key = (combined["span_level"]["f1"], combined["token_level"]["f1"], combined["pair_level"]["f1"])
            if combined["fpr"] <= 0.4 and (best is None or key > best[0]):
                best = (key, row)
    if best is None:
        raise SystemExit("нет конфигурации с FPR <= 0,40")
    chosen = best[1]
    weights["local_config"] = {"threshold": chosen["threshold"], "max_spans": chosen["max_spans"], "nms_iou": 0.10}
    target = ROOT / args.out
    target.write_text(json.dumps(weights, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    report = {
        "rule": "RAGTruth_ru phrase spans + v3 train/mFAVA; held docs excluded; dev only",
        "rag_pairs": len(rag[:args.rag_pairs]), "rows": len(X), "positive": int(y.sum()),
        "export_diff": diff, "chosen": chosen, "grid": grid,
        "weights_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(chosen, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
