"""V10: pair-relative decoding поверх baseline logits (dev только).

Идея из error analysis: gold-кандидат часто rank 1–20, но с низким абсолютным
risk; absolute threshold его отсекает. Здесь кандидаты оцениваются относительно
лучшего в паре (softmax с температурой), а pair gate остаётся из v3-весов.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans, document_context

W = json.loads((ROOT / "artifacts/ranker_weights_v3.json").read_text(encoding="utf-8"))
ranker._WEIGHTS = W


def cache(pairs):
    result = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        context = document_context(document)
        scored = []
        for start, end, text in candidate_spans(answer):
            x = ranker.features(text, start, end, answer, document, context[0], context[2])
            scored.append({
                "start": start, "end": end, "text": text, "features": x,
                "risk": ranker._logit(x, W["cand"], ranker.FEATURE_COUNT),
            })
        gate = ranker._logit(
            ranker.pair_features(scored, [item["risk"] for item in scored], answer),
            W["pair"], ranker.PAIR_FEATURE_COUNT,
        )
        result.append((pair, scored, gate))
    return result


def decode(entry, temperature, relative, gate_cut, max_spans, nms_iou=0.10):
    pair, scored, gate = entry
    if gate < gate_cut:
        return []
    if not scored:
        return []
    top = max(item["risk"] for item in scored)
    weights = []
    for item in scored:
        value = math.exp(min(30.0, (item["risk"] - top) / temperature))
        weights.append(value)
    total = sum(weights)
    chosen = []
    for item, weight in sorted(zip(scored, weights), key=lambda row: -row[1]):
        score = weight / total
        if score < relative or len(chosen) >= max_spans:
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
    entries = cache(val + mfava)
    by_id = {id(pair): entry for entry, pair, _ in [(e, e[0], e[2]) for e in entries]}
    rows = []
    for temperature in (0.02, 0.05, 0.10, 0.20):
        for relative in (0.01, 0.02, 0.05, 0.10, 0.20):
            for gate_cut in (0.35, 0.45, 0.50, 0.60):
                for max_spans in (1, 2, 3, 5):
                    def predictor(pair):
                        return decode(by_id[id(pair)], temperature, relative, gate_cut, max_spans)
                    combined = metrics(val + mfava, predictor)
                    rows.append({
                        "temperature": temperature, "relative": relative, "gate": gate_cut,
                        "max_spans": max_spans,
                        "combined": cell(combined), "val": cell(metrics(val, predictor)),
                        "mfava": cell(metrics(mfava, predictor)),
                    })
    rows.sort(key=lambda row: (row["combined"]["span"], row["combined"]["token"]), reverse=True)
    passing = [row for row in rows if row["combined"]["span"] >= 0.5 and row["combined"]["token"] >= 0.5
               and row["combined"]["pair"] >= 0.5 and row["combined"]["fpr"] <= 0.4]
    report = {
        "rule": "relative softmax over candidate logits; v3 weights; val+mFAVA dev only",
        "weights_sha256": hashlib.sha256((ROOT / "artifacts/ranker_weights_v3.json").read_bytes()).hexdigest(),
        "best": rows[:20], "passing": passing[:20], "configs": rows,
    }
    target = ROOT / "artifacts/relative_decode_v10.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("best", rows[0], "passing", len(passing), flush=True)
    for row in rows[:5]:
        print("cfg", row["temperature"], row["relative"], row["gate"], row["max_spans"],
              "combined", row["combined"], "val", row["val"], flush=True)


if __name__ == "__main__":
    main()
