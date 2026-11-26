"""Быстрый воспроизводимый sweep декодера с кэшем признаков кандидатов.

Эквивалентен ranker.select_spans с явными global-параметрами, но пересчитывает
признаки один раз на пару. Только dev-данные: val RusHallu + mFAVA-train.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans, document_context, predict_type


def cache(pairs, weights):
    result = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        context = document_context(document)
        scored = []
        for start, end, fragment in candidate_spans(answer):
            x = ranker.features(fragment, start, end, answer, document, context[0], context[2])
            risk = ranker._logit(x, weights["cand"], ranker.FEATURE_COUNT)
            scored.append({"start": start, "end": end, "text": fragment, "features": x, "risk": risk,
                           "type": predict_type(fragment, 0.0)})
        gate = ranker._logit(
            ranker.pair_features(scored, [item["risk"] for item in scored], answer),
            weights["pair"], ranker.PAIR_FEATURE_COUNT,
        )
        result.append((pair, scored, gate))
    return result


def decode(entry, gate, threshold, max_spans, nms_iou):
    pair, scored, gate_probability = entry
    if gate_probability < gate:
        return []
    chosen = []
    for item in sorted(scored, key=lambda row: (-row["risk"], row["start"])):
        if item["risk"] < threshold or len(chosen) >= max_spans:
            break
        if any(iou(item["start"], item["end"], other["start"], other["end"]) > nms_iou for other in chosen):
            continue
        chosen.append(item)
    return [{"start": x["start"], "end": x["end"], "type": x["type"]} for x in chosen]


def main():
    weights_path = ROOT / "spanver/ranker_weights.json"
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    extra = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    entries = cache(val + extra, weights)
    by_id = {id(pair): entry for entry, pair in [(e, e[0]) for e in entries]}
    grid = list(itertools.product(
        (0.2, 0.3, 0.4, 0.5, 0.6),
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        (1, 2, 3, 5, 8),
        (0.0, 0.1, 0.3),
    ))
    rows = []
    for gate, threshold, max_spans, nms_iou in grid:
        def predictor(pair, g=gate, t=threshold, m=max_spans, n=nms_iou):
            return decode(by_id[id(pair)], g, t, m, n)
        result = metrics(val + extra, predictor)
        rows.append({
            "gate": gate, "threshold": threshold, "max_spans": max_spans, "nms_iou": nms_iou,
            "span": result["span_level"]["f1"], "token": result["token_level"]["f1"],
            "pair": result["pair_level"]["f1"], "fpr": result["fpr"],
        })
    rows.sort(key=lambda row: (row["span"], row["token"], row["pair"]), reverse=True)
    passing = [row for row in rows
               if row["span"] >= 0.5 and row["token"] >= 0.5 and row["pair"] >= 0.5 and row["fpr"] <= 0.4]
    report = {
        "source": "v2 weights + val RusHallu + mFAVA-train (dev only)",
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "best15": rows[:15],
        "passing_guard_top15": passing[:15],
        "configs": rows,
    }
    target = ROOT / "artifacts/cached_sweep_v8.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("best", rows[0])
    print("passing", len(passing))


if __name__ == "__main__":
    main()
