"""V11 блок 2.3: генератор вариантов границы ±3 слова + обученный классификатор.

Для выбранного baseline-декодером спана строятся варианты: сдвиг левой/правой
границы на 1..3 слова, привязка к пунктуации/союзу, сжатие/расширение на clause.
Классификатор (train-only) предсказывает, улучшит ли вариант IoU с gold; на
инференсе берётся лучший вариант с вероятностью выше порога.

Held-out не читается.
"""
import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import v11_lab as L
import v11_ranker as R
from spanver.evaluate import iou, metrics, read_jsonl
from spanver.module import candidate_spans

BOUNDARY = re.compile(r"[,;:!?.\n]")


def variants(answer, start, end):
    """Все варианты границ вокруг спана: сдвиги словами и привязка к пунктуации."""
    out = {}
    left_words = list(re.finditer(r"\S+", answer[:start]))
    right_words = list(re.finditer(r"\S+", answer[end + 1:]))
    offset = len(answer) - len(answer[end + 1:])
    for step in (0, 1, 2, 3):
        new_start = start
        if step and left_words:
            new_start = left_words[max(0, len(left_words) - step)].start()
        for right_step in (0, 1, 2, 3):
            new_end = end
            if right_step and right_words:
                new_end = offset + right_words[min(len(right_words) - 1, right_step - 1)].end() - 1
            if new_start < new_end:
                out[(new_start, new_end)] = "shift"
    for match in BOUNDARY.finditer(answer, max(0, start - 4), min(len(answer), end + 5)):
        if match.start() <= start:
            key = (match.end(), end)
            if match.end() < end:
                out[key] = "left_punct"
        if match.start() >= end:
            key = (start, match.start() - 1)
            if match.start() - 1 > start:
                out[key] = "right_punct"
    return out


def features(answer, document, start, end, base_start, base_end):
    text = answer[start:end + 1]
    support = L.support_features(text, document)
    old = L.support_features(answer[base_start:base_end + 1], document)
    left_ok, right_ok = L.clause_boundaries(answer, start, end)
    words = L.WORD_RE.findall(text)
    return [
        support["token_cov"], support["cos"], support["token_cov"] - old["token_cov"],
        support["cos"] - old["cos"], left_ok, right_ok, math.log1p(len(text)),
        len(text) / max(1, len(answer)), abstract_punct(answer, start, end),
        float(support["verbatim"]), support["numbers_missing"], support["names_missing"],
        (end - start + 1) / max(1, base_end - base_start + 1),
        float(start == base_start), float(end == base_end), float(len(words) < 4),
    ]


def abstract_punct(answer, start, end):
    left = 1.0 if start == 0 or BOUNDARY.search(answer[max(0, start - 2):start]) else 0.0
    right = 1.0 if end >= len(answer) - 1 or BOUNDARY.search(answer[end + 1:end + 3]) else 0.0
    return left + right


def is_local_mode(pair, params):
    """Повторяет решение штатного mode gate v3 (без побочных эффектов)."""
    from spanver import ranker
    from spanver.local_ranker import logistic
    from spanver.module import candidate_spans, document_context
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    if not answer:
        return False
    context = document_context(document)
    scored = []
    for start, end, text in candidate_spans(answer):
        vector = ranker.features(text, start, end, answer, document, context[0], context[2])
        scored.append({"start": start, "end": end, "text": text, "features": vector,
                       "risk": ranker._logit(vector, params["cand"], ranker.FEATURE_COUNT)})
    probability = logistic(ranker.mode_features(scored, answer, document), params["mode"],
                           ranker.MODE_FEATURE_COUNT)
    return probability >= params.get("mode_cut", 0.70)


def baseline_pick_list(pair, cache, config, mode="global"):
    """Все спаны базового декодера (для mFAVA их несколько)."""
    if mode == "hybrid":
        from spanver import ranker
        spans = ranker.select_spans(pair.get("model_output") or "", pair.get("document_text") or "")
        return [{"start": span["start"], "end": span["end"], "risk": span["risk"]} for span in spans]
    if mode == "mode":
        from spanver import ranker
        spans = ranker.select_spans(pair.get("model_output") or "", pair.get("document_text") or "")
        return [{"start": span["start"], "end": span["end"], "risk": span["risk"]} for span in spans]
    rows, gate = cache.get(pair)
    if gate < config["gate"]:
        return []
    picked = L.nms_pick(rows, config["threshold"], config["max_spans"], 0.1)
    return [{"start": row["start"], "end": row["end"], "risk": row["risk"]} for row in picked]


def baseline_pick(pair, cache, config, mode="global"):
    if mode == "mode":
        from spanver import ranker
        spans = ranker.select_spans(pair.get("model_output") or "",
                                    pair.get("document_text") or "",
                                    max_spans=1, nms_iou=0.1) if False else \
            ranker.select_spans(pair.get("model_output") or "", pair.get("document_text") or "")
        if not spans:
            return None
        span = spans[0]
        return {"start": span["start"], "end": span["end"], "risk": span["risk"]}
    rows, gate = cache.get(pair)
    if gate < config["gate"]:
        return None
    picked = L.nms_pick(rows, config["threshold"], config["max_spans"], 0.1)
    return picked[0] if picked else None


def collect(pairs, cache, config, mode="global"):
    rows = []
    for pair in pairs:
        chosen = baseline_pick(pair, cache, config, mode)
        if chosen is None:
            continue
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        gold = pair.get("spans") or []
        base_iou = max((iou(chosen["start"], chosen["end"], span["start"], span["end"])
                        for span in gold), default=0.0)
        for (start, end), tag in variants(answer, chosen["start"], chosen["end"]).items():
            value = max((iou(start, end, span["start"], span["end"]) for span in gold), default=0.0)
            rows.append({
                "pair_id": pair["pair_id"], "tag": tag, "start": start, "end": end,
                "vector": features(answer, document, start, end, chosen["start"], chosen["end"]),
                "label": int(value > base_iou + 1e-6),
                "gain": round(value - base_iou, 4),
            })
    return rows


def train(rows):
    X = np.array([row["vector"] for row in rows], dtype=np.float64)
    y = np.array([row["label"] for row in rows], dtype=int)
    if y.sum() == 0:
        return None, None
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=11)
    model.fit(scaler.transform(X), y)
    return model, scaler


def adjust(pair, cache, config, model, scaler, cut, mode="global"):
    chosen_list = baseline_pick_list(pair, cache, config, mode)
    if not chosen_list:
        return []
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    if mode == "hybrid" and is_local_mode(pair, L.weights()):
        return [{"start": chosen["start"], "end": chosen["end"], "risk": chosen["risk"],
                 "type": "unsupported_addition", "confidence": chosen["risk"]}
                for chosen in chosen_list]
    result = []
    for chosen in chosen_list:
        if model is None:
            result.append({"start": chosen["start"], "end": chosen["end"], "risk": chosen["risk"],
                           "type": "unsupported_addition", "confidence": chosen["risk"]})
            continue
        best = None
        for (start, end), _ in variants(answer, chosen["start"], chosen["end"]).items():
            vector = features(answer, document, start, end, chosen["start"], chosen["end"])
            probability = float(model.predict_proba(scaler.transform([vector]))[0, 1])
            if best is None or probability > best[0]:
                best = (probability, start, end)
        if best is None or best[0] < cut:
            result.append({"start": chosen["start"], "end": chosen["end"], "risk": chosen["risk"],
                           "type": "unsupported_addition", "confidence": chosen["risk"]})
        else:
            result.append({"start": best[1], "end": best[2], "risk": chosen["risk"],
                           "type": "unsupported_addition", "confidence": chosen["risk"]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/boundary_adjust_v11.json")
    parser.add_argument("--decode", default="global", choices=["global", "mode", "hybrid"])
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    cache = L.Cache(params=params)
    train_pairs, val, mfava = L.load_all()
    config = dict(params["config"])
    train_rows = collect(train_pairs, cache, config, args.decode)
    model, scaler = train(train_rows)
    report = {
        "rule": "варианты границ ±3 слова/пунктуация; логистический классификатор на train; порог по val",
        "decode": args.decode,
        "train": {"rows": len(train_rows), "positives": int(sum(row["label"] for row in train_rows))},
        "sweep": [],
    }
    for cut in (0.2, 0.35, 0.5, 0.65, 0.8):
        def predictor(pair):
            return adjust(pair, cache, config, model, scaler, cut, args.decode)

        report["sweep"].append({"cut": cut, "val": L.cell(metrics(val, predictor)),
                                "mfava": L.cell(metrics(mfava, predictor)),
                                "combined": L.cell(metrics(val + mfava, predictor))})

    def baseline(pair):
        return [{"start": row["start"], "end": row["end"], "risk": row["risk"],
                 "type": "unsupported_addition", "confidence": row["risk"]}
                for row in baseline_pick_list(pair, cache, config, args.decode)]

    report["baseline"] = {"val": L.cell(metrics(val, baseline)),
                          "mfava": L.cell(metrics(mfava, baseline)),
                          "combined": L.cell(metrics(val + mfava, baseline))}
    report["best"] = max(report["sweep"], key=lambda row: row["combined"]["span"])
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("train", json.dumps(report["train"], ensure_ascii=False), flush=True)
    print("baseline", json.dumps(report["baseline"], ensure_ascii=False), flush=True)
    for row in report["sweep"]:
        print("cut", row["cut"], "val", row["val"], "mfava", row["mfava"]["span"], flush=True)


if __name__ == "__main__":
    main()
