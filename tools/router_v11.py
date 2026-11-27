"""V11 блок 3: 3-way router whole/clause/short только на inference-признаках.

Признаки router не используют gold (ни длину gold-спана, ни его позицию): только
ответ, документ, clause-сетку и baseline-скоринг. Обучение — train + pseudo +
synth + mFAVA; пороги калибруются на val (dev). Held-out не читается.

Критерий плана: если router относит к whole пары, которые на самом деле не whole,
чаще чем в 15% случаев — конфигурация отклоняется (reject), а не «натягивается».
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import v11_lab as L
import v11_ranker as R
from spanver.evaluate import iou, metrics, read_jsonl

MODES = ("whole", "clause", "short")
ROUTER_FEATURES = (
    "log_answer_len", "n_sentences", "mean_sentence_len", "n_clauses", "mean_clause_len",
    "answer_cov", "answer_cos", "answer_sentence_cov", "answer_numbers_missing",
    "answer_names_missing", "min_clause_cov", "mean_clause_cov", "max_clause_cov",
    "median_clause_cov", "cov_spread", "share_low_cov", "n_low_cov",
    "min_clause_cos", "mean_clause_cos", "gate", "top1_risk", "mean_risk", "share_risk_high",
    "digit_share", "capital_share", "n_numbers", "query_cov",
)


def pair_kind(pair):
    answer = pair.get("model_output") or ""
    spans = pair.get("spans") or []
    if not spans:
        return "neg"
    if any(span["start"] == 0 and span["end"] == len(answer) - 1 for span in spans):
        return "whole"
    longest = max(span["end"] - span["start"] + 1 for span in spans)
    return "short" if longest < 60 else "clause"


def router_vector(pair, params, ctx=None):
    """Inference-признаки пары; gold не используется."""
    ctx = ctx or R.pair_context(pair, params)
    answer = ctx["answer"]
    document = ctx["document"]
    whole = ctx["whole"]
    clauses = ctx["clauses"]
    covs = ctx["clause_covs"] or [0.0]
    coss = [L.support_features(text, document)["cos"] for _, _, text in clauses] or [0.0]
    sentences = [m.group(0) for m in L.SENT_RE.finditer(answer)] or [answer]
    risks = [row["risk"] for row in ctx["rows"]] or [0.0]
    words = L.WORD_RE.findall(answer)
    digits = sum(1 for ch in answer if ch.isdigit())
    capitals = sum(1 for word in words if word[:1].isupper())
    query = pair.get("query_text") or ""
    query_words = set(w.lower() for w in L.WORD_RE.findall(query))
    doc_words = set(w.lower() for w in L.WORD_RE.findall(document))
    query_cov = (sum(1 for word in query_words if word in doc_words) / len(query_words)
                 if query_words else 0.0)
    values = [
        math.log1p(len(answer)),
        float(len(sentences)),
        float(len(answer) / max(1, len(sentences))),
        float(len(clauses)),
        float(np.mean([len(text) for _, _, text in clauses])) if clauses else 0.0,
        whole["token_cov"],
        whole["cos"],
        whole["sentence_cov"],
        whole["numbers_missing"],
        whole["names_missing"],
        min(covs),
        float(np.mean(covs)),
        max(covs),
        float(np.median(covs)),
        max(covs) - min(covs),
        float(sum(1 for value in covs if value < 0.5) / len(covs)),
        float(sum(1 for value in covs if value < 0.5)),
        min(coss),
        float(np.mean(coss)),
        ctx["gate"],
        ctx["top1_risk"],
        float(np.mean(risks)),
        float(sum(1 for value in risks if value > 0.5) / len(risks)),
        digits / max(1, len(answer)),
        capitals / max(1, len(words)),
        float(len(L.NUM_RE.findall(answer))),
        query_cov,
    ]
    return values


def train_router(params):
    gold = read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
    train = [p for p in gold if p.get("split") == "train" and p.get("spans")]
    pseudo = read_jsonl(str(ROOT / "artifacts/clause_pseudo_v11.jsonl"))
    synth = read_jsonl(str(ROOT / "artifacts/rushallu_synth_v11.jsonl"))
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    rows, labels = [], []
    for pair in train:
        kind = pair_kind(pair)
        if kind == "neg":
            continue
        rows.append(router_vector(pair, params))
        labels.append(MODES.index(kind))
    for pair in pseudo + synth + mfava:
        rows.append(router_vector(pair, params))
        labels.append(MODES.index("clause"))
    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=int)
    counts = np.array([max(1, int((y == index).sum())) for index in range(len(MODES))], dtype=np.float64)
    weights = (len(y) / (len(MODES) * counts))[y]
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=1.0, random_state=11,
    ).fit(X, y, sample_weight=weights)
    return model, {"rows": len(y), "counts": {MODES[i]: int(counts[i]) for i in range(len(MODES))},
                   "features": list(ROUTER_FEATURES)}


def load_ranker(path):
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return payload, payload["columns"]


def score_ranker(payload, cols, pair, params, cache):
    key = pair["pair_id"]
    if key in cache:
        return cache[key]
    ctx = R.pair_context(pair, params)
    if not ctx["rows"]:
        cache[key] = (ctx, [])
        return cache[key]
    X = np.array([R.vector(ctx, row["start"], row["end"], row["text"]) for row in ctx["rows"]],
                 dtype=np.float64)
    scores = [R.predict_raw(payload, list(row[cols])) for row in X]
    cache[key] = (ctx, list(zip(ctx["rows"], scores)))
    return cache[key]


def pick(pairs, order, payload, cols, params, config, cache):
    """order: {"whole": [ids], ...} — router назначает режим на пару."""
    gate_cut = config["gate"]
    threshold = config["threshold"]
    max_spans = config["max_spans"]
    short_ratio = config["short_ratio"]
    result = {}
    for pair in pairs:
        ctx, scored = score_ranker(payload, cols, pair, params, cache)
        mode = order.get(pair["pair_id"], "clause")
        chosen = []
        if ctx["gate"] >= gate_cut:
            if mode == "whole":
                answer = ctx["answer"]
                chosen = [{"start": 0, "end": len(answer) - 1, "risk": ctx["gate"],
                           "type": "unsupported_addition", "confidence": ctx["gate"]}]
            else:
                limit = max_spans
                if mode == "short":
                    limit = 1
                for row, score in sorted(scored, key=lambda item: (-item[1], item[0]["start"])):
                    if score < threshold:
                        break
                    if mode == "short" and (row["end"] - row["start"] + 1) > short_ratio * len(ctx["answer"]):
                        continue
                    if any(iou(row["start"], row["end"], other["start"], other["end"]) > 0.1
                           for other in chosen):
                        continue
                    chosen.append({"start": row["start"], "end": row["end"],
                                   "risk": round(float(score), 4),
                                   "type": "unsupported_addition", "confidence": 0.5})
                    if len(chosen) >= limit:
                        break
        result[pair["pair_id"]] = chosen

    def predictor(pair):
        return result[pair["pair_id"]]

    return metrics(pairs, predictor), result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker", default="artifacts/listwise_full_model_v11.json")
    parser.add_argument("--report", default="artifacts/router_v11.json")
    args = parser.parse_args(argv)

    params = L.weights()
    model, train_stats = train_router(params)
    payload, cols = load_ranker(args.ranker)
    _train, val, mfava = L.load_all()
    cache = {}
    truth = {}
    for pair in val + mfava:
        ctx, _ = score_ranker(payload, cols, pair, params, cache)
        truth[pair["pair_id"]] = (pair_kind(pair), router_vector(pair, params, ctx))

    X = np.array([truth[pair["pair_id"]][1] for pair in val + mfava], dtype=np.float64)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    mode_index = {name: (classes.index(index) if index in classes else None) for index, name in enumerate(MODES)}

    def order_for(cut):
        order = {}
        for position, pair in enumerate(val + mfava):
            row = probabilities[position]
            best_position = int(np.argmax(row))
            best = MODES[classes[best_position]]
            confidence = float(row[best_position])
            if modes_allowed is not None and best not in modes_allowed:
                best = "clause"
            if best == "whole" and confidence < cut:
                best = "clause"
            order[pair["pair_id"]] = best
        return order

    results = []
    for modes_allowed in (None, ("clause",), ("short", "clause")):
        for cut in (0.3, 0.4, 0.5, 0.6, 0.7):
            order = order_for(cut)
            whole_ids = [key for key, value in order.items() if value == "whole"]
            whole_truth = [truth[key][0] for key in whole_ids]
            false_whole = sum(1 for kind in whole_truth if kind != "whole")
            whole_fp_rate = false_whole / len(whole_truth) if whole_truth else 0.0
            if whole_fp_rate > 0.15:
                continue
            for threshold in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
                for max_spans in (1, 2, 3):
                    for gate_cut in (0.3, 0.5, 0.6):
                        config = {"threshold": threshold, "max_spans": max_spans,
                                  "gate": gate_cut, "short_ratio": 0.25}
                        combined, _ = pick(val + mfava, order, payload, cols, params, config, cache)
                        val_metric, _ = pick(val, order, payload, cols, params, config, cache)
                        mfava_metric, _ = pick(mfava, order, payload, cols, params, config, cache)
                        results.append({
                            "modes": "all" if modes_allowed is None else "+".join(modes_allowed),
                            "whole_cut": cut, "threshold": threshold, "max_spans": max_spans,
                            "gate": gate_cut, "whole_fp_rate": round(whole_fp_rate, 4),
                            "whole_routed": len(whole_ids),
                            "combined": L.cell(combined), "val": L.cell(val_metric),
                            "mfava": L.cell(mfava_metric),
                        })
    results.sort(key=lambda row: (row["val"]["span"], row["combined"]["span"]), reverse=True)
    best = results[0]
    guard_rows = [row for row in results if L.guard([row["val"], row["combined"]])]

    confusion = {}
    for position, pair in enumerate(val + mfava):
        gold_kind = truth[pair["pair_id"]][0]
        predicted = MODES[classes[int(np.argmax(probabilities[position]))]]
        confusion.setdefault(gold_kind, {}).setdefault(predicted, 0)
        confusion[gold_kind][predicted] += 1

    report = {
        "rule": "router обучается на train/pseudo/synth/mFAVA; пороги только на val+mFAVA; gold-длина не признак",
        "train": train_stats,
        "confusion_dev": confusion,
        "best": best, "top": results[:15], "passing": guard_rows[:10],
        "results": results,
        "ranker": args.ranker,
    }
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("train", json.dumps(train_stats["counts"], ensure_ascii=False), flush=True)
    print("confusion", json.dumps(confusion, ensure_ascii=False), flush=True)
    print("best", json.dumps(best, ensure_ascii=False), flush=True)
    print("guard", len(guard_rows), "configs", len(results), flush=True)


if __name__ == "__main__":
    main()
