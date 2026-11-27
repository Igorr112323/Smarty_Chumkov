"""V11 блок 3 (v2): router whole/clause/short на относительных скорах двух голов.

Отличие от router_v11: признаки включают не только pair-level support, но и
конкуренцию гипотез — риск whole-кандидата против лучшего clause у baseline-головы,
у local HGB и у listwise. Именно этой информации не хватало: baseline уверенно
находит whole (20/23 top-1 на val), local HGB — clause (9/18).

Признаки inference-only (gold-длина и gold-позиция не используются).
Пороги — на val+mFAVA. Критерий плана: >15% ложных whole → reject.
"""
import argparse
import json
import math
import pickle
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
from spanver.local_ranker import predict_trees

MODES = ("whole", "clause", "short")
FEATURES = (
    "base_whole", "base_clause", "base_gap", "base_whole_rank", "base_whole_top1",
    "local_whole", "local_clause", "local_gap", "listwise_whole", "listwise_clause",
    "listwise_gap", "gate", "answer_cov", "answer_cos", "min_clause_cov", "mean_clause_cov",
    "cov_spread", "n_clauses", "log_answer_len", "n_sentences", "digit_share", "capital_share",
    "share_low_cov", "mean_risk", "n_rows",
)


def kind(pair):
    answer = pair.get("model_output") or ""
    spans = pair.get("spans") or []
    if not spans:
        return "neg"
    if any(span["start"] == 0 and span["end"] == len(answer) - 1 for span in spans):
        return "whole"
    return "short" if max(span["end"] - span["start"] + 1 for span in spans) < 60 else "clause"


def build_cache(pairs, params, model, cols):
    cache = {}
    for pair in pairs:
        ctx = R.pair_context(pair, params)
        rows = []
        for row in ctx["rows"]:
            vector = R.vector(ctx, row["start"], row["end"], row["text"])
            rows.append({
                "start": row["start"], "end": row["end"],
                "base": row["risk"],
                "local": predict_trees(params["local_hgb"], row["features"]),
                "listwise": R.predict_raw(model, [vector[index] for index in cols]),
            })
        cache[pair["pair_id"]] = {"ctx": ctx, "rows": rows, "kind": kind(pair)}
    return cache


def pair_features(entry):
    ctx = entry["ctx"]
    rows = entry["rows"]
    answer = ctx["answer"]
    whole_flag = [row for row in rows if row["start"] == 0 and row["end"] == len(answer) - 1]
    clause_rows = [row for row in rows if not (row["start"] == 0 and row["end"] == len(answer) - 1)]
    base_whole = max((row["base"] for row in whole_flag), default=0.0)
    base_clause = max((row["base"] for row in clause_rows), default=0.0)
    local_whole = max((row["local"] for row in whole_flag), default=0.0)
    local_clause = max((row["local"] for row in clause_rows), default=0.0)
    list_whole = max((row["listwise"] for row in whole_flag), default=0.0)
    list_clause = max((row["listwise"] for row in clause_rows), default=0.0)
    order = sorted(rows, key=lambda row: (-row["base"], row["start"]))
    whole_rank = next((position + 1 for position, row in enumerate(order)
                       if row["start"] == 0 and row["end"] == len(answer) - 1), len(rows))
    covs = ctx["clause_covs"] or [0.0]
    words = L.WORD_RE.findall(answer)
    digits = sum(1 for ch in answer if ch.isdigit())
    capitals = sum(1 for word in words if word[:1].isupper())
    return [
        base_whole, base_clause, base_whole - base_clause, whole_rank / max(1, len(rows)),
        float(whole_rank == 1),
        local_whole, local_clause, local_whole - local_clause,
        list_whole, list_clause, list_whole - list_clause,
        ctx["gate"], ctx["whole"]["token_cov"], ctx["whole"]["cos"],
        min(covs), sum(covs) / len(covs), max(covs) - min(covs), len(ctx["clauses"]),
        math.log1p(len(answer)), len(L.SENT_RE.findall(answer)) or 1,
        digits / max(1, len(answer)), capitals / max(1, len(words)),
        sum(1 for value in covs if value < 0.5) / len(["x"] * len(covs)),
        sum(row["base"] for row in rows) / max(1, len(rows)), len(rows),
    ]


def decode(entry, mode, config, chooser):
    ctx = entry["ctx"]
    rows = entry["rows"]
    answer = ctx["answer"]
    if ctx["gate"] < config["gate"] or not rows:
        return []
    if mode == "whole":
        return [{"start": 0, "end": len(answer) - 1, "risk": round(ctx["gate"], 4),
                 "type": "unsupported_addition", "confidence": round(ctx["gate"], 4)}]
    scored = chooser(entry)
    chosen = []
    for row, score in scored:
        if row["start"] == 0 and row["end"] == len(answer) - 1 and mode != "whole":
            continue
        if score < config["threshold"]:
            break
        if any(iou(row["start"], row["end"], other["start"], other["end"]) > 0.1 for other in chosen):
            continue
        chosen.append({"start": row["start"], "end": row["end"], "risk": round(float(score), 4),
                       "type": "unsupported_addition", "confidence": 0.5})
        limit = 1 if mode == "short" else config["max_spans"]
        if len(chosen) >= limit:
            break
    return chosen


def chooser_for(name):
    if name == "local":
        return lambda entry: sorted(((row, row["local"]) for row in entry["rows"]),
                                    key=lambda item: (-item[1], item[0]["start"]))
    if name == "base":
        return lambda entry: sorted(((row, row["base"]) for row in entry["rows"]),
                                    key=lambda item: (-item[1], item[0]["start"]))
    if name == "listwise":
        return lambda entry: sorted(((row, row["listwise"]) for row in entry["rows"]),
                                    key=lambda item: (-item[1], item[0]["start"]))
    raise SystemExit(name)


def run(cache, pairs, modes, config, chooser):
    picked = {}
    for pair in pairs:
        entry = cache[pair["pair_id"]]
        picked[pair["pair_id"]] = decode(entry, modes.get(pair["pair_id"], "clause"), config, chooser)

    def predictor(pair):
        return picked[pair["pair_id"]]

    return metrics(pairs, predictor)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/listwise_whole_model_v11.json")
    parser.add_argument("--cache", default="artifacts/router2_cache_v11.pkl")
    parser.add_argument("--report", default="artifacts/router_v2_v11.json")
    parser.add_argument("--with-synth", action="store_true")
    args = parser.parse_args(argv)

    params = L.weights()
    payload = json.loads((ROOT / args.model).read_text(encoding="utf-8"))
    cols = payload["columns"]
    gold = read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
    train = [p for p in gold if p.get("split") == "train" and p.get("spans")]
    val = [p for p in gold if p.get("split") == "val"]
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    synth = read_jsonl(str(ROOT / "artifacts/rushallu_synth_v11.jsonl")) if args.with_synth else []

    if Path(args.cache).exists():
        with open(args.cache, "rb") as handle:
            cache = pickle.load(handle)
        missing = [p for p in train + synth if p["pair_id"] not in cache]
        if missing:
            cache.update(build_cache(missing, params, payload, cols))
            with open(args.cache, "wb") as handle:
                pickle.dump(cache, handle)
    else:
        cache = build_cache(train + val + mfava + synth, params, payload, cols)
        with open(args.cache, "wb") as handle:
            pickle.dump(cache, handle)

    labelled = [p for p in train if kind(p) != "neg"] + synth + mfava
    X = np.array([pair_features(cache[p["pair_id"]]) for p in labelled], dtype=np.float64)
    y = np.array([MODES.index("clause") if p in synth or p in mfava else MODES.index(kind(p))
                  for p in labelled], dtype=int)
    counts = np.array([max(1, int((y == index).sum())) for index in range(len(MODES))], dtype=np.float64)
    weights = (len(y) / (len(MODES) * counts))[y]
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=8,
        l2_regularization=1.0, random_state=11,
    ).fit(X, y, sample_weight=weights)
    train_stats = {"rows": int(len(y)), "counts": {MODES[i]: int(counts[i]) for i in range(len(MODES))},
                   "with_synth": args.with_synth}

    dev = val + mfava
    Xd = np.array([pair_features(cache[p["pair_id"]]) for p in dev], dtype=np.float64)
    probabilities = model.predict_proba(Xd)
    classes = list(model.classes_)
    confusion = {}
    for position, pair in enumerate(dev):
        predicted = MODES[classes[int(np.argmax(probabilities[position]))]]
        gold_kind = cache[pair["pair_id"]]["kind"]
        confusion.setdefault(gold_kind, {}).setdefault(predicted, 0)
        confusion[gold_kind][predicted] += 1

    rows = []
    for cut in (0.3, 0.4, 0.5, 0.6, 0.7):
        modes = {}
        for position, pair in enumerate(dev):
            probabilities_row = probabilities[position]
            best = MODES[classes[int(np.argmax(probabilities_row))]]
            confidence = float(probabilities_row[int(np.argmax(probabilities_row))])
            if best == "whole" and confidence < cut:
                best = "clause"
            modes[pair["pair_id"]] = best
        whole_ids = [key for key, value in modes.items() if value == "whole"]
        false_whole = sum(1 for key in whole_ids if cache[key]["kind"] != "whole")
        whole_fp = false_whole / len(whole_ids) if whole_ids else 0.0
        for chooser_name in ("local", "base", "listwise"):
            chooser = chooser_for(chooser_name)
            for threshold in (-6.0, -3.0, 0.0, 0.2, 0.4, 0.5):
                for max_spans in (1, 2, 3):
                    for gate_cut in (0.3, 0.5, 0.6):
                        config = {"threshold": threshold, "max_spans": max_spans, "gate": gate_cut}
                        rows.append({
                            "whole_cut": cut, "chooser": chooser_name, "threshold": threshold,
                            "max_spans": max_spans, "gate": gate_cut,
                            "whole_fp_rate": round(whole_fp, 4), "whole_routed": len(whole_ids),
                            "val": L.cell(run(cache, val, modes, config, chooser)),
                            "mfava": L.cell(run(cache, mfava, modes, config, chooser)),
                            "combined": L.cell(run(cache, dev, modes, config, chooser)),
                        })
    rows.sort(key=lambda row: (row["combined"]["span"], row["val"]["span"]), reverse=True)
    passing = [row for row in rows if L.guard([row["val"], row["combined"]])]
    report = {
        "rule": "router v2 на head-relative признаках; обучение train+mFAVA(+synth опционально); пороги val+mFAVA",
        "train": train_stats, "confusion_dev": confusion,
        "best": rows[:15], "passing": passing[:10], "configs": rows, "model": args.model,
    }
    target = ROOT / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("train", json.dumps(train_stats["counts"], ensure_ascii=False), flush=True)
    print("confusion", json.dumps(confusion, ensure_ascii=False), flush=True)
    print("best", json.dumps(rows[0], ensure_ascii=False), flush=True)
    print("guard", len(passing), "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
