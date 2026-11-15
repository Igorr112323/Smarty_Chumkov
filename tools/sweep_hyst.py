import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, lexfeat
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.splits import load_splits
from tools.train_cv import group_folds

START_GRID = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
EXTEND_GRID = (0.10, 0.15, 0.20, 0.25)
K_GRID = (1.0, 1.5, 2.0, 3.0)
PAIR_GATE = 0.45
MIN_CHARS = 2
FPR_LIMIT = 0.40
BATCH = 16
METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")


def token_fpr(pairs, predictor):
    support = 0
    flagged = 0
    for pair in pairs:
        answer = pair.get("model_output") or ""
        gold = pair.get("spans") or []
        predicted = predictor(pair)
        for start, end in token_positions(answer):
            if any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in gold):
                continue
            support += 1
            if any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in predicted):
                flagged += 1
    return round(flagged / support, 4) if support else 0.0


def flags_with(pieces, t_start, t_extend, ratio):
    flags = []
    active = False
    for _, _, score in pieces:
        allowed = score >= ratio
        if active:
            if allowed and score >= t_extend:
                flags.append(True)
                continue
            active = False
            flags.append(False)
            continue
        if allowed and score >= t_start:
            active = True
            flags.append(True)
            continue
        flags.append(False)
    return flags


def measure(pairs, cache, medians, t_start, t_extend, k, ratio):
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}

    def predict(pair):
        key = pair.get("pair_id")
        cached = cache.get(key)
        if not cached:
            return []
        top = max((value for _, _, value in cached["pieces"]), default=0.0)
        if top < PAIR_GATE:
            return []
        limit = medians.get(key, 0.0) * k if ratio else 0.0
        flags = flags_with(cached["pieces"], t_start, t_extend, limit)
        spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, MIN_CHARS)
        return snapped(spans, answers[key], "expand")

    result = metrics(pairs, predict)
    return {
        "token_precision": result["token_level"]["precision"],
        "token_recall": result["token_level"]["recall"],
        "token_f1": result["token_level"]["f1"],
        "span_precision": result["span_level"]["precision"],
        "span_recall": result["span_level"]["recall"],
        "span_f1": result["span_level"]["f1"],
        "pair_f1": result["pair_level"]["f1"],
        "fpr_pair": result["fpr"],
        "fpr_token": token_fpr(pairs, predict),
        "type_accuracy": result["type_accuracy"],
    }


def aggregate(rows):
    summary = {}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[metric] = {"mean": round(mean, 4), "std": round(variance ** 0.5, 4)}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--cv", default="artifacts/cv/feat5")
    parser.add_argument("--source", default="models/rubert-tiny2")
    parser.add_argument("--report", default="artifacts/sweep_hyst.json")
    args = parser.parse_args(argv)

    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    by_id = {pair.get("pair_id"): pair for pair in gold}
    ours = [by_id[pair_id] for name in ("train", "val") for pair_id in splits[name] if pair_id in by_id]
    ours = [pair for pair in ours if pair.get("model_output")]
    buckets = group_folds(ours, 5)
    report = {"start_grid": list(START_GRID), "extend_grid": list(EXTEND_GRID), "k_grid": list(K_GRID), "fpr_limit": FPR_LIMIT, "hysteresis": {}, "contrast": {}}

    folds = []
    for index, folder in enumerate(sorted(Path(args.cv).glob("fold_*"))):
        tokenizer, model = lexfeat.load_saved(folder, args.source)
        held = buckets[index]
        cache = lexfeat.batched_probabilities(tokenizer, model, held, top=4, batch=BATCH)
        medians = {}
        for key, value in cache.items():
            scores = [score for _, _, score in value["pieces"]]
            medians[key] = statistics.median(scores) if scores else 0.0
        folds.append({"pairs": held, "cache": cache, "medians": medians})
        print(f"фолд {index + 1}: пар {len(cache)}", flush=True)

    for t_start in START_GRID:
        for t_extend in EXTEND_GRID:
            if t_extend >= t_start:
                continue
            rows = [measure(item["pairs"], item["cache"], item["medians"], t_start, t_extend, 1.0, False) for item in folds]
            summary = aggregate(rows)
            report["hysteresis"][f"{t_start:.2f}/{t_extend:.2f}"] = summary
            print(
                f"{t_start:.2f}/{t_extend:.2f}: token F1 {summary['token_f1']['mean']} span F1 {summary['span_f1']['mean']} "
                f"FPR_pair {summary['fpr_pair']['mean']} FPR_token {summary['fpr_token']['mean']}",
                flush=True,
            )

    allowed = [
        (key, value)
        for key, value in report["hysteresis"].items()
        if value["fpr_pair"]["mean"] <= FPR_LIMIT and value["fpr_token"]["mean"] <= FPR_LIMIT
    ]
    pool = allowed or list(report["hysteresis"].items())
    best_key = max(pool, key=lambda item: item[1]["token_f1"]["mean"])[0]
    best_start, best_extend = (float(piece) for piece in best_key.split("/"))
    report["best_hysteresis"] = {"pair": best_key, "summary": report["hysteresis"][best_key]}
    print(f"лучшая пара порогов: {best_key}", flush=True)

    for k in K_GRID:
        rows = [measure(item["pairs"], item["cache"], item["medians"], best_start, best_extend, k, True) for item in folds]
        summary = aggregate(rows)
        report["contrast"][f"{k:.1f}"] = summary
        print(
            f"K {k:.1f}: token F1 {summary['token_f1']['mean']} span F1 {summary['span_f1']['mean']} "
            f"FPR_pair {summary['fpr_pair']['mean']} FPR_token {summary['fpr_token']['mean']}",
            flush=True,
        )

    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
