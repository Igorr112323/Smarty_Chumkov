import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import metrics, read_jsonl
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import load_model, pair_probabilities

GRID = tuple(round(0.20 + 0.01 * step, 2) for step in range(31))
NEIGHBORS = (0.8, None)
MIN_CHARS = (1, 2)
FPR_LIMIT = 0.40


def token_row(cache, pairs, threshold, neighbor, min_chars):
    tp = fp = fn = 0
    negative_pairs = 0
    flagged_pairs = 0
    support_tokens = 0
    support_flagged = 0
    for pair in pairs:
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            continue
        pieces = cached["pieces"]
        flags = decode.flags_for(pieces, threshold, neighbor)
        spans = decode.spans_from_flags(cached["answer"], pieces, flags, min_chars)
        kept = [(span["start"], span["end"]) for span in spans]
        gold = pair.get("spans") or []
        hit = False
        for (start, end, score), flag in zip(pieces, flags):
            inside = any(min(end, span["end"] + 1) - max(start, span["start"]) > 0 for span in gold)
            predicted = flag and any(left <= start and end - 1 <= right for left, right in kept)
            if predicted:
                hit = True
            if inside and predicted:
                tp += 1
            elif inside:
                fn += 1
            elif predicted:
                fp += 1
            if not inside:
                support_tokens += 1
                if predicted:
                    support_flagged += 1
        if not gold:
            negative_pairs += 1
            if hit:
                flagged_pairs += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "neighbor": neighbor,
        "min_chars": min_chars,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr_pair": round(flagged_pairs / negative_pairs, 4) if negative_pairs else 0.0,
        "fpr_token": round(support_flagged / support_tokens, 4) if support_tokens else 0.0,
        "support_tokens": support_tokens,
        "support_flagged": support_flagged,
    }


def full_row(cache, pairs, config):
    predict = decode.predictor_from_cache(cache, config["threshold"], config["neighbor"], config["min_chars"])
    spans = metrics(pairs, predict)
    row = dict(config)
    row["span_precision"] = spans["span_level"]["precision"]
    row["span_recall"] = spans["span_level"]["recall"]
    row["span_f1"] = spans["span_level"]["f1"]
    row["pair_f1"] = spans["pair_level"]["f1"]
    row["type_accuracy"] = spans["type_accuracy"]
    return row


def evaluate_checkpoint(tokenizer, model, torch, pairs, top=4):
    cache = {}
    clock = time.perf_counter()
    for pair in pairs:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=top)
    seconds = time.perf_counter() - clock
    rows = []
    for neighbor in NEIGHBORS:
        for min_chars in MIN_CHARS:
            for threshold in GRID:
                token = token_row(cache, pairs, threshold, neighbor, min_chars)
                rows.append(token)
    allowed = [row for row in rows if row["fpr_pair"] <= FPR_LIMIT]
    pool = allowed or rows
    best = max(pool, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5)))
    best_full = full_row(cache, pairs, {"threshold": best["threshold"], "neighbor": best["neighbor"], "min_chars": best["min_chars"]})
    return {
        "pairs": len(pairs),
        "seconds": round(seconds, 2),
        "chars": sum(len(pair.get("model_output") or "") for pair in pairs),
        "rows": rows,
        "best": best,
        "best_full": best_full,
        "fpr_limit": FPR_LIMIT,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--report", default="artifacts/decode_grid_report.json")
    args = parser.parse_args(argv)

    splits = load_splits()
    gold = read_jsonl(Path(args.data))
    val150 = [pair for pair in ordered_pairs(gold, splits["val"]) if pair.get("model_output")]
    torch, tokenizer, _ = load_model(args.weights[0])
    report = {"grid": list(GRID), "neighbors": [0.8, "off"], "min_chars": [1, 2], "fpr_limit": FPR_LIMIT, "val_pairs": len(val150), "checkpoints": {}}
    for path in args.weights:
        _, tokenizer, model = load_model(path)
        result = evaluate_checkpoint(tokenizer, model, torch, val150, top=args.top)
        result["path"] = path
        report["checkpoints"][path] = result
        best = result["best"]
        full = result["best_full"]
        print(
            f"{path}: порог {best['threshold']} neighbor {best['neighbor']} min {best['min_chars']} | "
            f"token P {best['precision']} R {best['recall']} F1 {best['f1']} | "
            f"FPR_pair {best['fpr_pair']} FPR_token {best['fpr_token']} | "
            f"span F1 {full['span_f1']} pair F1 {full['pair_f1']}",
            flush=True,
        )
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
