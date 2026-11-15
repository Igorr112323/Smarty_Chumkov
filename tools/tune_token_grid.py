import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import load_model, pair_probabilities

PAIR_GATE = 0.45
MIN_CHARS = 2
FPR_LIMIT = 0.40
GRID = tuple(round(0.05 + 0.01 * step, 2) for step in range(26))


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/negatives/epoch_1")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--split", default="val")
    parser.add_argument("--report", default="artifacts/token_grid.json")
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    pairs = [pair for pair in ordered_pairs(gold, splits[args.split]) if pair.get("model_output")]
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}
    torch, tokenizer, model = load_model(args.checkpoint)
    cache = {}
    for pair in pairs:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=args.top)
    clock = time.perf_counter()

    def build(threshold):
        def predict(pair):
            cached = cache.get(pair.get("pair_id"))
            if not cached:
                return []
            top = max((value for _, _, value in cached["pieces"]), default=0.0)
            if top < PAIR_GATE:
                return []
            flags = decode.flags_for(cached["pieces"], threshold, None)
            spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, MIN_CHARS)
            return snapped(spans, answers[pair.get("pair_id")], "expand")

        return predict

    rows = []
    for threshold in GRID:
        predictor = build(threshold)
        result = metrics(pairs, predictor)
        rows.append(
            {
                "threshold": threshold,
                "token_precision": result["token_level"]["precision"],
                "token_recall": result["token_level"]["recall"],
                "token_f1": result["token_level"]["f1"],
                "span_precision": result["span_level"]["precision"],
                "span_recall": result["span_level"]["recall"],
                "span_f1": result["span_level"]["f1"],
                "pair_f1": result["pair_level"]["f1"],
                "fpr_pair": result["fpr"],
                "fpr_token": token_fpr(pairs, predictor),
                "type_accuracy": result["type_accuracy"],
            }
        )
        print(
            f"{threshold:.2f} token {result['token_level']['precision']:.4f}/{result['token_level']['recall']:.4f}/"
            f"{result['token_level']['f1']:.4f} span {result['span_level']['f1']:.4f} "
            f"pair {result['pair_level']['f1']:.4f} FPR_pair {result['fpr']:.4f} FPR_token {rows[-1]['fpr_token']:.4f}",
            flush=True,
        )

    allowed = [row for row in rows if row["fpr_pair"] <= FPR_LIMIT and row["fpr_token"] <= FPR_LIMIT]
    pool = allowed or rows
    best = max(pool, key=lambda row: (row["token_f1"], -row["fpr_pair"]))
    report = {
        "split": args.split,
        "pairs": len(pairs),
        "checkpoint": args.checkpoint,
        "pair_gate": PAIR_GATE,
        "min_chars": MIN_CHARS,
        "snap": "expand",
        "grid": list(GRID),
        "fpr_limit": FPR_LIMIT,
        "rows": rows,
        "best": best,
        "best_internal": bool(best["threshold"] not in (min(GRID), max(GRID))),
        "seconds": round(time.perf_counter() - clock, 1),
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"лучший порог {best['threshold']:.2f}: token F1 {best['token_f1']:.4f} FPR_pair {best['fpr_pair']:.4f} "
        f"FPR_token {best['fpr_token']:.4f} span F1 {best['span_f1']:.4f}",
        flush=True,
    )
    print(f"граница сетки проверена вниз: {not report['best_internal']}", flush=True)
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
