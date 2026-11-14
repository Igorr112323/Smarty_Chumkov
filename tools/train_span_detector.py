import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import (
    EPOCHS,
    SEED,
    evaluate,
    examples_for,
    load_model,
    metrics,
    predictor_from_cache,
    runtime,
    save_checkpoint,
    scan_threshold,
    train,
)

SMOKE_PAIRS = 200
SMOKE_EPOCHS = 2


def pick_training(pairs, limit):
    positive = [pair for pair in sorted(pairs, key=lambda item: item.get("pair_id") or "") if pair.get("spans")]
    negative = [pair for pair in sorted(pairs, key=lambda item: item.get("pair_id") or "") if not pair.get("spans")]
    if not limit:
        return positive + negative
    share = int(limit * 0.8)
    chosen = positive[:share]
    chosen += negative[: max(0, limit - len(chosen))]
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/rubert-tiny2")
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--out", default="artifacts/span_detector")
    parser.add_argument("--limit", type=int, default=SMOKE_PAIRS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    splits = load_splits()
    pairs = read_jsonl(Path(args.data))
    extra = Path("artifacts/pairs_unlabeled.jsonl")
    if extra.exists():
        pairs.extend(read_jsonl(extra))
    train_pairs = ordered_pairs(pairs, splits["train"])
    val_pairs = ordered_pairs(pairs, splits["val"])
    chosen = pick_training(train_pairs, args.limit)
    epochs = args.epochs if args.epochs is not None else (SMOKE_EPOCHS if args.limit else EPOCHS)

    torch, tokenizer, model = load_model(args.weights)
    rows = examples_for(chosen, tokenizer)
    print(f"пар в обучении: {len(chosen)}, примеров: {len(rows)}, эпох: {epochs}", flush=True)
    losses = train(torch, model, tokenizer, rows, epochs=epochs)
    print(f"loss по эпохам: {losses}", flush=True)
    checkpoint = save_checkpoint(model, tokenizer, args.out)
    result, cache = evaluate(tokenizer, model, torch, val_pairs, args.eval_limit)
    usable = val_pairs[: args.eval_limit] if args.eval_limit else val_pairs
    best, scan = scan_threshold(cache, usable)
    result = metrics(usable, predictor_from_cache(cache, best["threshold"]))
    report = {
        "model": str(args.weights),
        "checkpoint": str(checkpoint),
        "train_pairs": len(chosen),
        "examples": len(rows),
        "epochs": epochs,
        "loss": losses,
        "val_pairs": result["n_pairs"],
        "val_span": result["span_level"],
        "val_token": result["token_level"],
        "val_pair": result["pair_level"],
        "val_fpr": result["fpr"],
        "val_type_accuracy": result["type_accuracy"],
        "val_threshold": best["threshold"],
        "val_threshold_history": scan,
        "val_best": best,
        "seed": SEED,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
