import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.decode import GRID, MIN_CHARS, NEIGHBOR, sweep
from spanver.evaluate import read_jsonl
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import load_model, pair_probabilities


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/span_detector_v5_last")
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--report", default="artifacts/decode_report.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    splits = load_splits()
    gold = read_jsonl(Path(args.data))
    val_labeled = [pair for pair in ordered_pairs(gold, splits["val"]) if pair.get("model_output")]
    extra = Path("artifacts/pairs_unlabeled.jsonl")
    everything = list(gold)
    if extra.exists():
        everything.extend(read_jsonl(extra))
    val_split = [pair for pair in ordered_pairs(everything, splits["val"]) if pair.get("model_output")]

    torch, tokenizer, model = load_model(args.weights)
    report = {"weights": args.weights, "top_k": args.top, "neighbor": NEIGHBOR, "min_chars": MIN_CHARS, "grid": list(GRID)}
    for name, pairs in (("val150", val_labeled), ("val637", val_split)):
        cache = {}
        clock = time.perf_counter()
        for pair in pairs:
            cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=args.top)
        seconds = time.perf_counter() - clock
        rows = sweep(cache, pairs)
        report[name] = {
            "pairs": len(pairs),
            "seconds": round(seconds, 1),
            "chars": sum(len(pair.get("model_output") or "") for pair in pairs),
            "rows": rows,
        }
        print(f"== {name} ({len(pairs)} пар, {seconds:.1f} с)", flush=True)
        for row in rows:
            print(
                f"  t {row['threshold']:.2f}: token P {row['precision']:.4f} R {row['recall']:.4f} "
                f"F1 {row['f1']:.4f} FPR {row['fpr']:.4f} | span F1 {row['span_f1']:.4f} | pair F1 {row['pair_f1']:.4f}",
                flush=True,
            )
    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = max(report["val150"]["rows"], key=lambda row: (row["f1"], -row["threshold"]))
    print(
        f"лучший на val150: t {best['threshold']} token F1 {best['f1']} FPR {best['fpr']} "
        f"порог цели 0,50 на val637 при FPR <= 0,40",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
