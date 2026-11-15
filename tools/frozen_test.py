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

TOP = 4


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
    parser.add_argument("--config", default="artifacts/frozen_config.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--report", default="artifacts/test_frozen.json")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    pairs = [pair for pair in ordered_pairs(gold, splits[args.split]) if pair.get("model_output")]
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}
    torch, tokenizer, model = load_model(config["checkpoint"])
    started = time.perf_counter()
    cache = {}
    for pair in pairs:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=TOP)
    model_seconds = time.perf_counter() - started

    def predict(pair):
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            return []
        top = max((value for _, _, value in cached["pieces"]), default=0.0)
        if top < config["pair_gate"]:
            return []
        flags = decode.flags_for(cached["pieces"], config["token_threshold"], config["neighbor"])
        spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, config["min_chars"])
        return snapped(spans, answers[pair.get("pair_id")], config["snap"])

    result = metrics(pairs, predict)
    report = {
        "split": args.split,
        "pairs": len(pairs),
        "threshold": config["token_threshold"],
        "pair_gate": config["pair_gate"],
        "checkpoint": config["checkpoint"],
        "config_sha256_source": args.config,
        "token_precision": result["token_level"]["precision"],
        "token_recall": result["token_level"]["recall"],
        "token_f1": result["token_level"]["f1"],
        "span_precision": result["span_level"]["precision"],
        "span_recall": result["span_level"]["recall"],
        "span_f1": result["span_level"]["f1"],
        "pair_precision": result["pair_level"]["precision"],
        "pair_recall": result["pair_level"]["recall"],
        "pair_f1": result["pair_level"]["f1"],
        "fpr_pair": result["fpr"],
        "fpr_token": token_fpr(pairs, predict),
        "type_accuracy": result["type_accuracy"],
        "seconds": round(time.perf_counter() - started, 1),
        "ms_per_pair": round(model_seconds * 1000.0 / max(1, len(pairs)), 2),
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
