import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, module
from spanver.evaluate import metrics, read_jsonl
from spanver.snap import snapped
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import load_model, pair_probabilities

FPR_LIMIT = 0.40
GATE_GRID = tuple(round(0.20 + 0.05 * step, 2) for step in range(11))
TOKEN_THRESHOLD = 0.20


def score(pairs, predictor):
    result = metrics(pairs, predictor)
    return {
        "span_precision": result["span_level"]["precision"],
        "span_recall": result["span_level"]["recall"],
        "span_f1": result["span_level"]["f1"],
        "token_precision": result["token_level"]["precision"],
        "token_recall": result["token_level"]["recall"],
        "token_f1": result["token_level"]["f1"],
        "pair_f1": result["pair_level"]["f1"],
        "fpr_pair": result["fpr"],
        "type_accuracy": result["type_accuracy"],
    }


def lexical(pairs):
    store = {}
    for pair in pairs:
        store[pair.get("pair_id")] = module.verify(
            pair.get("model_output") or "", pair.get("document_text") or ""
        )["spans"]
    return store


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--model", default="artifacts/negatives/epoch_1")
    parser.add_argument("--report", default="artifacts/schemes_report.json")
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    splits = load_splits()
    gold = read_jsonl(Path(args.data))
    pairs = [pair for pair in ordered_pairs(gold, splits["val"]) if pair.get("model_output")]
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}
    report = {"pairs": len(pairs), "fpr_limit": FPR_LIMIT, "scheme3": {}, "scheme2": {}}

    store = lexical(pairs)
    base = score(pairs, lambda pair: store[pair.get("pair_id")])
    report["scheme3"]["lexical_base"] = base
    print(f"схема 3, лексический базис: span F1 {base['span_f1']} token F1 {base['token_f1']}", flush=True)
    for mode in ("expand", "shrink", "nearest"):
        predictor = lambda pair, m=mode: snapped(store[pair.get("pair_id")], answers[pair.get("pair_id")], m)
        result = score(pairs, predictor)
        report["scheme3"][f"lexical_{mode}"] = result
        print(f"  {mode}: span F1 {result['span_f1']} token F1 {result['token_f1']} FPR {result['fpr_pair']}", flush=True)

    torch, tokenizer, model = load_model(args.model)
    cache = {}
    clock = time.perf_counter()
    for pair in pairs:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=args.top)
    model_seconds = round(time.perf_counter() - clock, 1)
    model_spans = {}
    for pair in pairs:
        cached = cache[pair.get("pair_id")]
        flags = decode.flags_for(cached["pieces"], TOKEN_THRESHOLD, None)
        model_spans[pair.get("pair_id")] = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, 2)
    model_base = score(pairs, lambda pair: model_spans[pair.get("pair_id")])
    report["scheme3"]["model_base"] = dict(model_base, threshold=TOKEN_THRESHOLD, seconds=model_seconds)
    print(f"модель {args.model} при {TOKEN_THRESHOLD}: span F1 {model_base['span_f1']} token F1 {model_base['token_f1']} FPR {model_base['fpr_pair']}", flush=True)
    for mode in ("expand", "shrink", "nearest"):
        predictor = lambda pair, m=mode: snapped(model_spans[pair.get("pair_id")], answers[pair.get("pair_id")], m)
        result = score(pairs, predictor)
        report["scheme3"][f"model_{mode}"] = result
        print(f"  модель {mode}: span F1 {result['span_f1']} token F1 {result['token_f1']} FPR {result['fpr_pair']}", flush=True)

    rows = []
    for gate in GATE_GRID:
        def predict(pair, limit=gate):
            cached = cache.get(pair.get("pair_id"))
            if not cached:
                return []
            top = max((score_value for _, _, score_value in cached["pieces"]), default=0.0)
            if top < limit:
                return []
            flags = decode.flags_for(cached["pieces"], TOKEN_THRESHOLD, None)
            return decode.spans_from_flags(cached["answer"], cached["pieces"], flags, 2)

        result = score(pairs, predict)
        rows.append(dict(result, gate=gate))
        print(
            f"схема 2, гейт пары {gate}: token F1 {result['token_f1']} span F1 {result['span_f1']} "
            f"FPR {result['fpr_pair']}",
            flush=True,
        )
    allowed = [row for row in rows if row["fpr_pair"] <= FPR_LIMIT]
    report["scheme2"] = {
        "grid": list(GATE_GRID),
        "token_threshold": TOKEN_THRESHOLD,
        "rows": rows,
        "best_token": max(allowed or rows, key=lambda row: row["token_f1"]),
        "best_span": max(allowed or rows, key=lambda row: row["span_f1"]),
        "base_model": model_base,
    }
    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = report["scheme2"]["best_token"]
    print(f"схема 2, лучший token F1 {best['token_f1']} при гейте {best['gate']}, FPR {best['fpr_pair']}", flush=True)
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
