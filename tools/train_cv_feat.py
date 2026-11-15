import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, lexfeat
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.splits import load_splits
from spanver.stage_train import ENCODER_RATE, HEAD_RATE, stage_c_plans
from tools.train_cv import group_folds

TOKEN_THRESHOLD = 0.25
PAIR_GATE = 0.45
MIN_CHARS = 2
TOP = 4
OURS_PER_BATCH = 8
NEG_PER_BATCH = 8
SEED = 0


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


def evaluate(tokenizer, model, pairs):
    import torch

    cache = {}
    with torch.no_grad():
        for pair in pairs:
            cached = lexfeat.pair_probabilities(tokenizer, model, pair, top=TOP)
            if cached:
                cache[pair.get("pair_id")] = cached
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}

    def predict(pair):
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            return []
        top = max((value for _, _, value in cached["pieces"]), default=0.0)
        if top < PAIR_GATE:
            return []
        flags = decode.flags_for(cached["pieces"], TOKEN_THRESHOLD, None)
        spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, MIN_CHARS)
        return snapped(spans, answers[pair.get("pair_id")], "expand")

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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/rubert-tiny2")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--negatives", default="artifacts/rufacts/rufacts_negative.jsonl")
    parser.add_argument("--extra-negatives", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out", default="artifacts/cv/feat5")
    parser.add_argument("--report", default="artifacts/cv_report_feat5.json")
    args = parser.parse_args(argv)
    import torch

    started = time.perf_counter()
    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    by_id = {pair.get("pair_id"): pair for pair in gold}
    ours = [by_id[pair_id] for name in ("train", "val") for pair_id in splits[name] if pair_id in by_id]
    ours = [pair for pair in ours if pair.get("model_output")]
    negatives = [pair for pair in read_jsonl(Path(args.negatives)) if pair.get("model_output")]
    if args.extra_negatives and Path(args.extra_negatives).exists():
        negatives = negatives + [pair for pair in read_jsonl(Path(args.extra_negatives)) if pair.get("model_output")]
    buckets = group_folds(ours, args.folds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "tag": "feat5",
        "folds": args.folds,
        "epochs": args.epochs,
        "pairs": len(ours),
        "negatives": len(negatives),
        "features": list(lexfeat.FEATURE_COUNT for _ in range(1)),
        "decode": {"token_threshold": TOKEN_THRESHOLD, "pair_gate": PAIR_GATE, "min_chars": MIN_CHARS, "snap": "expand"},
        "fold_sizes": [len(bucket) for bucket in buckets],
        "rows": [],
    }
    print(f"пар {len(ours)}, фолдов {args.folds}, негативов {len(negatives)}, размеры {report['fold_sizes']}", flush=True)
    for index in range(args.folds):
        clock = time.perf_counter()
        held = buckets[index]
        training = [pair for position, bucket in enumerate(buckets) if position != index for pair in bucket]
        tokenizer, model = lexfeat.load_feature_model(args.weights)
        table = lexfeat.build_table(tokenizer, training + negatives)
        ours_ids = [pair["pair_id"] for pair in training if pair["pair_id"] in table]
        negative_ids = [pair["pair_id"] for pair in negatives if pair["pair_id"] in table]
        keep = int(round(len(negative_ids) * len(ours_ids) / max(1, len(ours))))
        sampled = random.Random(SEED + index).sample(negative_ids, min(keep, len(negative_ids)))
        plans = stage_c_plans(ours_ids, sampled, args.epochs, ours_per_batch=OURS_PER_BATCH, helper_per_batch=NEG_PER_BATCH)
        print(f"фолд {index + 1}: батчей {len(plans[0])}, строк {len(table)}", flush=True)
        history = lexfeat.train(torch, model, table, plans, HEAD_RATE, ENCODER_RATE, tokenizer.pad_token_id)
        row = evaluate(tokenizer, model, held)
        checkpoint = lexfeat.save_feature_model(model, tokenizer, out / f"fold_{index + 1}", args.weights)
        row.update({"fold": index + 1, "held_pairs": len(held), "train_pairs": len(training), "seconds": round(time.perf_counter() - clock, 1), "train": history[-1], "checkpoint": str(checkpoint)})
        report["rows"].append(row)
        print(
            f"фолд {index + 1}: token F1 {row['token_f1']} span F1 {row['span_f1']} pair F1 {row['pair_f1']} "
            f"FPR_pair {row['fpr_pair']} FPR_token {row['fpr_token']} за {row['seconds']} с",
            flush=True,
        )
    summary = {}
    for metric in ("token_f1", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "token_precision", "token_recall", "span_precision", "span_recall", "type_accuracy"):
        values = [row[metric] for row in report["rows"]]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[metric] = {"mean": round(mean, 4), "std": round(variance ** 0.5, 4)}
    report["summary"] = summary
    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for metric, values in summary.items():
        print(f"CV {metric}: {values['mean']} ± {values['std']}", flush=True)
    print(f"время {report['elapsed_s']} с", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
