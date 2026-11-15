import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.splits import load_splits
from spanver.stage_train import HEAD_RATE, ENCODER_RATE, build_table, run_stage, stage_c_plans
from spanver.span_detector import load_model, pair_probabilities, save_checkpoint

OURS_PER_BATCH = 8
NEG_PER_BATCH = 8
SEED = 0
TOKEN_THRESHOLD = 0.25
PAIR_GATE = 0.45
MIN_CHARS = 2
TOP = 4


def group_folds(pairs, folds):
    by_doc = {}
    for pair in pairs:
        key = (pair.get("doc_ids") or [pair.get("pair_id")])[0]
        by_doc.setdefault(key, []).append(pair)
    keys = sorted(by_doc)
    random.Random(SEED).shuffle(keys)
    buckets = [[] for _ in range(folds)]
    for index, key in enumerate(keys):
        buckets[index % folds].extend(by_doc[key])
    return buckets


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


def evaluate(tokenizer, model, torch, pairs):
    cache = {}
    for pair in pairs:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=TOP)
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
    row = {
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
    dump = {key: {"answer": value["answer"], "pieces": [list(piece) for piece in value["pieces"]]} for key, value in cache.items()}
    return row, dump


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/rubert-tiny2")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--negatives", default="artifacts/rufacts/rufacts_negative.jsonl")
    parser.add_argument("--extra-negatives", default="")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out", default="artifacts/cv")
    parser.add_argument("--report", default="artifacts/cv_report.json")
    parser.add_argument("--tag", default="base")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    by_id = {pair.get("pair_id"): pair for pair in gold}
    ours = [by_id[pair_id] for name in ("train", "val") for pair_id in splits[name] if pair_id in by_id]
    ours = [pair for pair in ours if pair.get("model_output")]
    negatives = [pair for pair in read_jsonl(Path(args.negatives)) if pair.get("model_output")]
    extras = []
    if args.extra_negatives and Path(args.extra_negatives).exists():
        extras = [pair for pair in read_jsonl(Path(args.extra_negatives)) if pair.get("model_output")]
    buckets = group_folds(ours, args.folds)
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "tag": args.tag,
        "folds": args.folds,
        "epochs": args.epochs,
        "pairs": len(ours),
        "negatives": len(negatives),
        "hard_negatives": len(extras),
        "decode": {"token_threshold": TOKEN_THRESHOLD, "pair_gate": PAIR_GATE, "min_chars": MIN_CHARS, "snap": "expand"},
        "fold_sizes": [len(bucket) for bucket in buckets],
        "rows": [],
    }
    print(f"пар {len(ours)}, фолдов {args.folds}, негативов {len(negatives)}, размеры фолдов {report['fold_sizes']}", flush=True)

    for index in range(args.folds):
        clock = time.perf_counter()
        held = buckets[index]
        training = [pair for position, bucket in enumerate(buckets) if position != index for pair in bucket]
        held_ids = {pair.get("pair_id") for pair in held}
        extra_allowed = [pair for pair in extras if pair.get("origin_pair") not in held_ids]
        pool_negatives = negatives + extra_allowed
        torch, tokenizer, model = load_model(args.weights, labels=2)
        table = build_table(tokenizer, training + pool_negatives, "token")
        ours_ids = [pair["pair_id"] for pair in training if pair["pair_id"] in table]
        negative_ids = [pair["pair_id"] for pair in pool_negatives if pair["pair_id"] in table]
        keep = int(round(len(negative_ids) * len(ours_ids) / max(1, len(ours))))
        sampled = random.Random(SEED + index).sample(negative_ids, min(keep, len(negative_ids)))
        plans = stage_c_plans(ours_ids, sampled, args.epochs, ours_per_batch=OURS_PER_BATCH, helper_per_batch=NEG_PER_BATCH)
        print(f"фолд {index + 1}: обучение на {len(training)} наших + {len(sampled)} негативов, батчей {len(plans[0])}", flush=True)
        history = run_stage(torch, model, tokenizer, table, plans, HEAD_RATE, ENCODER_RATE)
        row, dump = evaluate(tokenizer, model, torch, held)
        checkpoint = save_checkpoint(model, tokenizer, out / f"fold_{index + 1}")
        (out / f"oof_{index + 1}.json").write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")
        row.update({"fold": index + 1, "held_pairs": len(held), "train_pairs": len(training), "hard_negatives_used": len(extra_allowed), "seconds": round(time.perf_counter() - clock, 1), "train": history[-1], "checkpoint": str(checkpoint)})
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
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for metric, values in summary.items():
        print(f"CV {metric}: {values['mean']} ± {values['std']}", flush=True)
    print(f"время {report['elapsed_s']} с, отчёт {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
