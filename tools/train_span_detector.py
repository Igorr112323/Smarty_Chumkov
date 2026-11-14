import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import (
    ACCUMULATE,
    BATCH,
    CHUNK_TOKENS,
    ENCODER_RATE,
    EPOCHS,
    HEAD_RATE,
    OVERLAP,
    SEED,
    TOP_K,
    WEIGHT_DECAY,
    WARMUP,
    choose_chunks,
    encode_pair,
    evaluate,
    examples_for,
    load_model,
    metrics,
    pair_probabilities,
    predictor_from_cache,
    save_checkpoint,
    scan_threshold,
    train,
)

SAMPLE_PAIRS = 20
STOP_LOSS = 0.35
STOP_AFTER = 5
GATE_TRAIN_F1 = 0.40


def train_pool(pairs):
    return [pair for pair in sorted(pairs, key=lambda item: item.get("pair_id") or "") if pair.get("model_output")]


def sample_pairs(pool, size=SAMPLE_PAIRS):
    positive = [pair for pair in pool if pair.get("spans")]
    negative = [pair for pair in pool if not pair.get("spans")]
    half = size // 2
    rng = random.Random(SEED)
    picked = rng.sample(positive, min(half, len(positive))) + rng.sample(negative, min(size - half, len(negative)))
    if len(picked) < size:
        rest = [pair for pair in pool if pair not in picked]
        picked += rng.sample(rest, min(size - len(picked), len(rest)))
    return sorted(picked, key=lambda item: item.get("pair_id") or "")


def score_pairs(tokenizer, model, torch, pairs, top=TOP_K, threshold=0.5):
    cache = {}
    started = time.perf_counter()
    for pair in pairs:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=top)
    elapsed = time.perf_counter() - started
    result = metrics(pairs, predictor_from_cache(cache, threshold))
    return result, cache, elapsed


def diagnostics(tokenizer, pairs, sample):
    rows = []
    for pair in sample:
        answer = pair.get("model_output") or ""
        gold = pair.get("spans") or []
        chunks = choose_chunks(tokenizer, pair, top=TOP_K)
        chosen = chunks[0] if chunks else {"start": 0, "end": 0, "text": "", "tokens": 0}
        encoded = encode_pair(tokenizer, chosen["text"], answer)
        marks = []
        for span in gold:
            text = answer[span["start"] : span["end"] + 1]
            manual = answer.find(text)
            inside = text in (chosen["text"] or "")
            matched = None
            if encoded is not None:
                for index, start, end in encoded["answer_tokens"]:
                    if start <= manual <= end or start <= manual + len(text) - 1 <= end:
                        matched = [start, end]
                        break
            marks.append(
                {
                    "gold": text[:60],
                    "gold_span": [span["start"], span["end"]],
                    "manual_find": manual,
                    "token_offsets": matched,
                    "inside_chunk": inside,
                }
            )
        rows.append(
            {
                "pair_id": pair.get("pair_id"),
                "answer_chars": len(answer),
                "document_chars": len(pair.get("document_text") or ""),
                "document_tokens": len(tokenizer(pair.get("document_text") or "", add_special_tokens=False)["input_ids"]),
                "chunks": len(chunks),
                "chosen_chars": len(chosen["text"]),
                "chosen_tokens": chosen["tokens"],
                "gold_spans": len(gold),
                "gold_checks": marks,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/rubert-tiny2")
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--out", default="artifacts/span_detector")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--top", type=int, default=TOP_K)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=SAMPLE_PAIRS)
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
    pool = train_pool(train_pairs)
    if args.limit:
        pool = pool[: args.limit]
    with_spans = sum(1 for pair in pool if pair.get("spans"))
    sample = sample_pairs(pool, args.sample)
    print(
        f"ФАКТ train-пар: всего {len(train_pairs)}, с answer {len(train_pool(train_pairs))}, "
        f"в обучении {len(pool)}, из них со фрагментами {with_spans} ({with_spans / max(1, len(pool)):.3f})",
        flush=True,
    )
    print(f"ФАКТ выборка train-гейта: {len(sample)} пар, со фрагментами {sum(1 for p in sample if p.get('spans'))}", flush=True)
    print(
        f"чанки: {CHUNK_TOKENS} токенов, перекрытие {OVERLAP}, top-k {args.top}, batch {BATCH} x накопление {ACCUMULATE} "
        f"(эффективный {BATCH * ACCUMULATE}), эпох {args.epochs}, "
        f"LR голова {HEAD_RATE} / энкодер {ENCODER_RATE}, wd {WEIGHT_DECAY}, warmup {WARMUP}",
        flush=True,
    )

    torch, tokenizer, model = load_model(args.weights)
    rows = examples_for(tokenizer, pool, top=1)
    print(f"примеров собрано: {len(rows)}", flush=True)

    def on_epoch(entry):
        result, _, _ = score_pairs(tokenizer, model, torch, sample, top=args.top)
        entry["train_f1_sample"] = result["span_level"]["f1"]
        entry["train_fpr_sample"] = result["fpr"]
        print(
            f"эпоха {entry['epoch']}: loss {entry['loss']}, train F1 ({len(sample)} пар) "
            f"{entry['train_f1_sample']}, FPR {entry['train_fpr_sample']}, lr {entry['lr_head']}",
            flush=True,
        )

    outcome = train(
        torch,
        model,
        tokenizer,
        rows,
        epochs=args.epochs,
        batch=BATCH,
        accumulate=ACCUMULATE,
        head_rate=HEAD_RATE,
        encoder_rate=ENCODER_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup=WARMUP,
        seed=SEED,
        on_epoch=on_epoch,
        stop_loss=STOP_LOSS,
        stop_after=STOP_AFTER,
    )
    checkpoint = save_checkpoint(model, tokenizer, args.out)
    train_result, train_cache, train_elapsed = score_pairs(tokenizer, model, torch, sample, top=args.top)
    print(f"ШАГ A train F1 ({len(sample)} пар): {train_result['span_level']['f1']}", flush=True)

    report = {
        "model": str(args.weights),
        "checkpoint": str(checkpoint),
        "top_k": args.top,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap": OVERLAP,
        "batch": BATCH,
        "accumulate": ACCUMULATE,
        "batch_effective": BATCH * ACCUMULATE,
        "epochs_planned": args.epochs,
        "epochs_done": len(outcome["history"]),
        "steps_per_epoch": outcome["steps_per_epoch"],
        "stopped_early": outcome["stopped_early"],
        "loss": outcome["history"],
        "train_pairs_total": len(train_pairs),
        "train_pairs_with_answer": len(train_pool(train_pairs)),
        "train_pairs_used": len(pool),
        "train_pairs_with_spans": with_spans,
        "train_examples": len(rows),
        "sample_ids": [pair.get("pair_id") for pair in sample],
        "train_f1_sample": train_result["span_level"]["f1"],
        "train_span_metrics": train_result["span_level"],
        "train_fpr_sample": train_result["fpr"],
        "gate_train_f1": GATE_TRAIN_F1,
        "gate_passed": train_result["span_level"]["f1"] >= GATE_TRAIN_F1,
        "train_eval_s": round(train_elapsed, 2),
        "seed": SEED,
    }

    if not report["gate_passed"]:
        report["diagnostics"] = diagnostics(tokenizer, val_pairs, sample)
    else:
        val_result, val_cache, val_elapsed = score_pairs(
            tokenizer, model, torch, val_pairs[: args.eval_limit] if args.eval_limit else val_pairs, top=args.top
        )
        usable = val_pairs[: args.eval_limit] if args.eval_limit else val_pairs
        best, scan = scan_threshold(val_cache, usable)
        tuned = metrics(usable, predictor_from_cache(val_cache, best["threshold"]))
        chars = sum(len(pair.get("model_output") or "") for pair in usable)
        report.update(
            {
                "val_pairs": tuned["n_pairs"],
                "val_span": tuned["span_level"],
                "val_token": tuned["token_level"],
                "val_pair": tuned["pair_level"],
                "val_fpr": tuned["fpr"],
                "val_type_accuracy": tuned["type_accuracy"],
                "val_threshold": best["threshold"],
                "val_best": best,
                "val_scan": scan,
                "val_at_050": val_result["span_level"],
                "val_elapsed_s": round(val_elapsed, 2),
                "val_ms_per_1000_chars": round(val_elapsed / max(1, chars) * 1000 * 1000, 3),
                "val_chars": chars,
            }
        )
        print(f"ШАГ B val: F1 {best['f1']} при пороге {best['threshold']}, FPR {best['fpr']}", flush=True)

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    print(json.dumps({key: report[key] for key in report if key not in ("val_scan", "diagnostics")}, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
