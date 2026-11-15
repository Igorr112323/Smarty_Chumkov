import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import IOU_THRESHOLD, iou, metrics, read_jsonl
from spanver.splits import load_splits, ordered_pairs
from spanver.stage_train import (
    ENCODER_RATE,
    HEAD_RATE,
    PAIR_WEIGHT,
    STAGE_C_DIVIDER,
    build_table,
    run_stage,
    stage_a_plans,
    stage_b_plans,
    stage_c_plans,
)
from spanver.span_detector import load_model, pair_probabilities, save_checkpoint

STAGE_A_EPOCHS = 6
STAGE_B_EPOCHS = 2
STAGE_C_EPOCHS = 8
STAGE_B_SIDE = 1382
THRESHOLD = 0.5
FPR_LIMIT = 0.40
TARGET_TOKEN_F1 = 0.50
DOMAIN_FLOOR = 0.2835


def cached(tokenizer, model, torch, pairs, top=4):
    store = {}
    started = time.perf_counter()
    for pair in pairs:
        store[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=top)
    return store, time.perf_counter() - started


def stage_metrics(tokenizer, model, torch, pairs, top=4):
    store, seconds = cached(tokenizer, model, torch, pairs, top=top)
    tokens = decode.token_counts(store, pairs, THRESHOLD)
    spans = metrics(pairs, decode.predictor_from_cache(store, THRESHOLD))
    return {
        "pairs": len(pairs),
        "token_f1": tokens["f1"],
        "token_precision": tokens["precision"],
        "token_recall": tokens["recall"],
        "fpr": tokens["fpr"],
        "span_f1": spans["span_level"]["f1"],
        "span_precision": spans["span_level"]["precision"],
        "span_recall": spans["span_level"]["recall"],
        "pair_f1": spans["pair_level"]["f1"],
        "seconds": round(seconds, 1),
    }, store


def best_threshold(store, pairs, fpr_limit=FPR_LIMIT):
    rows = decode.sweep(store, pairs)
    allowed = [row for row in rows if row["fpr"] <= fpr_limit]
    pool = allowed or rows
    best = max(pool, key=lambda row: (row["f1"], row["threshold"]))
    return best, rows


def type_counts(cache, pairs, threshold=THRESHOLD):
    predict = decode.predictor_from_cache(cache, threshold)
    matched = 0
    correct = 0
    for pair in pairs:
        gold = pair.get("spans") or []
        used = set()
        for span in predict(pair):
            for index, truth in enumerate(gold):
                if index in used:
                    continue
                if iou(span["start"], span["end"], truth["start"], truth["end"]) >= IOU_THRESHOLD:
                    used.add(index)
                    matched += 1
                    if span.get("type") == truth.get("type"):
                        correct += 1
                    break
    return {"correct": correct, "matched": matched, "accuracy": round(correct / matched, 4) if matched else 0.0}


def speed_report(tokenizer, model, torch, pairs, top=4):
    longest = max(pairs, key=lambda pair: len(pair.get("document_text") or ""))
    clock = time.perf_counter()
    pair_probabilities(tokenizer, model, torch, longest, top=top)
    elapsed = time.perf_counter() - clock
    document_chars = len(longest.get("document_text") or "")
    answer_chars = len(longest.get("model_output") or "")
    return {
        "pair_id": longest.get("pair_id"),
        "document_chars": document_chars,
        "answer_chars": answer_chars,
        "seconds": round(elapsed, 3),
        "ms_per_1000_document_chars": round(elapsed / max(1, document_chars) * 1000 * 1000, 3),
        "ms_per_1000_answer_chars": round(elapsed / max(1, answer_chars) * 1000 * 1000, 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/rubert-tiny2")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--out", default="artifacts/stages")
    parser.add_argument("--report", default="artifacts/stage_report.json")
    parser.add_argument("--stage-a", type=int, default=STAGE_A_EPOCHS)
    parser.add_argument("--stage-b", type=int, default=STAGE_B_EPOCHS)
    parser.add_argument("--stage-c", type=int, default=STAGE_C_EPOCHS)
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    our_train = [pair for pair in ordered_pairs(gold, splits["train"]) if pair.get("model_output")]
    val150 = [pair for pair in ordered_pairs(gold, splits["val"]) if pair.get("model_output")]
    everything = list(gold)
    extra = Path("artifacts/pairs_unlabeled.jsonl")
    if extra.exists():
        everything.extend(read_jsonl(extra))
    val637 = [pair for pair in ordered_pairs(everything, splits["val"]) if pair.get("model_output")]
    localized = read_jsonl(Path("artifacts/rufacts/rufacts_localized.jsonl"))
    pairlevel = read_jsonl(Path("artifacts/rufacts/rufacts_pairlevel.jsonl"))
    negative = read_jsonl(Path("artifacts/rufacts/rufacts_negative.jsonl"))
    print(
        f"данные: наши train {len(our_train)}, val150 {len(val150)}, val637 {len(val637)}, "
        f"RuFacts localized {len(localized)}, pairlevel {len(pairlevel)}, negative {len(negative)}",
        flush=True,
    )

    torch, tokenizer, model = load_model(args.weights)
    print("сборка строк ступени A", flush=True)
    table_token = build_table(tokenizer, localized + our_train, "token")
    print("сборка строк ступени B", flush=True)
    table_pair = build_table(tokenizer, pairlevel + negative, "pair")
    print(f"строк: token {len(table_token)}, pair {len(table_pair)}", flush=True)

    report = {
        "weights": args.weights,
        "top_k": args.top,
        "pair_weight": PAIR_WEIGHT,
        "head_rate": HEAD_RATE,
        "encoder_rate": ENCODER_RATE,
        "stage_c_divider": STAGE_C_DIVIDER,
        "threshold": THRESHOLD,
        "counts": {
            "our_train": len(our_train),
            "val150": len(val150),
            "val637": len(val637),
            "localized": len(localized),
            "pairlevel": len(pairlevel),
            "negative": len(negative),
        },
        "stages": {},
    }

    localized_ids = [pair["pair_id"] for pair in localized if pair["pair_id"] in table_token]
    plans_a = stage_a_plans(localized_ids, args.stage_a)
    history_a = run_stage(torch, model, tokenizer, table_token, plans_a, HEAD_RATE, ENCODER_RATE)
    for entry in history_a:
        print(f"  A эпоха {entry['epoch']}: token {entry['token_loss']} pair {entry['pair_loss']} lr {entry['lr_head']}", flush=True)
    stage_a, _ = stage_metrics(tokenizer, model, torch, val150, top=args.top)
    save_checkpoint(model, tokenizer, Path(args.out) / "stage_a")
    report["stages"]["A"] = {"history": history_a, "val150": stage_a}
    print(f"ступень A: val150 token F1 {stage_a['token_f1']} span F1 {stage_a['span_f1']} FPR {stage_a['fpr']}", flush=True)
    if stage_a["span_f1"] < DOMAIN_FLOOR:
        print(f"ВНИМАНИЕ: span F1 после ступени A {stage_a['span_f1']} < {DOMAIN_FLOOR}", flush=True)

    positive_ids = [pair["pair_id"] for pair in pairlevel if pair["pair_id"] in table_pair][:STAGE_B_SIDE]
    negative_ids = [pair["pair_id"] for pair in negative if pair["pair_id"] in table_pair][:STAGE_B_SIDE]
    plans_b = stage_b_plans(positive_ids, negative_ids, args.stage_b)
    history_b = run_stage(torch, model, tokenizer, table_pair, plans_b, HEAD_RATE, ENCODER_RATE)
    for entry in history_b:
        print(f"  B эпоха {entry['epoch']}: token {entry['token_loss']} pair {entry['pair_loss']} lr {entry['lr_head']}", flush=True)
    stage_b, _ = stage_metrics(tokenizer, model, torch, val150, top=args.top)
    save_checkpoint(model, tokenizer, Path(args.out) / "stage_b")
    report["stages"]["B"] = {"history": history_b, "val150": stage_b, "positive_ids": len(positive_ids), "negative_ids": len(negative_ids)}
    print(f"ступень B: val150 token F1 {stage_b['token_f1']} span F1 {stage_b['span_f1']} FPR {stage_b['fpr']}", flush=True)
    if stage_b["span_f1"] < DOMAIN_FLOOR:
        print(f"ВНИМАНИЕ: span F1 после ступени B {stage_b['span_f1']} < {DOMAIN_FLOOR}", flush=True)

    our_ids = [pair["pair_id"] for pair in our_train if pair["pair_id"] in table_token]
    helper_ids = [
        pair["pair_id"]
        for pair in (localized + pairlevel + negative)
        if pair["pair_id"] in table_pair or pair["pair_id"] in table_token
    ]
    combined = dict(table_pair)
    combined.update(table_token)
    plans_c = stage_c_plans(our_ids, helper_ids, args.stage_c)
    history_c = run_stage(
        torch, model, tokenizer, combined, plans_c, HEAD_RATE / STAGE_C_DIVIDER, ENCODER_RATE / STAGE_C_DIVIDER
    )
    best_epoch = None
    best_value = -1.0
    for entry in history_c:
        print(f"  C эпоха {entry['epoch']}: token {entry['token_loss']} pair {entry['pair_loss']} lr {entry['lr_head']}", flush=True)
    save_checkpoint(model, tokenizer, Path(args.out) / "stage_c_last")
    stage_c, store150 = stage_metrics(tokenizer, model, torch, val150, top=args.top)
    report["stages"]["C"] = {"history": history_c, "val150": stage_c, "helper_ids": len(helper_ids), "ours": len(our_ids)}
    print(f"ступень C: val150 token F1 {stage_c['token_f1']} span F1 {stage_c['span_f1']} FPR {stage_c['fpr']}", flush=True)

    best150, rows150 = best_threshold(store150, val150)
    store637, seconds637 = cached(tokenizer, model, torch, val637, top=args.top)
    best637, rows637 = best_threshold(store637, val637)
    chosen = best150["threshold"]
    span150 = metrics(val150, decode.predictor_from_cache(store150, chosen))
    span637 = metrics(val637, decode.predictor_from_cache(store637, chosen))
    report["final"] = {
        "checkpoint": str(Path(args.out) / "stage_c_last"),
        "threshold_chosen": chosen,
        "threshold_rule": f"максимум token F1 при FPR <= {FPR_LIMIT} на val150",
        "val150_at_chosen": {
            "tokens": decode.token_counts(store150, val150, chosen),
            "span": span150["span_level"],
            "pair": span150["pair_level"],
            "fpr": span150["fpr"],
            "type": type_counts(store150, val150, chosen),
        },
        "val637_at_chosen": {
            "tokens": decode.token_counts(store637, val637, chosen),
            "span": span637["span_level"],
            "pair": span637["pair_level"],
            "fpr": span637["fpr"],
            "type": type_counts(store637, val637, chosen),
        },
        "sweep150": rows150,
        "sweep637": rows637,
        "best150": best150,
        "best637": best637,
        "seconds637": round(seconds637, 1),
        "speed": speed_report(tokenizer, model, torch, val637, top=args.top),
        "gate_target": TARGET_TOKEN_F1,
        "gate_passed": bool(best150["f1"] >= TARGET_TOKEN_F1 and best150["fpr"] <= FPR_LIMIT),
    }
    final = report["final"]
    print(
        f"итог: порог {chosen} | val150 token F1 {final['val150_at_chosen']['tokens']['f1']} "
        f"span F1 {span150['span_level']['f1']} FPR {final['val150_at_chosen']['fpr']} | "
        f"val637 token F1 {final['val637_at_chosen']['tokens']['f1']} span F1 {span637['span_level']['f1']} | "
        f"гейт {final['gate_passed']}",
        flush=True,
    )
    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
