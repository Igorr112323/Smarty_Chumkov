import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import metrics, read_jsonl
from spanver.module import confidence, predict_type
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import (
    ACCUMULATE,
    BATCH,
    BATCH_NEGATIVE,
    BATCH_POSITIVE,
    EPOCH_BATCHES,
    EPOCHS,
    GATE_GRID,
    SEED,
    TOP_K,
    VAL_GRID,
    epoch_batches,
    evaluate,
    examples_for,
    extended_diagnostics,
    gate_table,
    load_model,
    make_optimizer,
    pair_probabilities,
    predictor_from_cache,
    rows_by_id,
    run_epoch,
    save_checkpoint,
    scan_threshold,
    spans_from,
    token_stats,
)

GATE_PAIRS = 100
GATE_HALF = 50
LEVEL_ONE = 0.40
LEVEL_TWO = 0.20
SPREAD_MIN = 0.02
SPAN_FLOOR = 0.10


def pick(pairs):
    return [pair for pair in pairs if pair.get("model_output")]


def stratified(pairs, size=GATE_PAIRS, seed=SEED):
    positive = [pair for pair in sorted(pairs, key=lambda item: item.get("pair_id") or "") if pair.get("spans")]
    negative = [pair for pair in sorted(pairs, key=lambda item: item.get("pair_id") or "") if not pair.get("spans")]
    rng = random.Random(seed)
    half = size // 2
    chosen = rng.sample(positive, min(half, len(positive))) + rng.sample(negative, min(size - half, len(negative)))
    return sorted(chosen, key=lambda item: item.get("pair_id") or "")


def cached(tokenizer, model, torch, pairs, top=TOP_K):
    store = {}
    started = time.perf_counter()
    for pair in pairs:
        store[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=top)
    return store, time.perf_counter() - started


def selfcheck(row):
    precision = row["precision"]
    recall = row["recall"]
    expected = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(abs(expected - row["f1"]), 4) <= 0.001


def describe(epoch, row, best, span, batch_info, extra):
    return {
        "epoch": epoch,
        "loss": extra["loss"],
        "lr_head": extra["lr_head"],
        "batch_positive": batch_info[0],
        "batch_negative": batch_info[1],
        "at_050": row,
        "at_best": best,
        "span_at_best": span,
        "selfcheck": selfcheck(row) and selfcheck(best),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/rubert-tiny2")
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--out", default="artifacts/span_detector_v4")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--top", type=int, default=TOP_K)
    parser.add_argument("--report", default="")
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--no-early-exit", action="store_true")
    parser.add_argument("--val-all", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    splits = load_splits()
    gold = read_jsonl(Path(args.data))
    train_labeled = ordered_pairs(gold, splits["train"])
    val_labeled = ordered_pairs(gold, splits["val"])
    pool = pick(train_labeled)
    positive = [pair for pair in pool if pair.get("spans")]
    negative = [pair for pair in pool if not pair.get("spans")]
    share = round(len(positive) / max(1, len(pool)), 4)
    print(
        f"ФАКТ train: размеченных {len(train_labeled)}, в обучении {len(pool)}, позитивов {len(positive)} "
        f"({share}), негативов {len(negative)}; неразмеченных из сплита исключено "
        f"{len(splits['train']) - len(train_labeled)}",
        flush=True,
    )
    print(f"ФАКТ val: размеченных {len(val_labeled)} из {len(splits['val'])} в сплите", flush=True)
    gate = stratified(pool)
    print(
        f"ФАКТ гейт: {len(gate)} пар, позитивов {sum(1 for p in gate if p.get('spans'))}, "
        f"негативов {sum(1 for p in gate if not p.get('spans'))}, seed {SEED}",
        flush=True,
    )
    print(
        f"эпоха: {EPOCH_BATCHES} батчей по {BATCH_POSITIVE} позитивов + {BATCH_NEGATIVE} негативов, "
        f"batch {BATCH} x накопление {ACCUMULATE}, эпох {args.epochs}, top-k {args.top}",
        flush=True,
    )

    torch, tokenizer, model = load_model(args.weights)
    rows = examples_for(tokenizer, pool, top=1)
    table = rows_by_id(rows)
    print(f"примеров собрано: {len(rows)}", flush=True)

    total_steps = max(1, args.epochs * EPOCH_BATCHES)
    optimizer, scheduler = make_optimizer(torch, model, total_steps)
    positive_ids = [row["pair_id"] for row in rows if row["positive"]]
    negative_ids = [row["pair_id"] for row in rows if not row["positive"]]

    history = []
    gate_ok = False
    stop_reason = ""
    best_score = -1.0
    best_path = Path(f"{args.out}_best")
    last_path = Path(f"{args.out}_last")
    for epoch in range(1, args.epochs + 1):
        plan = epoch_batches(positive_ids, negative_ids, seed=SEED + epoch)
        batch_info = (
            sum(1 for pair_id in plan[0] if table[pair_id]["positive"]),
            sum(1 for pair_id in plan[0] if not table[pair_id]["positive"]),
        )
        extra = run_epoch(torch, model, tokenizer, table, plan, optimizer, scheduler)
        store, seconds = cached(tokenizer, model, torch, gate, top=args.top)
        table_gate = gate_table(store, gate)
        entry = describe(epoch, table_gate["fixed"], table_gate["best"], table_gate["span_at_best"], batch_info, extra)
        entry["gate_seconds"] = round(seconds, 1)
        history.append(entry)
        row = table_gate["fixed"]
        best = table_gate["best"]
        span = table_gate["span_at_best"]
        print(
            f"эпоха {epoch}: loss {extra['loss']} lr_head {extra['lr_head']} батч {batch_info[0]}/{batch_info[1]} | "
            f"0,5: TP {row['tp']} FP {row['fp']} FN {row['fn']} P {row['precision']} R {row['recall']} F1 {row['f1']} | "
            f"{best['threshold']}: TP {best['tp']} FP {best['fp']} FN {best['fn']} P {best['precision']} R {best['recall']} F1 {best['f1']} | "
            f"span P {span['precision']} R {span['recall']} F1 {span['f1']} | "
            f"FPR {row['pair_fpr']} | p̄ поз {row['mean_p_positive_pairs']} негат {row['mean_p_negative_pairs']} max {row['max_p']}",
            flush=True,
        )
        if best["f1"] > best_score:
            resave = save_checkpoint(model, tokenizer, best_path)
            best_score = best["f1"]
            print(f"  чекпоинт лучший по гейту: {resave} (F1 {best_score})", flush=True)
        spread = abs(row["mean_p_positive_pairs"] - row["mean_p_negative_pairs"])
        if epoch == 4 and spread < SPREAD_MIN:
            stop_reason = f"к 4-й эпохе средняя p не расходится (разрыв {round(spread, 4)} < {SPREAD_MIN})"
            break
        if len(history) >= 3:
            previous = history[-3]["at_best"]["f1"]
            growth = best["f1"] > previous
        else:
            growth = False
        span_floor = span["f1"] >= SPAN_FLOOR
        if args.no_early_exit:
            if best["f1"] >= LEVEL_ONE and span_floor:
                print(f"  уровень 1 достигнут на эпохе {epoch} (ранний выход отключён флагом)", flush=True)
            growth = False
        if best["f1"] >= LEVEL_ONE and span_floor and not args.no_early_exit:
            gate_ok = True
            stop_reason = f"уровень 1: лучший token F1 {best['f1']} >= {LEVEL_ONE}, span F1 {span['f1']}"
            break
        if best["f1"] >= LEVEL_ONE and not span_floor:
            print(
                f"  уровень 1 формально (token F1 {best['f1']}), но span F1 {span['f1']} < {SPAN_FLOOR}: "
                "признак константного предсказания, обучение продолжается",
                flush=True,
            )
        if LEVEL_TWO <= best["f1"] < LEVEL_ONE and growth and epoch >= 4 and span_floor and not args.no_early_exit:
            gate_ok = True
            stop_reason = f"уровень 2: лучший token F1 {best['f1']} в [{LEVEL_TWO}; {LEVEL_ONE}) и рост за 3 эпохи"
            break
        if epoch >= 3 and history[1]["at_best"]["f1"] < 0.05 and history[2]["at_best"]["f1"] < 0.05:
            stop_reason = "лучший token F1 < 0,05 на эпохах 2 и 3"
            break

    last = save_checkpoint(model, tokenizer, last_path)
    final = history[-1]["at_best"]["f1"]
    if args.no_early_exit and final >= LEVEL_TWO and history[-1]["span_at_best"]["f1"] >= SPAN_FLOOR:
        gate_ok = True
        stop_reason = f"без раннего выхода, финальная эпоха: лучший token F1 {final}, span F1 {history[-1]['span_at_best']['f1']}"
    if not gate_ok and final < LEVEL_TWO and not stop_reason:
        stop_reason = f"финальный лучший token F1 {final} < {LEVEL_TWO}"

    report = {
        "model": str(args.weights),
        "checkpoint_last": str(last),
        "checkpoint_best": str(best_path),
        "epochs_planned": args.epochs,
        "epochs_done": len(history),
        "epoch_batches": EPOCH_BATCHES,
        "batch": BATCH,
        "accumulate": ACCUMULATE,
        "batch_effective": BATCH * ACCUMULATE,
        "batch_positive": BATCH_POSITIVE,
        "batch_negative": BATCH_NEGATIVE,
        "top_k": args.top,
        "gate_grid": list(GATE_GRID),
        "train_labeled": len(train_labeled),
        "train_used": len(pool),
        "train_positive": len(positive),
        "train_negative": len(negative),
        "train_positive_share": share,
        "train_unlabeled_excluded": len(splits["train"]) - len(train_labeled),
        "train_examples": len(rows),
        "val_labeled": len(val_labeled),
        "val_split": len(splits["val"]),
        "gate_pairs": len(gate),
        "gate_positive": sum(1 for p in gate if p.get("spans")),
        "gate_ids": [pair.get("pair_id") for pair in gate],
        "history": history,
        "gate_passed": gate_ok,
        "gate_best": final,
        "stop_reason": stop_reason,
        "seed": SEED,
    }
    print(f"ВЕРДИКТ ГЕЙТА: {'пройден' if gate_ok else 'СТОП'} ({stop_reason})", flush=True)

    if gate_ok:
        val_store, val_seconds = cached(tokenizer, model, torch, val_labeled, top=args.top)
        val_metrics = {}
        for threshold in VAL_GRID:
            val_metrics[str(threshold)] = metrics(val_labeled, predictor_from_cache(val_store, threshold))
        main_row = val_metrics["0.5"]
        best_entry = max(VAL_GRID, key=lambda value: val_metrics[str(value)]["span_level"]["f1"])
        chars = sum(len(pair.get("model_output") or "") for pair in val_labeled)
        report["val"] = {
            "pairs": len(val_labeled),
            "at_050": {
                "span": main_row["span_level"],
                "token": main_row["token_level"],
                "pair": main_row["pair_level"],
                "fpr": main_row["fpr"],
                "type": main_row["type_accuracy"],
            },
            "best_threshold": best_entry,
            "at_best": {
                "span": val_metrics[str(best_entry)]["span_level"],
                "token": val_metrics[str(best_entry)]["token_level"],
                "pair": val_metrics[str(best_entry)]["pair_level"],
                "fpr": val_metrics[str(best_entry)]["fpr"],
                "type": val_metrics[str(best_entry)]["type_accuracy"],
            },
            "sweep": [
                {
                    "threshold": value,
                    "span_f1": val_metrics[str(value)]["span_level"]["f1"],
                    "fpr": val_metrics[str(value)]["fpr"],
                }
                for value in VAL_GRID
            ],
            "seconds": round(val_seconds, 1),
            "chars": chars,
            "ms_per_1000_chars": round(val_seconds / max(1, chars) * 1000 * 1000, 3),
            "threshold_contract": 0.2595,
            "contract_passed": val_metrics[str(best_entry)]["span_level"]["f1"] > 0.2595,
        }
        if args.val_all:
            extra_file = Path("artifacts/pairs_unlabeled.jsonl")
            everything = list(gold)
            if extra_file.exists():
                everything.extend(read_jsonl(extra_file))
            val_split = ordered_pairs(everything, splits["val"])
            split_store, split_seconds = cached(tokenizer, model, torch, val_split, top=args.top)
            split_metrics = {}
            for threshold in VAL_GRID:
                split_metrics[str(threshold)] = metrics(val_split, predictor_from_cache(split_store, threshold))
            split_best = max(VAL_GRID, key=lambda value: split_metrics[str(value)]["span_level"]["f1"])
            split_chars = sum(len(pair.get("model_output") or "") for pair in val_split)
            report["val_all"] = {
                "pairs": len(val_split),
                "labeled": len(val_labeled),
                "unlabeled": len(val_split) - len(val_labeled),
                "at_050": {
                    "span": split_metrics["0.5"]["span_level"],
                    "token": split_metrics["0.5"]["token_level"],
                    "pair": split_metrics["0.5"]["pair_level"],
                    "fpr": split_metrics["0.5"]["fpr"],
                    "type": split_metrics["0.5"]["type_accuracy"],
                },
                "best_threshold": split_best,
                "at_best": {
                    "span": split_metrics[str(split_best)]["span_level"],
                    "token": split_metrics[str(split_best)]["token_level"],
                    "pair": split_metrics[str(split_best)]["pair_level"],
                    "fpr": split_metrics[str(split_best)]["fpr"],
                    "type": split_metrics[str(split_best)]["type_accuracy"],
                },
                "sweep": [
                    {
                        "threshold": value,
                        "span_f1": split_metrics[str(value)]["span_level"]["f1"],
                        "fpr": split_metrics[str(value)]["fpr"],
                    }
                    for value in VAL_GRID
                ],
                "seconds": round(split_seconds, 1),
                "ms_per_1000_chars": round(split_seconds / max(1, split_chars) * 1000 * 1000, 3),
                "contract_passed": split_metrics[str(split_best)]["span_level"]["f1"] > 0.2595,
            }
            print(
                f"val_all {len(val_split)} (размеченных {len(val_labeled)}): span F1 при 0,5 "
                f"{split_metrics['0.5']['span_level']['f1']} FPR {split_metrics['0.5']['fpr']} | лучший порог "
                f"{split_best} span F1 {split_metrics[str(split_best)]['span_level']['f1']}",
                flush=True,
            )

        print(
            f"val {len(val_labeled)}: при 0,5 span F1 {main_row['span_level']['f1']} FPR {main_row['fpr']} | "
            f"лучший порог {best_entry} span F1 {val_metrics[str(best_entry)]['span_level']['f1']} "
            f"FPR {val_metrics[str(best_entry)]['fpr']} | "
            f"{report['val']['ms_per_1000_chars']} мс/1000 знаков",
            flush=True,
        )
    else:
        store, seconds = cached(tokenizer, model, torch, gate, top=args.top)
        report["diagnostics"] = extended_diagnostics(tokenizer, store, gate)
        report["diagnostics"]["seconds"] = round(seconds, 1)
        print("расширенная диагностика записана в отчёт", flush=True)

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    print(json.dumps({key: report[key] for key in report if key != "history"}, ensure_ascii=False, indent=2)[:4000], flush=True)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
