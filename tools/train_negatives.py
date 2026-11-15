import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import metrics, read_jsonl
from spanver.splits import load_splits, ordered_pairs
from spanver.stage_train import HEAD_RATE, ENCODER_RATE, build_table, run_stage, stage_c_plans
from spanver.span_detector import load_model, pair_probabilities, save_checkpoint

OURS_PER_BATCH = 8
NEGATIVES_PER_BATCH = 8
EPOCHS = 3
THRESHOLD = 0.5
GRID = tuple(round(0.20 + 0.01 * step, 2) for step in range(31))
FPR_LIMIT = 0.40
BASE_TOKEN_F1 = 0.4566
BASE_SPAN_F1 = 0.0569


def eval_val(tokenizer, model, torch, pairs, top=4):
    cache = {}
    clock = time.perf_counter()
    for pair in pairs:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=top)
    seconds = time.perf_counter() - clock
    tokens = decode.token_counts(cache, pairs, THRESHOLD)
    spans = metrics(pairs, decode.predictor_from_cache(cache, THRESHOLD))
    rows = []
    for threshold in GRID:
        tokens_grid = decode.token_counts(cache, pairs, threshold)
        spans_grid = metrics(pairs, decode.predictor_from_cache(cache, threshold))
        rows.append(
            {
                "threshold": threshold,
                "token_f1": tokens_grid["f1"],
                "fpr_pair": spans_grid["fpr"],
                "span_f1": spans_grid["span_level"]["f1"],
            }
        )
    allowed = [row for row in rows if row["fpr_pair"] <= FPR_LIMIT]
    best = max(allowed or rows, key=lambda row: row["token_f1"])
    return {
        "pairs": len(pairs),
        "seconds": round(seconds, 1),
        "at_050": {
            "token_precision": tokens["precision"],
            "token_recall": tokens["recall"],
            "token_f1": tokens["f1"],
            "span_f1": spans["span_level"]["f1"],
            "fpr_pair": spans["fpr"],
            "pair_f1": spans["pair_level"]["f1"],
        },
        "grid_best": best,
        "grid": rows,
    }


def split_report(gold):
    counts = {}
    for pair in gold:
        name = pair.get("split") or "unknown"
        entry = counts.setdefault(name, {"pairs": 0, "sources": {}})
        entry["pairs"] += 1
        source = pair.get("source") or "unknown"
        entry["sources"][source] = entry["sources"].get(source, 0) + 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="artifacts/span_detector_v5_last")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--gold1200", default="artifacts/gold_1200.jsonl")
    parser.add_argument("--negatives", default="artifacts/rufacts/rufacts_negative.jsonl")
    parser.add_argument("--out", default="artifacts/negatives")
    parser.add_argument("--report", default="artifacts/negatives_report.json")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    our_train = [pair for pair in ordered_pairs(gold, splits["train"]) if pair.get("model_output")]
    our_val = [pair for pair in ordered_pairs(gold, splits["val"]) if pair.get("model_output")]
    negatives = [pair for pair in read_jsonl(Path(args.negatives)) if pair.get("model_output")]
    both = read_jsonl(Path(args.gold1200)) if Path(args.gold1200).exists() else gold
    composition = split_report(both)
    train_ids = {pair.get("pair_id") for pair in our_train}
    test_ids = set(splits["test"])
    extra_ids = {pair.get("pair_id") for pair in both if (pair.get("split") or "") == "extra"}
    print(
        f"gold_1200: {json.dumps(composition, ensure_ascii=False)} | extra в train: {len(train_ids & extra_ids)} | "
        f"extra в test: {len(test_ids & extra_ids)} | наши train {len(our_train)}, val {len(our_val)}, "
        f"RuFacts-негативы {len(negatives)}",
        flush=True,
    )

    torch, tokenizer, model = load_model(args.base)
    base_eval = eval_val(tokenizer, model, torch, our_val, top=args.top)
    print(
        f"база {args.base}: token F1 при 0,5 {base_eval['at_050']['token_f1']} span F1 {base_eval['at_050']['span_f1']} "
        f"FPR {base_eval['at_050']['fpr_pair']} | лучший по сетке {base_eval['grid_best']['token_f1']} "
        f"(порог {base_eval['grid_best']['threshold']})",
        flush=True,
    )

    table = build_table(tokenizer, our_train + negatives, "token")
    ours_ids = [pair["pair_id"] for pair in our_train if pair["pair_id"] in table]
    negative_ids = [pair["pair_id"] for pair in negatives if pair["pair_id"] in table]
    plans = stage_c_plans(
        ours_ids, negative_ids, args.epochs, ours_per_batch=OURS_PER_BATCH, helper_per_batch=NEGATIVES_PER_BATCH
    )
    print(f"батчей на эпоху: {len(plans[0])}, строк в таблице {len(table)}", flush=True)

    report = {
        "base": args.base,
        "base_eval": base_eval,
        "threshold_fixed": THRESHOLD,
        "epochs": args.epochs,
        "batch": f"{OURS_PER_BATCH} наших + {NEGATIVES_PER_BATCH} негативов",
        "head_rate": HEAD_RATE,
        "encoder_rate": ENCODER_RATE,
        "data": {"our_train": len(our_train), "negatives": len(negatives)},
        "gold1200": composition,
        "extra_in_train": len(train_ids & extra_ids),
        "extra_in_test": len(test_ids & extra_ids),
        "epochs_log": [],
    }

    history = run_stage(
        torch,
        model,
        tokenizer,
        table,
        plans,
        HEAD_RATE,
        ENCODER_RATE,
    )
    stop = ""
    for index, entry in enumerate(history, 1):
        checkpoint = save_checkpoint(model, tokenizer, Path(args.out) / f"epoch_{index}")
        result = eval_val(tokenizer, model, torch, our_val, top=args.top)
        report["epochs_log"].append({"epoch": index, "train": entry, "val150": result, "checkpoint": str(checkpoint)})
        print(
            f"эпоха {index}: token {entry['token_loss']} pair {entry['pair_loss']} lr {entry['lr_head']} | "
            f"0,5: token P {result['at_050']['token_precision']} R {result['at_050']['token_recall']} "
            f"F1 {result['at_050']['token_f1']} span F1 {result['at_050']['span_f1']} FPR {result['at_050']['fpr_pair']} | "
            f"лучший token F1 {result['grid_best']['token_f1']} при {result['grid_best']['threshold']}",
            flush=True,
        )
        if index == 1 and result["at_050"]["span_f1"] < BASE_SPAN_F1:
            stop = "span F1 после 1-й эпохи ниже базы"
            break
    final = report["epochs_log"][-1]["val150"]["grid_best"]["token_f1"]
    best_epoch = max(report["epochs_log"], key=lambda item: item["val150"]["grid_best"]["token_f1"])
    report["stop"] = stop
    report["best_epoch"] = best_epoch["epoch"]
    report["best_checkpoint"] = best_epoch["checkpoint"]
    report["best_token_f1"] = best_epoch["val150"]["grid_best"]["token_f1"]
    report["base_token_f1"] = BASE_TOKEN_F1
    report["improved"] = bool(report["best_token_f1"] > BASE_TOKEN_F1)
    report["rollback"] = not report["improved"]
    print(
        f"итог: лучшая эпоха {report['best_epoch']} token F1 {report['best_token_f1']} против базы {BASE_TOKEN_F1} | "
        f"{'улучшение' if report['improved'] else 'откат на базу'} {stop}",
        flush=True,
    )
    if not report["improved"]:
        save_checkpoint(model, tokenizer, Path(args.out) / "rolled_back")
    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
