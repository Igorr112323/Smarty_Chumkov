import json
import math
import random
import statistics
import time
from pathlib import Path

from spanver import decode, lexfeat
from spanver.evaluate import metrics, read_jsonl
from spanver.snap import snapped
from spanver.stage_train import ENCODER_RATE, HEAD_RATE
from tools.preflight_13 import components, id_digest, our_keys, split_groups
from tools.prepare_ours33 import EXPECTED_IDS
from tools.sweep_hyst import flags_with
from tools.sweep_ragtruth30 import compute, prepare, within_fpr_limits
from tools.train_cv_feat import token_fpr
from tools.train_scheme13_full import loss_batch, save_json

GRID = tuple(round(index * 0.05, 2) for index in range(2, 20))
BATCH = 16
MICRO = 4
GATE = 0.45
METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")


def grouped_folds(pairs, count):
    buckets = split_groups(components(pairs, our_keys), count, 0)
    for index, held in enumerate(buckets):
        rest = [row for j, bucket in enumerate(buckets) if j != index for row in bucket]
        if {key for row in held for key in our_keys(row)} & {key for row in rest for key in our_keys(row)}:
            raise ValueError("документ попал в обучение и оценку")
    return buckets


def tune(pairs, cache):
    prepared = prepare(pairs, cache)
    rows = [compute(prepared, start, extend, GATE) for start in GRID for extend in GRID if extend <= start]
    allowed = [row for row in rows if within_fpr_limits(row)]
    chosen = max(allowed or rows, key=lambda row: (row["token"]["f1"], -row["start"], -row["extend"]))
    def predict(pair):
        item = cache[pair["pair_id"]]
        pieces = item["pieces"]
        if max((score for _, _, score in pieces), default=0.0) < GATE:
            return []
        flags = flags_with(pieces, chosen["start"], chosen["extend"], 0.0)
        return snapped(decode.spans_from_flags(item["answer"], pieces, flags, 2), item["answer"], "expand")
    reference = metrics(pairs, predict)
    for level in ("token", "span", "pair"):
        for metric in ("precision", "recall", "f1"):
            if reference[f"{level}_level"][metric] != chosen[level][metric]:
                raise ValueError("несовпадение с evaluator")
    if reference["fpr"] != chosen["fpr_pair"] or token_fpr(pairs, predict) != chosen["fpr_token"]:
        raise ValueError("несовпадение FPR с evaluator")
    result = {f"{level}_{metric}": reference[f"{level}_level"][metric] for level in ("token", "span") for metric in ("precision", "recall", "f1")}
    result.update({"pair_f1": reference["pair_level"]["f1"], "fpr_pair": reference["fpr"], "fpr_token": chosen["fpr_token"], "type_accuracy": reference["type_accuracy"], "type_correct": reference["type_correct"], "matched_spans": reference["span_level"]["tp"]})
    return {"start": chosen["start"], "extend": chosen["extend"], "eligible": bool(allowed), "eligible_candidates": len(allowed), "candidates": len(rows), "metrics": result}


class BudgetExceeded(RuntimeError):
    pass


def train(torch, model, tokenizer, pairs, epochs, seed, probe_folds=0, budget_seconds=0):
    table = lexfeat.build_table(tokenizer, pairs)
    if len(table) != len(pairs):
        raise ValueError("не все наши пары закодированы")
    head = [p for name, p in model.named_parameters() if p.requires_grad and (name.startswith("proj") or "classifier" in name)]
    encoder = [p for name, p in model.named_parameters() if p.requires_grad and not (name.startswith("proj") or "classifier" in name)]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": ENCODER_RATE}, {"params": head, "lr": HEAD_RATE}], weight_decay=0.01)
    total_steps = math.ceil(len(pairs) / BATCH) * epochs
    warmup = max(1, int(total_steps * 0.1))
    def rate(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    history = []
    for epoch in range(1, epochs + 1):
        ids = list(table)
        random.Random(seed + epoch).shuffle(ids)
        started = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for start in range(0, len(ids), BATCH):
            group = ids[start:start + BATCH]
            optimizer.zero_grad()
            for position in range(0, len(group), MICRO):
                subset = group[position:position + MICRO]
                piece = lexfeat.collate([table[key] for key in subset], tokenizer.pad_token_id)
                token, pair = loss_batch(torch, model, piece)
                loss = token + 0.3 * pair
                (loss * len(subset) / len(group)).backward()
                total += float(loss.detach()) * len(subset)
                count += len(subset)
            optimizer.step()
            scheduler.step()
        history.append({"epoch": epoch, "pairs": count, "loss": total / count, "seconds": round(time.perf_counter() - started, 3)})
        print(f"эпоха {epoch}/{epochs}: {count} пар, {history[-1]['seconds']} с", flush=True)
        if probe_folds and epoch == 1:
            estimate = history[-1]["seconds"] * epochs * probe_folds + 600 * probe_folds
            print(f"предварительная оценка CV: {estimate:.1f} с, бюджет {budget_seconds:.1f} с", flush=True)
            if estimate > budget_seconds:
                raise BudgetExceeded(f"прогноз трёх фолдов {estimate:.1f} с превышает бюджет {budget_seconds:.1f} с")
    model.eval()
    return history


def run(args):
    import torch

    if args.loss != "focal2.0" or args.class_weight != 1.0 or args.folds not in (2, 3, 5) or args.epochs != 3:
        raise ValueError("для схемы 33 нужны focal2.0, class_weight=1, 3 эпохи и 2, 3 или 5 фолдов")
    pairs = read_jsonl(Path(args.gold))
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS:
        raise ValueError("ожидались только 699 наших train/val пар")
    buckets = grouped_folds(pairs, args.folds)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "в процессе", "pairs": len(pairs), "folds": args.folds, "epochs": args.epochs, "fold_sizes": [len(bucket) for bucket in buckets], "fold_ids_sha256": [id_digest(bucket) for bucket in buckets], "loss": "focal gamma=2 + 0.3 pair", "external_data": False, "weights": args.weights, "features": 5, "selection": "max token F1 on held fold subject to FPR_pair and FPR_token <= 0.40; fallback unconstrained", "selection_bias": "thresholds tuned and evaluated on the same held fold", "grid": list(GRID), "pair_gate": GATE, "rows": []}
    save_json(path, report)
    started = time.perf_counter()
    for index, held in enumerate(buckets):
        clock = time.perf_counter()
        training = [row for number, bucket in enumerate(buckets) if number != index for row in bucket]
        tokenizer, model = lexfeat.load_feature_model(args.weights)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(index)
        print(f"фолд {index + 1}/{args.folds}: train {len(training)}, holdout {len(held)}", flush=True)
        try:
            history = train(torch, model, tokenizer, training, args.epochs, index * 1000, args.folds if args.folds == 3 and index == 0 else 0, args.budget_minutes * 60)
        except BudgetExceeded as exc:
            report["status"] = "пилот остановлен: прогноз превышает бюджет; нужен двухфолдовый CV"
            report["reason"] = str(exc)
            report["elapsed_s"] = round(time.perf_counter() - started, 3)
            save_json(path, report)
            print(str(exc), flush=True)
            return 75
        cache = {}
        for start in range(0, len(held), 16):
            cache.update(lexfeat.batched_probabilities(tokenizer, model, held[start:start + 16], top=4, batch=4))
        if len(cache) != len(held):
            raise ValueError("не все пары получили предсказание")
        tuned = tune(held, cache)
        checkpoint = lexfeat.save_feature_model(model, tokenizer, output / f"fold_{index + 1}", args.weights)
        row = {"fold": index + 1, "train_pairs": len(training), "held_pairs": len(held), "history": history, "tuning": tuned, "checkpoint": str(checkpoint), "seconds": round(time.perf_counter() - clock, 3)}
        report["rows"].append(row)
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(path, report)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del model, tokenizer, cache
    report["summary"] = {metric: {"mean": round(statistics.mean([row["tuning"]["metrics"][metric] for row in report["rows"]]), 4), "std": round(statistics.pstdev([row["tuning"]["metrics"][metric] for row in report["rows"]]), 4)} for metric in METRICS}
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    save_json(path, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0
