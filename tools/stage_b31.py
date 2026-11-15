import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, lexfeat
from spanver.evaluate import metrics
from spanver.snap import snapped
from spanver.stage_train import ENCODER_RATE, HEAD_RATE
from tools.preflight_13 import components, digest, id_digest, our_keys, our_rows, split_groups
from tools.sweep_hyst import EXTEND_GRID, START_GRID, flags_with
from tools.sweep_ragtruth30 import compute, prepare, within_fpr_limits
from tools.train_bce29 import weighted_loss
from tools.train_cv_feat import token_fpr
from tools.train_scheme13_full import load_state, loss_batch, save_json

SEED = 0
BATCH = 16
MICRO = 4
EPOCHS = 3
GATE = 0.45
METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")
CHECKPOINTS = {"bce": "artifacts/checkpoints/stage_a_bce_w11_e1", "focal": "artifacts/checkpoints/stage_a_full_e2"}


def folds_for(pairs, count=5):
    if len(pairs) != 699 or len({row["pair_id"] for row in pairs}) != 699:
        raise ValueError("ожидалось 699 разных наших пар")
    buckets = split_groups(components(pairs, our_keys), count, SEED)
    for index, held in enumerate(buckets):
        train = [row for number, bucket in enumerate(buckets) if number != index for row in bucket]
        if {key for row in held for key in our_keys(row)} & {key for row in train for key in our_keys(row)}:
            raise ValueError("документы пересекаются между фолдами")
    return buckets


def tune(pairs, cache):
    prepared = prepare(pairs, cache)
    rows = [compute(prepared, start, extend, GATE) for start in START_GRID for extend in EXTEND_GRID if extend < start]
    allowed = [row for row in rows if within_fpr_limits(row)]
    best = max(allowed or rows, key=lambda row: (row["token"]["f1"], -row["start"], -row["extend"]))
    start, extend = best["start"], best["extend"]
    def predict(pair):
        value = cache[pair["pair_id"]]
        pieces = value["pieces"]
        if max((piece[2] for piece in pieces), default=0.0) < GATE:
            return []
        flags = flags_with(pieces, start, extend, 0.0)
        return snapped(decode.spans_from_flags(value["answer"], pieces, flags, 2), value["answer"], "expand")
    exact = metrics(pairs, predict)
    for fast, original in (("token", "token_level"), ("span", "span_level"), ("pair", "pair_level")):
        if any(best[fast][key] != exact[original][key] for key in ("precision", "recall", "f1")):
            raise ValueError("метрики быстрого перебора расходятся с evaluator")
    if best["fpr_pair"] != exact["fpr"] or best["fpr_token"] != token_fpr(pairs, predict):
        raise ValueError("FPR расходится с evaluator")
    measured = {f"{level}_{key}": exact[f"{level}_level"][key] for level in ("token", "span") for key in ("precision", "recall", "f1")}
    measured.update({"pair_f1": exact["pair_level"]["f1"], "fpr_pair": exact["fpr"], "fpr_token": best["fpr_token"], "type_accuracy": exact["type_accuracy"], "type_correct": exact["type_correct"], "matched_spans": exact["span_level"]["tp"]})
    return {"start": start, "extend": extend, "eligible": bool(allowed), "candidates": len(rows), "eligible_candidates": len(allowed), "measurements": measured, "grid": rows}


def train_fold(torch, model, tokenizer, training, model_name, fold):
    table = lexfeat.build_table(tokenizer, training)
    if len(table) != len(training):
        raise ValueError("не все пары закодированы")
    head = [p for name, p in model.named_parameters() if p.requires_grad and (name.startswith("proj") or "classifier" in name)]
    encoder = [p for name, p in model.named_parameters() if p.requires_grad and not (name.startswith("proj") or "classifier" in name)]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": ENCODER_RATE / 5}, {"params": head, "lr": HEAD_RATE / 5}], weight_decay=0.01)
    steps_epoch = math.ceil(len(training) / BATCH)
    steps = steps_epoch * EPOCHS
    warmup = max(1, int(steps * 0.1))
    def factor(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (steps - step) / max(1, steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    history = []
    for epoch in range(1, EPOCHS + 1):
        ids = list(table)
        random.Random(SEED + 1000 * fold + epoch).shuffle(ids)
        model.train()
        total = 0.0
        examples = 0
        clock = time.perf_counter()
        for start in range(0, len(ids), BATCH):
            group = ids[start:start + BATCH]
            optimizer.zero_grad()
            for offset in range(0, len(group), MICRO):
                part = group[offset:offset + MICRO]
                piece = lexfeat.collate([table[key] for key in part], tokenizer.pad_token_id)
                if model_name == "bce":
                    logits = model(input_ids=piece["input_ids"], attention_mask=piece["attention_mask"], features=piece["features"])
                    loss, _ = weighted_loss(torch, logits, piece["labels"])
                else:
                    token, pair = loss_batch(torch, model, piece)
                    loss = token + 0.3 * pair
                (loss * (len(part) / len(group))).backward()
                total += float(loss.detach()) * len(part)
                examples += len(part)
            optimizer.step()
            scheduler.step()
        history.append({"epoch": epoch, "loss_per_example": total / examples, "examples": examples, "steps": steps_epoch, "seconds": round(time.perf_counter() - clock, 3)})
        print(f"{model_name} фолд {fold}, эпоха {epoch}/{EPOCHS}: {examples} пар, {history[-1]['seconds']} с", flush=True)
    model.eval()
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--folds", type=int, choices=(3, 5), default=5)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    pairs = our_rows()
    buckets = folds_for(pairs, args.folds)
    folder = ROOT / CHECKPOINTS[args.model]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    report_path = ROOT / f"artifacts/stage_b31_{args.model}.json"
    identity = {"model": args.model, "folds": args.folds, "pair_ids_sha256": id_digest(pairs), "fold_ids_sha256": [id_digest(bucket) for bucket in buckets], "checkpoint_parts_sha256": [item["sha256"] for item in manifest["training_state_parts"]], "code_sha256": digest(Path(__file__))}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["identity"] != identity:
            raise ValueError("существующий отчёт относится к другому прогону")
    else:
        report = {"status": "в процессе", "identity": identity, "config": {"epochs": EPOCHS, "encoder_lr": ENCODER_RATE / 5, "head_lr": HEAD_RATE / 5, "batch": BATCH, "micro_batch": MICRO, "loss": "weighted BCE pos_weight=11 без pair loss" if args.model == "bce" else "focal gamma=2 + 0.3 pair loss", "optimizer": "fresh AdamW, linear warmup/decay", "feature_count": 5, "pair_gate": GATE, "hysteresis_start": START_GRID, "hysteresis_extend": EXTEND_GRID, "selection": "max token F1 with both FPR <= 0.4, else max token F1; tuned and evaluated on same held fold", "snap": "expand", "iou": 0.5}, "rows": []}
        save_json(report_path, report)
    for index, held in enumerate(buckets):
        if index < len(report["rows"]):
            continue
        clock = time.perf_counter()
        training = [row for number, bucket in enumerate(buckets) if number != index for row in bucket]
        torch.manual_seed(SEED + index)
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
        model.base.config._attn_implementation = "eager"
        state = load_state(folder, manifest["training_state_parts"], torch)
        model.load_state_dict(state["model"], strict=True)
        del state
        print(f"{args.model} фолд {index + 1}/{args.folds}: train {len(training)}, holdout {len(held)}", flush=True)
        history = train_fold(torch, model, tokenizer, training, args.model, index + 1)
        cache = {}
        for start in range(0, len(held), 16):
            cache.update(lexfeat.batched_probabilities(tokenizer, model, held[start:start + 16], top=4, batch=4))
        if len(cache) != len(held):
            raise ValueError("часть фолда не оценена")
        result = tune(held, cache)
        report["rows"].append({"fold": index + 1, "train_pairs": len(training), "held_pairs": len(held), "train_id_sha256": id_digest(training), "held_id_sha256": id_digest(held), "history": history, "tuning": result, "seconds": round(time.perf_counter() - clock, 3)})
        save_json(report_path, report)
        print(f"{args.model} фолд {index + 1} готов: {json.dumps(result['measurements'], ensure_ascii=False)}", flush=True)
        del model, tokenizer, cache
    report["summary"] = {}
    for name in METRICS:
        values = [row["tuning"]["measurements"][name] for row in report["rows"]]
        report["summary"][name] = {"mean": round(statistics.mean(values), 4), "std": round(statistics.pstdev(values), 4)}
    report["status"] = "готово"
    save_json(report_path, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
