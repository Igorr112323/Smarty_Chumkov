import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from spanver.sentence_snap import sentence_intervals
from spanver.stage_train import ENCODER_RATE, HEAD_RATE
from tools.export_predictions40 import word_probabilities
from tools.nested38 import split, target_inner
from tools.nested_cv35 import GATE, predict, probabilities
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.train_base33 import BATCH, MICRO, grouped_folds
from tools.train_scheme13_full import loss_batch, save_json

EPOCHS = 15
PATIENCE = 3
DECAY = 0.999
BUDGET_SECONDS = 150 * 60
EXPECTED_GOLD_SHA = "0ba55b91943c149a61ffe9db35a91db2a32deda573d72ba8fb6575f51b027f63"


def build_optimizer(model, total_steps):
    embeddings = []
    encoder = []
    head = []
    for name, parameter in model.named_parameters():
        if name.startswith("base.bert.embeddings."):
            embeddings.append(parameter)
        elif name.startswith("proj") or "classifier" in name:
            head.append(parameter)
        else:
            encoder.append(parameter)
    if not embeddings or not encoder or not head:
        raise ValueError("не найдены три группы параметров")
    optimizer = torch.optim.AdamW([{"params": embeddings, "lr": ENCODER_RATE * 0.1}, {"params": encoder, "lr": ENCODER_RATE}, {"params": head, "lr": HEAD_RATE}], weight_decay=0.01)
    warmup = max(1, int(total_steps * 0.1))
    def rate(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, rate), embeddings, warmup


def set_embedding_epoch(embeddings, epoch):
    for parameter in embeddings:
        parameter.requires_grad_(epoch > 5)


def clone_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def update_ema(ema, model, decay=DECAY):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                ema[name].copy_(value)


def validation_loss(model, table, tokenizer):
    model.eval()
    total = 0.0
    count = 0
    keys = list(table)
    with torch.no_grad():
        for start in range(0, len(keys), MICRO):
            batch = keys[start:start + MICRO]
            piece = lexfeat.collate([table[key] for key in batch], tokenizer.pad_token_id)
            token, pair = loss_batch(torch, model, piece)
            total += float(token + 0.3 * pair) * len(batch)
            count += len(batch)
    if not count:
        raise ValueError("пустой внутренний фолд")
    return total / count


def train(model, tokenizer, training, inner, seed):
    train_table = lexfeat.build_table(tokenizer, training)
    val_table = lexfeat.build_table(tokenizer, inner)
    if lexfeat.table_pairs(train_table) != len(training) or lexfeat.table_pairs(val_table) != len(inner):
        raise ValueError("не все пары закодированы")
    optimizer, scheduler, embeddings, warmup = build_optimizer(model, math.ceil(len(training) / BATCH) * EPOCHS)
    ema = clone_state(model)
    best = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        set_embedding_epoch(embeddings, epoch)
        keys = list(train_table)
        random.Random(seed * 1000 + epoch).shuffle(keys)
        model.train()
        total = count = 0
        for start in range(0, len(keys), BATCH):
            batch = keys[start:start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(batch), MICRO):
                part = batch[offset:offset + MICRO]
                piece = lexfeat.collate([train_table[key] for key in part], tokenizer.pad_token_id)
                token, pair = loss_batch(torch, model, piece)
                loss = token + 0.3 * pair
                (loss * len(part) / len(batch)).backward()
                total += float(loss.detach()) * len(part)
                count += len(part)
            optimizer.step()
            scheduler.step()
            update_ema(ema, model)
        live = clone_state(model)
        model.load_state_dict(ema, strict=True)
        current = validation_loss(model, val_table, tokenizer)
        if not math.isfinite(current):
            raise ValueError("нечисловая внутренняя ошибка")
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            best = clone_state(model)
            stale = 0
        else:
            stale += 1
        model.load_state_dict(live, strict=True)
        del live
        history.append({"epoch": epoch, "train_loss": total / count, "inner_ema_loss": current, "best_epoch": best_epoch, "stale": stale, "embeddings_frozen": epoch <= 5, "embedding_lr": scheduler.get_last_lr()[0], "seconds": round(time.perf_counter() - started, 3)})
        print(f"seed {seed} эпоха {epoch}: inner loss {current:.6f}, best {best_epoch}, ожидание {stale}, секунд {history[-1]['seconds']}", flush=True)
        if epoch >= 6 and stale >= PATIENCE:
            break
    if best is None:
        raise ValueError("не получены EMA веса")
    model.load_state_dict(best, strict=True)
    model.eval()
    return {"history": history, "epochs_ran": len(history), "best_epoch": best_epoch, "best_inner_ema_loss": best_loss, "warmup_steps": warmup, "minimum_stop_epoch": 6}


def input_without_gold(row):
    return {key: value for key, value in row.items() if key not in ("spans", "label")}


def inner_rows(pairs, cache, outer, inner, seed):
    result = []
    for row in pairs:
        value = cache[row["pair_id"]]
        result.append({"pair_id": row["pair_id"], "doc_ids": row["doc_ids"], "fold": outer, "inner_fold": inner, "seed": seed, "answer": value["answer"], "pair_prob": max((piece[2] for piece in value["pieces"]), default=0.0), "pair_prob_definition": "max_piece_probability_gate_score_not_separate_pair_head", "pieces": value["pieces"], "word_probabilities": word_probabilities(value["answer"], value["pieces"]), "sentence_intervals": sentence_intervals(value["answer"]), "gold_spans": row["spans"], "predicted_spans": predict(value, 0.5), "prediction_status": "temporary_0.5_before_inner_selection"})
    return result


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--inner", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    source = ROOT / "artifacts/ours33.jsonl"
    data_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    pairs = read_jsonl(source)
    if data_sha != EXPECTED_GOLD_SHA or len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS or len({row["pair_id"] for row in pairs}) != 699:
        raise ValueError("неверные 699 пар train/val")
    folds = grouped_folds(pairs, 5)
    outer, inner = args.outer - 1, args.inner - 1
    training, tuning, held = split(folds, outer, inner)
    primary = inner == target_inner(outer)
    blinded_held = [input_without_gold(row) for row in held] if primary else []
    target = ROOT / f"artifacts/train44_fold{args.outer}_inner{args.inner}.json"
    report = {"status": "в процессе", "outer_fold": args.outer, "inner_fold": args.inner, "primary": primary, "git_sha": args.git_sha, "gold_sha256": data_sha, "fold_ids_sha256": [id_digest(fold) for fold in folds], "train_id_sha256": id_digest(training), "inner_id_sha256": id_digest(tuning), "outer_id_sha256": id_digest(held), "test301": False, "external_data": False, "config": {"max_epochs": EPOCHS, "patience": PATIENCE, "minimum_stop_epoch": 6, "ema_decay": DECAY, "embeddings_frozen_epochs": 5, "embedding_lr_multiplier": 0.1, "warmup_fraction": 0.1, "focal_gamma": 2, "pair_loss_weight": 0.3, "gate": GATE}, "seeds": []}
    target.parent.mkdir(parents=True, exist_ok=True)
    save_json(target, report)
    started = time.perf_counter()
    for seed in range(5):
        seed_started = time.perf_counter()
        torch.manual_seed(seed)
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        info = train(model, tokenizer, training, tuning, seed)
        inner_cache = probabilities(tokenizer, model, tuning)
        inner_path = ROOT / f"artifacts/inner_predictions_fold{args.outer}_inner{args.inner}_seed{seed}.jsonl"
        write_rows(inner_path, inner_rows(tuning, inner_cache, args.outer, args.inner, seed))
        if primary:
            raw = probabilities(tokenizer, model, blinded_held)
            out_rows = [{"pair_id": row["pair_id"], "doc_ids": row["doc_ids"], "fold": args.outer, "seed": seed, "answer": raw[row["pair_id"]]["answer"], "pieces": raw[row["pair_id"]]["pieces"]} for row in blinded_held]
            write_rows(ROOT / f"artifacts/outer44_raw_fold{args.outer}_seed{seed}.jsonl", out_rows)
        info.update({"seed": seed, "seconds": round(time.perf_counter() - seed_started, 3), "inner_file": str(inner_path.relative_to(ROOT)), "inner_initial_sha256": hashlib.sha256(inner_path.read_bytes()).hexdigest()})
        report["seeds"].append(info)
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(target, report)
        del model, tokenizer, inner_cache
        elapsed = time.perf_counter() - started
        if seed >= 1 and seed < 4 and elapsed / (seed + 1) * 5 > BUDGET_SECONDS:
            report["budget_fallback"] = "прогноз 5 сидов превышает 150 минут; для итогов допускаются только seed 0 и 1"
            break
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    save_json(target, report)
    print(f"fold {args.outer}/{args.inner} готов; seeds {len(report['seeds'])}; секунд {report['elapsed_s']}", flush=True)


if __name__ == "__main__":
    main()
