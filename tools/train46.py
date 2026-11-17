import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import transformers

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from spanver.stage_train import ENCODER_RATE, HEAD_RATE
from tools.nested38 import GRID, PART_BYTES, file_digest, split, target_inner
from tools.nested_cv35 import GATE, averaged, probabilities
from tools.predict_frozen38 import load_state
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.sweep_ragtruth30 import compute, prepare
from tools.train_base33 import BATCH, MICRO, grouped_folds, train as control_train
from tools.train_scheme13_full import loss_batch, save_json

EPOCHS = 15
PATIENCE = 3
DECAY = 0.999
SEEDS = (0, 1)


def groups(model, freeze):
    embeddings, encoder, head = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith("base.bert.embeddings."):
            embeddings.append(parameter)
        elif name.startswith("proj") or "classifier" in name:
            head.append(parameter)
        else:
            encoder.append(parameter)
    if not embeddings or not encoder or not head:
        raise ValueError("не найдены группы параметров")
    if freeze:
        return [{"params": embeddings, "lr": ENCODER_RATE * 0.1}, {"params": encoder, "lr": ENCODER_RATE}, {"params": head, "lr": HEAD_RATE}], embeddings
    return [{"params": embeddings + encoder, "lr": ENCODER_RATE}, {"params": head, "lr": HEAD_RATE}], embeddings


def validation_loss(model, table, tokenizer):
    model.eval()
    total = count = 0
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


def clone_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def update_ema(ema, model):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema[name].mul_(DECAY).add_(value.detach(), alpha=1 - DECAY)
            else:
                ema[name].copy_(value)


def train_variant(model, tokenizer, training, inner, seed, variant):
    training_table = lexfeat.build_table(tokenizer, training)
    validation_table = lexfeat.build_table(tokenizer, inner)
    if len(training_table) != len(training) or len(validation_table) != len(inner):
        raise ValueError("не все пары закодированы")
    frozen = variant in ("v2", "v3")
    ema_enabled = variant == "v3"
    params, embeddings = groups(model, frozen)
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    total_steps = math.ceil(len(training) / BATCH) * EPOCHS
    warmup = max(1, int(total_steps * 0.1))
    def rate(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    ema = clone_state(model) if ema_enabled else None
    best_loss = float("inf")
    best_epoch = stale = 0
    history = []
    with tempfile.TemporaryDirectory(prefix="best46-") as directory:
        best_path = Path(directory) / "best.pt"
        for epoch in range(1, EPOCHS + 1):
            clock = time.perf_counter()
            for parameter in embeddings:
                parameter.requires_grad_(not frozen or epoch > 5)
            keys = list(training_table)
            random.Random(seed * 1000 + epoch).shuffle(keys)
            model.train()
            total = count = 0
            for start in range(0, len(keys), BATCH):
                batch = keys[start:start + BATCH]
                optimizer.zero_grad()
                for offset in range(0, len(batch), MICRO):
                    part = batch[offset:offset + MICRO]
                    piece = lexfeat.collate([training_table[key] for key in part], tokenizer.pad_token_id)
                    token, pair = loss_batch(torch, model, piece)
                    loss = token + 0.3 * pair
                    (loss * len(part) / len(batch)).backward()
                    total += float(loss.detach()) * len(part)
                    count += len(part)
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    update_ema(ema, model)
            if ema is not None:
                live = clone_state(model)
                model.load_state_dict(ema, strict=True)
            current = validation_loss(model, validation_table, tokenizer)
            if not math.isfinite(current):
                raise ValueError("нечисловая внутренняя ошибка")
            if current < best_loss:
                best_loss = current
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                stale = 0
            else:
                stale += 1
            if ema is not None:
                model.load_state_dict(live, strict=True)
                del live
            history.append({"epoch": epoch, "train_loss": total / count, "inner_loss": current, "best_epoch": best_epoch, "stale": stale, "embeddings_frozen": frozen and epoch <= 5, "seconds": round(time.perf_counter() - clock, 3)})
            print(f"{variant} seed {seed}, эпоха {epoch}: loss {current:.6f}, best {best_epoch}, stale {stale}", flush=True)
            if stale >= PATIENCE:
                break
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return {"history": history, "epochs_ran": len(history), "best_epoch": best_epoch, "best_inner_loss": best_loss, "warmup_steps": warmup}


def save_checkpoint(model, variant, outer, seed):
    folder = ROOT / "weights/ablation46" / variant / f"fold{outer}" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weights46-") as directory:
        raw = Path(directory) / "checkpoint.pt"
        torch.save(model.state_dict(), raw)
        entry = {"seed": seed, "filename": "checkpoint.pt", "size": raw.stat().st_size, "sha256": file_digest(raw), "chunk_bytes": PART_BYTES, "chunks": []}
        with raw.open("rb") as handle:
            index = 0
            while True:
                data = handle.read(PART_BYTES)
                if not data:
                    break
                name = f"part-{index:03d}.bin"
                (folder / name).write_bytes(data)
                entry["chunks"].append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                index += 1
    save_json(folder / "manifest.json", entry)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("control", "v1", "v2", "v3"))
    parser.add_argument("--outer", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--inner", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    source = ROOT / "artifacts/ours33.jsonl"
    summary = json.loads((ROOT / "artifacts/nested38_summary.json").read_text(encoding="utf-8"))
    pairs = read_jsonl(source)
    if file_digest(source) != summary["gold_train_val_sha256"] or len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS:
        raise ValueError("неверный набор train/val")
    folds = grouped_folds(pairs, 5)
    if [id_digest(fold) for fold in folds] != summary["fold_ids_sha256"]:
        raise ValueError("изменено разбиение")
    outer, inner = args.outer - 1, args.inner - 1
    training, tuning, held = split(folds, outer, inner)
    primary = inner == target_inner(outer)
    blind = [{key: value for key, value in row.items() if key not in ("spans", "label")} for row in held]
    path = ROOT / f"artifacts/ablation46_{args.variant}_outer{args.outer}_inner{args.inner}.json"
    report = {"status": "в процессе", "variant": args.variant, "outer_fold": args.outer, "inner_fold": args.inner, "primary": primary, "git_sha": args.git_sha, "torch": str(torch.__version__), "transformers": transformers.__version__, "gold_sha256": file_digest(source), "fold_ids_sha256": summary["fold_ids_sha256"], "train_id_sha256": id_digest(training), "inner_id_sha256": id_digest(tuning), "outer_id_sha256": id_digest(held), "seeds": [], "test301": False, "external_data": False}
    save_json(path, report)
    inner_caches, outer_caches = [], []
    started = time.perf_counter()
    for seed in SEEDS:
        clock = time.perf_counter()
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(seed)
        if args.variant == "control":
            history = control_train(torch, model, tokenizer, training, 3, seed * 1000)
            info = {"history": history, "epochs_ran": len(history), "best_epoch": 3}
        else:
            info = train_variant(model, tokenizer, training, tuning, seed, args.variant)
        inner_caches.append(probabilities(tokenizer, model, tuning))
        checkpoint = None
        if primary:
            original = probabilities(tokenizer, model, blind)
            checkpoint = save_checkpoint(model, args.variant, args.outer, seed)
            del model, tokenizer
            tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
            model.base.config._attn_implementation = "eager"
            load_state(torch, model, ROOT / "weights/ablation46" / args.variant / f"fold{args.outer}" / f"seed_{seed}", checkpoint)
            restored = probabilities(tokenizer, model, blind)
            if restored != original:
                raise ValueError("вероятности после восстановления не совпали")
            checkpoint["restore_predictions_equal"] = True
            outer_caches.append(restored)
        info.update({"seed": seed, "seconds": round(time.perf_counter() - clock, 3), "checkpoint": checkpoint})
        report["seeds"].append(info)
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(path, report)
        print(f"{args.variant} outer {args.outer} inner {args.inner} seed {seed}: готов", flush=True)
        del model, tokenizer
    inner_cache = {row["pair_id"]: averaged(row, inner_caches, range(len(SEEDS))) for row in tuning}
    prepared = prepare(tuning, inner_cache)
    report["inner_grid"] = [compute(prepared, threshold, threshold / 2, GATE) for threshold in GRID]
    if primary:
        outer_cache = {row["pair_id"]: averaged(row, outer_caches, range(len(SEEDS))) for row in blind}
        report["outer_predictions"] = {key: value["pieces"] for key, value in outer_cache.items()}
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    save_json(path, report)
    print(f"{args.variant} outer {args.outer} inner {args.inner}: завершено", flush=True)


if __name__ == "__main__":
    main()
