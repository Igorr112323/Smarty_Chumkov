import gzip
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.preflight_13 import digest, id_digest
from tools.train_scheme13_a1 import evaluate, read_pairs
from tools.train_scheme13_full import save_json, save_state

SEED = 0
POS_WEIGHT = 11.0
BATCH_SIZE = 16
MICRO_BATCH = 4


def weighted_loss(torch, logits, labels):
    valid = labels != -100
    if not bool(valid.any()):
        raise ValueError("нет размеченных токенов")
    scores = (logits[:, :, 1] - logits[:, :, 0])[valid]
    targets = labels[valid].to(torch.float)
    weight = torch.tensor([POS_WEIGHT], dtype=scores.dtype, device=scores.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, targets, pos_weight=weight), int(valid.sum())


def control(torch, tokenizer, model, pairs):
    from spanver import lexfeat

    rows = [lexfeat.build_row(tokenizer, pair) for pair in pairs]
    probability = {"near_zero": 1e-9, "p099": 0.99, "p05": 0.5}
    totals = {"model": 0.0, **{name: 0.0 for name in probability}}
    count = 0
    labels_count = {"positive": 0, "negative": 0}
    model.eval()
    with torch.no_grad():
        for index in range(0, len(rows), MICRO_BATCH):
            batch = lexfeat.collate(rows[index:index + MICRO_BATCH], tokenizer.pad_token_id)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], features=batch["features"])
            loss, valid_count = weighted_loss(torch, logits, batch["labels"])
            totals["model"] += float(loss) * valid_count
            labels_count["positive"] += int((batch["labels"] == 1).sum())
            labels_count["negative"] += int((batch["labels"] == 0).sum())
            for name, p in probability.items():
                constant = torch.zeros_like(logits)
                constant[:, :, 1] = math.log(p / (1.0 - p))
                value, _ = weighted_loss(torch, constant, batch["labels"])
                totals[name] += float(value) * valid_count
            count += valid_count
    losses = {name: value / count for name, value in totals.items()}
    return {"loss": losses, "labels": labels_count, "pos_weight": POS_WEIGHT, "strictly_below_all": all(losses["model"] < losses[name] for name in probability), "near_zero_probability": probability["near_zero"], "exact_zero_loss": "infinity при наличии положительных меток"}


def main():
    import torch
    from spanver import lexfeat
    from spanver.stage_train import ENCODER_RATE, HEAD_RATE

    started = time.perf_counter()
    audit = json.loads((ROOT / "artifacts/bce29_audit.json").read_text(encoding="utf-8"))
    before = json.loads((ROOT / "artifacts/bce29_control.json").read_text(encoding="utf-8"))
    training, heldout = read_pairs()
    for name, pairs in (("train", training), ("holdout", heldout)):
        row = audit[name]
        if row["pair_ids_sha256"] != id_digest(pairs) or row["answers_with_missing_subwords"] or row["invalid_spans"]:
            raise ValueError("проверка ответов или разбиения не пройдена")
    if audit["encoder_code_sha256"] != digest(ROOT / "spanver/span_detector.py") or not before["control"]["strictly_below_all"]:
        raise ValueError("проверка взвешенного loss или кода не пройдена")
    if len(training) != 6042 or len(heldout) != 1458:
        raise ValueError("неожиданный размер данных")
    target = ROOT / "artifacts/checkpoints/stage_a_bce_w11_e1"
    if target.exists():
        raise FileExistsError(target)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
    model.base.config._attn_implementation = "eager"
    torch.set_num_threads(2)
    head = [p for name, p in model.named_parameters() if p.requires_grad and (name.startswith("proj") or "classifier" in name)]
    encoder = [p for name, p in model.named_parameters() if p.requires_grad and not (name.startswith("proj") or "classifier" in name)]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": ENCODER_RATE}, {"params": head, "lr": HEAD_RATE}], weight_decay=0.01)
    steps = math.ceil(len(training) / BATCH_SIZE)
    warmup = max(1, int(steps * 0.1))
    def factor(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (steps - step) / max(1, steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    order = list(range(len(training)))
    random.Random(SEED + 1).shuffle(order)
    model.train()
    total_loss = total_tokens = 0.0
    epoch_start = time.perf_counter()
    for step, start in enumerate(range(0, len(order), BATCH_SIZE), 1):
        rows = [lexfeat.build_row(tokenizer, training[index]) for index in order[start:start + BATCH_SIZE]]
        if any(row is None for row in rows):
            raise ValueError("пара не закодирована")
        optimizer.zero_grad()
        parts = math.ceil(len(rows) / MICRO_BATCH)
        for offset in range(0, len(rows), MICRO_BATCH):
            batch = lexfeat.collate(rows[offset:offset + MICRO_BATCH], tokenizer.pad_token_id)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], features=batch["features"])
            loss, count = weighted_loss(torch, logits, batch["labels"])
            (loss / parts).backward()
            total_loss += float(loss.detach()) * count
            total_tokens += count
        optimizer.step()
        scheduler.step()
        if step % 32 == 0 or step == steps:
            print(f"шаг {step}/{steps}, пар {start + len(rows)}/{len(training)}, секунд {time.perf_counter() - epoch_start:.1f}", flush=True)
    epoch_seconds = time.perf_counter() - epoch_start
    model.eval()
    target.mkdir(parents=True, exist_ok=False)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(), "epoch": 1}
    names = save_state(target, state, torch)
    half = {key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu() for key, value in model.state_dict().items()}
    with (target / "state_dict_fp16.pt.gz").open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as compressed:
            torch.save({"state_dict": half}, compressed)
    del half
    entries = [{"name": name, "bytes": (target / name).stat().st_size, "sha256": digest(target / name)} for name in names + ["state_dict_fp16.pt.gz"]]
    manifest = {"epoch": 1, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "training_code_sha256": digest(Path(__file__)), "lexfeat_sha256": digest(ROOT / "spanver/lexfeat.py"), "encoder_sha256": digest(ROOT / "spanver/span_detector.py"), "data_sha256": {name: digest(ROOT / "Dataset/RAGTruth_ru" / name) for name in ("ragtruth_ru.jsonl", "ragtruth_ru_negative.jsonl")}, "base_weights_sha256": digest(ROOT / "models/rubert-tiny2/pytorch_model.bin"), "train_id_sha256": id_digest(training), "holdout_id_sha256": id_digest(heldout), "config": {"from_scratch": True, "loss": "BCEWithLogitsLoss", "pos_weight": POS_WEIGHT, "auxiliary_pair_loss": False, "encoder_rate": ENCODER_RATE, "head_rate": HEAD_RATE, "seed": SEED, "features": 5, "batch_size": BATCH_SIZE, "micro_batch": MICRO_BATCH, "epochs": 1, "scheduler": "linear warmup 10% and decay", "threads": 2, "torch": str(torch.__version__)}, "training_loss": total_loss / total_tokens, "epoch_seconds": round(epoch_seconds, 3), "train_pairs": len(training), "holdout_pairs": len(heldout), "training_state_parts": entries[:-1], "fp16": entries[-1]}
    save_json(target / "manifest.json", manifest)
    print(f"чекпоинт сохранён {target}, эпоха {epoch_seconds:.3f} с", flush=True)
    by_id = {pair["pair_id"]: pair for pair in training}
    batch_pairs = [by_id[pair_id] for pair_id in before["control"]["batch_pair_ids"]]
    result = {"status": "проверка обученной модели", "checkpoint": str(target.relative_to(ROOT)), "training_loss": manifest["training_loss"], "epoch_seconds": manifest["epoch_seconds"], "before_training": before["control"], "after_training": control(torch, tokenizer, model, batch_pairs)}
    report_path = ROOT / "artifacts/bce29_report.json"
    save_json(report_path, result)
    print("контроль после обучения", json.dumps(result["after_training"], ensure_ascii=False), flush=True)
    if not result["after_training"]["strictly_below_all"]:
        result["status"] = "STOP: взвешенный loss не ниже всех констант"
        result["total_seconds"] = round(time.perf_counter() - started, 3)
        save_json(report_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 2
    result["status"] = "оценка holdout"
    save_json(report_path, result)
    result["holdout"] = evaluate(tokenizer, model, heldout)
    result["status"] = "готово"
    result["total_seconds"] = round(time.perf_counter() - started, 3)
    save_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
