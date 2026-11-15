import argparse
import gzip
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.preflight_13 import digest, id_digest
from tools.train_scheme13_a1 import evaluate, read_pairs

SEED = 0
EPOCHS = 3
BATCH_SIZE = 16
MICRO_BATCH = 4
PART_BYTES = 48 * 1024 * 1024


class PartWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.current = None
        self.remaining = 0
        self.names = []

    def write(self, data):
        view = memoryview(data)
        length = len(view)
        while view:
            if not self.remaining:
                if self.current is not None:
                    self.current.close()
                name = f"training_state_{len(self.names):03d}.gzpart"
                self.names.append(name)
                self.current = (self.directory / name).open("wb")
                self.remaining = PART_BYTES
            size = min(len(view), self.remaining)
            self.current.write(view[:size])
            self.remaining -= size
            view = view[size:]
        return length

    def flush(self):
        if self.current is not None:
            self.current.flush()

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None


class PartReader:
    def __init__(self, directory, names):
        self.paths = [Path(directory) / name for name in names]
        self.index = 0
        self.current = None

    def read(self, size=-1):
        result = bytearray()
        while size < 0 or len(result) < size:
            if self.current is None:
                if self.index == len(self.paths):
                    break
                self.current = self.paths[self.index].open("rb")
                self.index += 1
            chunk = self.current.read(-1 if size < 0 else size - len(result))
            if chunk:
                result.extend(chunk)
            else:
                self.current.close()
                self.current = None
        return bytes(result)

    def close(self):
        if self.current:
            self.current.close()
            self.current = None


def save_state(directory, state, torch):
    writer = PartWriter(directory)
    try:
        with gzip.GzipFile(fileobj=writer, mode="wb", mtime=0, filename="", compresslevel=6) as compressed:
            torch.save(state, compressed)
    finally:
        writer.close()
    return writer.names


def load_state(directory, files, torch):
    for entry in files:
        path = Path(directory) / entry["name"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"контрольная сумма не совпадает: {path}")
    reader = PartReader(directory, [entry["name"] for entry in files])
    try:
        with gzip.GzipFile(fileobj=reader, mode="rb") as compressed:
            data = compressed.read()
    finally:
        reader.close()
    import io
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def loss_batch(torch, model, piece):
    from spanver.lexfeat import EDGE, IGNORE

    logits = model(input_ids=piece["input_ids"], attention_mask=piece["attention_mask"], features=piece["features"])
    labels = piece["labels"]
    flat = torch.nn.functional.cross_entropy(logits.view(-1, 2), labels.view(-1), ignore_index=IGNORE, label_smoothing=0.1, reduction="none")
    keep = (labels.view(-1) != IGNORE).to(flat.dtype)
    probabilities = torch.nn.functional.softmax(logits.view(-1, 2), dim=-1)
    chosen = probabilities.gather(1, labels.view(-1, 1).clamp(min=0)).squeeze(1)
    token_loss = (flat * (1.0 - chosen).clamp(min=0.0) ** 2 * keep).sum() / keep.sum().clamp(min=1.0)
    diff = logits[:, :, 1] - logits[:, :, 0]
    scores = diff.masked_fill(piece["answer_mask"] == 0, EDGE).max(dim=1).values
    pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, piece["targets"])
    return token_loss, pair_loss


def run_epoch(torch, model, tokenizer, training, optimizer, scheduler, epoch):
    from spanver import lexfeat

    order = list(range(len(training)))
    random.Random(SEED + epoch).shuffle(order)
    model.train()
    token_sum = pair_sum = total_weight = 0.0
    started = time.perf_counter()
    steps = math.ceil(len(order) / BATCH_SIZE)
    for step, start in enumerate(range(0, len(order), BATCH_SIZE), 1):
        ids = order[start:start + BATCH_SIZE]
        rows = [lexfeat.build_row(tokenizer, training[index]) for index in ids]
        if any(row is None for row in rows):
            raise ValueError("не удалось закодировать тренировочную пару")
        optimizer.zero_grad()
        for offset in range(0, len(rows), MICRO_BATCH):
            part = rows[offset:offset + MICRO_BATCH]
            piece = lexfeat.collate(part, tokenizer.pad_token_id)
            token_loss, pair_loss = loss_batch(torch, model, piece)
            ((token_loss + 0.3 * pair_loss) / 4).backward()
            token_sum += float(token_loss.detach()) * len(part)
            pair_sum += float(pair_loss.detach()) * len(part)
            total_weight += len(part)
        optimizer.step()
        scheduler.step()
        if step % 32 == 0 or step == steps:
            print(f"эпоха {epoch}: шаг {step}/{steps}, пар {start + len(ids)}/{len(training)}, секунд {time.perf_counter() - started:.1f}", flush=True)
    model.eval()
    return {"token_loss": token_sum / total_weight, "pair_loss": pair_sum / total_weight, "total_loss": (token_sum + 0.3 * pair_sum) / total_weight, "epoch_seconds": round(time.perf_counter() - started, 3), "batches": steps, "examples": int(total_weight)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    import torch
    from spanver import lexfeat
    from spanver.stage_train import ENCODER_RATE, HEAD_RATE

    started = time.perf_counter()
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    training, heldout = read_pairs()
    expected = json.loads((ROOT / "artifacts/preflight_13.json").read_text(encoding="utf-8"))
    if len(training) != 6042 or len(heldout) != 1458 or id_digest(training) != expected["rag_train"]["id_sha256"] or id_digest(heldout) != expected["rag_holdout"]["id_sha256"]:
        raise ValueError("разбиение расходится с проверенным preflight")
    if not args.evaluate_only and (args.epoch == 1) != (not args.resume):
        raise ValueError("эпоха 1 начинается с нуля; для остальных нужен чекпоинт предыдущей эпохи")
    if args.evaluate_only and args.resume:
        raise ValueError("при повторной оценке не нужен --resume")
    target = ROOT / f"artifacts/checkpoints/stage_a_full_e{args.epoch}"
    if not args.evaluate_only and target.exists():
        raise FileExistsError(target)
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
    model.base.config._attn_implementation = "eager"
    torch.set_num_threads(2)
    head = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("proj") or "classifier" in n)]
    encoder = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("proj") or "classifier" in n)]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": ENCODER_RATE}, {"params": head, "lr": HEAD_RATE}], weight_decay=0.01)
    steps = math.ceil(len(training) / BATCH_SIZE) * EPOCHS
    warmup = max(1, int(steps * 0.1))
    def factor(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (steps - step) / max(1, steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    if args.evaluate_only:
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if manifest["epoch"] != args.epoch or manifest["train_id_sha256"] != id_digest(training) or manifest["holdout_id_sha256"] != id_digest(heldout):
            raise ValueError("манифест не подходит для оценки")
        state = load_state(target, manifest["training_state_parts"], torch)
        model.load_state_dict(state["model"], strict=True)
        del state
        model.eval()
        report_path = ROOT / f"artifacts/scheme13_full_e{args.epoch}.json"
        report = {"status": "оценка holdout", "epoch": args.epoch, "checkpoint": str(target.relative_to(ROOT)), "loss": manifest["loss"]}
        save_json(report_path, report)
        report["holdout"] = evaluate(tokenizer, model, heldout)
        report["status"] = "готово"
        report["evaluation_restart_seconds"] = round(time.perf_counter() - started, 3)
        save_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    if args.resume:
        folder = ROOT / args.resume
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["epoch"] != args.epoch - 1 or manifest["train_id_sha256"] != id_digest(training) or manifest["holdout_id_sha256"] != id_digest(heldout) or manifest["training_code_sha256"] != digest(Path(__file__)):
            raise ValueError("манифест не подходит для продолжения")
        state = load_state(folder, manifest["training_state_parts"], torch)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        del state
    history = run_epoch(torch, model, tokenizer, training, optimizer, scheduler, args.epoch)
    target.mkdir(parents=True, exist_ok=False)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(), "epoch": args.epoch}
    names = save_state(target, state, torch)
    half = {key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu() for key, value in model.state_dict().items()}
    with (target / "state_dict_fp16.pt.gz").open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as compressed:
            torch.save({"state_dict": half}, compressed)
    del half
    paths = names + ["state_dict_fp16.pt.gz"]
    entries = [{"name": name, "bytes": (target / name).stat().st_size, "sha256": digest(target / name)} for name in paths]
    manifest = {
        "epoch": args.epoch, "seed": SEED, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "training_code_sha256": digest(Path(__file__)), "preflight_code_sha256": digest(ROOT / "tools/preflight_13.py"),
        "data_sha256": {name: digest(ROOT / "Dataset/RAGTruth_ru" / name) for name in ("ragtruth_ru.jsonl", "ragtruth_ru_negative.jsonl")},
        "base_weights_sha256": digest(ROOT / "models/rubert-tiny2/pytorch_model.bin"),
        "train_pairs": len(training), "train_positives": sum(bool(row["spans"]) for row in training), "train_negatives": sum(not row["spans"] for row in training),
        "train_id_sha256": id_digest(training), "holdout_pairs": len(heldout), "holdout_id_sha256": id_digest(heldout),
        "config": {"features": 5, "focal_gamma": 2.0, "label_smoothing": 0.1, "pair_weight": 0.3, "batch_size": BATCH_SIZE, "micro_batch": MICRO_BATCH, "encoder_rate": ENCODER_RATE, "head_rate": HEAD_RATE, "weight_decay": 0.01, "scheduler": "linear warmup 10% and decay across 3 epochs", "threads": 2, "torch": str(torch.__version__)},
        "loss": history, "training_state_parts": entries[:-1], "fp16": entries[-1],
        "resume": f"python tools/train_scheme13_full.py --epoch {args.epoch + 1} --resume {target.relative_to(ROOT)}" if args.epoch < 3 else None,
    }
    save_json(target / "manifest.json", manifest)
    print(f"чекпоинт эпохи {args.epoch} сохранён, файлов {len(entries)}, секунд {time.perf_counter() - started:.1f}", flush=True)
    report_path = ROOT / f"artifacts/scheme13_full_e{args.epoch}.json"
    report = {"status": "оценка holdout", "epoch": args.epoch, "checkpoint": str(target.relative_to(ROOT)), "loss": history}
    save_json(report_path, report)
    report["holdout"] = evaluate(tokenizer, model, heldout)
    report["status"] = "готово"
    report["total_seconds"] = round(time.perf_counter() - started, 3)
    save_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
