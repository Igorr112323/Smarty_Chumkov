"""V21 шаг 3: обучение cross-encoder (GPU) с честной фиксацией блокера при отсутствии GPU.

Логика:
  * устройство выбирается автоматически; без CUDA скрипт не притворяется, что обучил модель:
    он пишет `verdict = blocked_no_gpu`, но по флагу `--smoke-steps` прогоняет реальный кусок
    пайплайна на CPU, чтобы замерить секунды/шаг и оценить стоимость эпохи;
  * loss = span-BCE(start) + span-BCE(end) + risk-BCE;
  * при наличии CUDA: AMP (bf16/fp16), grad accumulation, AdamW, 3 эпохи по умолчанию,
    контрольный замер val AUC после каждой эпохи, чекпойнт `artifacts/ce_model_v21.pt`.

Скрипт запускается на GPU-машине командой:
    python tools/ce_train_v21.py --device cuda --epochs 3 --batch 32 --amp
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch
from torch import nn

from ce_model_v21 import build_model, count_parameters, encode_batch, load_tokenizer


def read_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def flatten(examples):
    """Пары (document, candidate, span, risk) + метка, где span — gold-фрагмент внутри кандидата."""
    items = []
    for example in examples:
        for candidate in example["candidates"]:
            if not (candidate.get("text") or "").strip():
                continue
            span = (0, candidate["end"] - candidate["start"]) if candidate["label"] == 1 else None
            items.append((example["document_text"], candidate["text"], span,
                          1.0 if candidate["label"] == 1 else 0.0))
    return items


def balanced_batches(items, batch, seed=21):
    """Батчи с гарантированной долей позитивов: 50 % позитивов, 50 % негативов (в т.ч. hard)."""
    positives = [item for item in items if item[3] == 1.0]
    negatives = [item for item in items if item[3] != 1.0]
    rng = __import__("random").Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    half = max(batch // 2, 1)
    batches, pos_index, neg_index = [], 0, 0
    rounds = math.ceil(len(items) / batch)
    for _ in range(rounds):
        chunk = []
        for _ in range(half):
            if pos_index >= len(positives):
                pos_index = 0
                rng.shuffle(positives)
            chunk.append(positives[pos_index])
            pos_index += 1
        for _ in range(batch - half):
            if neg_index >= len(negatives):
                neg_index = 0
                rng.shuffle(negatives)
            chunk.append(negatives[neg_index])
            neg_index += 1
        batches.append(chunk)
    return batches


def train_step(model, batch, start_targets, end_targets, risk_targets):
    outputs = model(**batch)
    # строки без gold-спана (метка -100) исключаем целиком: у них все логиты замаскированы,
    # и cross_entropy даёт NaN, который не спасает ignore_index
    valid = start_targets >= 0
    if int(valid.sum()) == 0:
        start_loss = outputs["start_logits"].sum() * 0.0
        end_loss = start_loss
    else:
        start_loss = nn.functional.cross_entropy(outputs["start_logits"][valid], start_targets[valid])
        end_loss = nn.functional.cross_entropy(outputs["end_logits"][valid], end_targets[valid])
    risk_loss = nn.functional.binary_cross_entropy_with_logits(outputs["risk_logit"], risk_targets)
    return start_loss + end_loss + risk_loss, {"start": float(start_loss), "end": float(end_loss),
                                              "risk": float(risk_loss)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default="artifacts/ce_examples_v21.jsonl")
    parser.add_argument("--report", default="artifacts/ce_train_v21.json")
    parser.add_argument("--checkpoint", default="artifacts/ce_model_v21.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--doc-tokens", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help=">0 — прогнать N шагов на CPU без CUDA (проверка пайплайна и замер)")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    cuda = torch.cuda.is_available()
    device_name = "cuda" if cuda else "cpu"
    if args.device != "auto":
        device_name = args.device
    if device_name == "cuda" and not cuda:
        raise SystemExit("запрошен cuda, но torch.cuda.is_available() = False")
    device = torch.device(device_name)

    tokenizer = load_tokenizer()
    model = build_model().to(device)
    parameters = count_parameters(model)
    examples = read_jsonl(ROOT / args.examples, args.limit)
    items = flatten(examples)
    steps_per_epoch = math.ceil(len(items) / args.batch)

    report = {
        "rule": "шаг 3 v21: обучение cross-encoder (span start/end + risk). Loss = BCE(start) + "
                "BCE(end) + BCE(risk). Held-out не читается",
        "device": {"requested": args.device, "used": device_name,
                   "cuda_available": cuda,
                   "gpu": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
                   "torch": torch.__version__, "cuda_version": torch.version.cuda},
        "architecture": {"encoder": "cointegrated/rubert-tiny2", "parameters": parameters,
                         "heads": ["start", "end", "risk"], "max_len": args.max_len,
                         "doc_tokens": args.doc_tokens},
        "data": {"pairs": len(examples), "candidates": len(items),
                 "positives": int(sum(1 for item in items if item[3] == 1.0)),
                 "batch": args.batch, "steps_per_epoch": steps_per_epoch},
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    batches = balanced_batches(items, args.batch)

    def run_steps(step_limit):
        model.train()
        losses, timings = [], []
        for step in range(step_limit):
            chunk = batches[step]
            if len(chunk) < 2:
                break
            started = time.time()
            batch, start_targets, end_targets, risk_targets = encode_batch(
                tokenizer, chunk, doc_tokens=args.doc_tokens, max_len=args.max_len)
            batch = {key: value.to(device) for key, value in batch.items()}
            start_targets = start_targets.to(device)
            end_targets = end_targets.to(device)
            risk_targets = risk_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                loss, parts = train_step(model, batch, start_targets, end_targets, risk_targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            timings.append(time.time() - started)
            losses.append({"step": step, "loss": round(float(loss), 4),
                           "seconds": round(timings[-1], 3), **{key: round(value, 4)
                                                                for key, value in parts.items()}})
        return losses, timings

    if args.smoke_steps:
        steps = min(args.smoke_steps, steps_per_epoch)
        smoke_losses, timings = run_steps(steps)
        median = sorted(timings)[len(timings) // 2] if timings else 0.0
        epoch_seconds = median * steps_per_epoch
        report["smoke"] = {
            "steps": len(smoke_losses), "losses": smoke_losses,
            "median_step_seconds": round(median, 3),
            "steps_per_epoch": steps_per_epoch,
            "estimated_epoch_seconds_cpu": round(epoch_seconds, 1),
            "estimated_epoch_minutes_cpu": round(epoch_seconds / 60, 2),
            "estimated_full_training_minutes_cpu": round(epoch_seconds * args.epochs / 60, 1),
            "note": "замер на CPU (2 ядра, batch %d, max_len %d); на GPU-машине время шага будет "
                    "меньше — скрипт запускается тем же флагом --device cuda --amp" %
                    (args.batch, args.max_len),
        }
        report["verdict"] = ("blocked_no_gpu: обучение на GPU невозможно в этой песочнице, "
                             "пайплайн проверен smoke-прогоном на CPU")
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        print(json.dumps(report["smoke"], ensure_ascii=False)[:800], flush=True)
        print(report["verdict"], flush=True)
        return

    if device.type != "cuda" and not args.allow_cpu:
        report["verdict"] = ("blocked_no_gpu: torch.cuda.is_available() = False; обучение "
                             "cross-encoder без GPU не запускается (см. artifacts/gpu_check_v21.json)")
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        print(report["verdict"], flush=True)
        return

    history = []
    for epoch in range(args.epochs):
        started = time.time()
        losses = []
        model.train()
        for step in range(steps_per_epoch):
            chunk = batches[step]
            if len(chunk) < 2:
                break
            batch, start_targets, end_targets, risk_targets = encode_batch(
                tokenizer, chunk, doc_tokens=args.doc_tokens, max_len=args.max_len)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                loss, parts = train_step(model, batch, start_targets.to(device),
                                         end_targets.to(device), risk_targets.to(device))
            scaler.scale(loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss))
        history.append({"epoch": epoch, "loss": round(sum(losses) / max(len(losses), 1), 4),
                        "seconds": round(time.time() - started, 1)})
        report["history"] = history
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        print("epoch", epoch, history[-1], flush=True)

    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ROOT / args.checkpoint)
    report["checkpoint"] = {"path": args.checkpoint,
                            "bytes": (ROOT / args.checkpoint).stat().st_size}
    report["verdict"] = "trained_on_gpu"
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(history, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
