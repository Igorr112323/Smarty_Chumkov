import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import local_model
from spanver.evaluate import read_jsonl
from spanver.model_features import (
    FRAGMENT_TOKENS,
    SEGMENT_TOKENS,
    index_chunks,
    retrieve,
    select_candidates,
)
from spanver.splits import load_splits, ordered_pairs

SEED = 0
EPOCHS = 6
BATCH = 32
RATE = 0.001
PER_PAIR = 15
SUPPORTED_PER_PAIR = 5
MODEL_DIR = "models"
META = "artifacts/local_model_train.json"


def overlap(first_start, first_end, second_start, second_end):
    left = max(first_start, second_start)
    right = min(first_end, second_end)
    if right < left:
        return 0.0
    union = max(first_end, second_end) - min(first_start, second_start) + 1
    return (right - left + 1) / union if union else 0.0


def build_examples(pairs, rng):
    examples = []
    positives = 0
    negatives = 0
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        gold = pair.get("spans") or []
        if not answer or not document:
            continue
        pieces, stems = index_chunks(document)
        if gold:
            candidates = select_candidates(answer, document, PER_PAIR * 2)[:PER_PAIR]
            for start, end, fragment in candidates:
                label = 1 if any(overlap(start, end, span["start"], span["end"]) >= 0.5 for span in gold) else 0
                examples.append((document, fragment, label))
                positives += label
                negatives += 1 - label
        sentences = [piece.strip() for piece in pieces if len(piece.strip()) > 20]
        if sentences:
            for _ in range(SUPPORTED_PER_PAIR):
                fragment = sentences[rng.randrange(len(sentences))]
                examples.append((document, fragment, 0))
                negatives += 1
    rng.shuffle(examples)
    positive_rows = [row for row in examples if row[2] == 1]
    negative_rows = [row for row in examples if row[2] == 0]
    if not positive_rows:
        return [], 0, 0
    size = min(len(positive_rows), len(negative_rows))
    chosen = positive_rows[:size] + negative_rows[:size]
    rng.shuffle(chosen)
    return chosen, size, size


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--out", default=MODEL_DIR)
    parser.add_argument("--meta", default=META)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    pairs = read_jsonl(Path(args.data))
    splits = load_splits()
    train_pairs = ordered_pairs(pairs, splits["train"])
    rng = random.Random(args.seed)
    examples, positives, negatives = build_examples(train_pairs, rng)
    if not examples:
        print("нет обучающих примеров")
        return 2

    torch, model, tokenizer, config = local_model.create(seed=args.seed)
    torch.manual_seed(args.seed)
    encoded = []
    for document, fragment, label in examples:
        packed = local_model.encode(tokenizer, document, fragment, SEGMENT_TOKENS, FRAGMENT_TOKENS)
        if packed is None:
            continue
        encoded.append((packed["ids"], label))
    if not encoded:
        print("пустой набор после токенизации")
        return 2

    width = max(len(ids) for ids, _ in encoded)
    pad = tokenizer.pad_id
    inputs = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in encoded], dtype=torch.long)
    labels = torch.tensor([label for _, label in encoded], dtype=torch.long)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        total = 0.0
        batches = 0
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for start in range(0, len(order), BATCH):
            index = order[start : start + BATCH]
            batch_inputs = inputs[index]
            batch_labels = labels[index]
            optimizer.zero_grad()
            logits, _, _ = model(batch_inputs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        history.append(round(total / batches, 6))
        print(f"эпоха {epoch + 1}/{args.epochs}: loss {history[-1]}", flush=True)
    model.eval()

    path = local_model.save(Path(args.out), model, tokenizer, config, extra={"rows": len(encoded), "seed": args.seed})
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    meta = {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "sha256": sha,
        "parameters": local_model.parameters(model),
        "config": config,
        "train_pairs": len(train_pairs),
        "examples": len(encoded),
        "positives": positives,
        "negatives": negatives,
        "epochs": args.epochs,
        "batch": BATCH,
        "rate": RATE,
        "loss": history,
        "final_loss": history[-1] if history else None,
        "seed": args.seed,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    local_model.write_meta(Path(args.meta), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
