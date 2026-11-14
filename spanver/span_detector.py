import math
import random
from pathlib import Path

from .evaluate import metrics
from .module import confidence, predict_type, stem, tokens

MAX_LENGTH = 512
CHUNK_TOKENS = 380
OVERLAP = 0.25
MIN_CHUNK_TOKENS = 64
TOP_K = 4
BATCH = 4
ACCUMULATE = 4
EPOCHS = 8
HEAD_RATE = 1e-3
ENCODER_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP = 0.1
SEED = 0
THREADS = 4
IGNORE = -100
LABEL_NAMES = ("supported", "hallucination")
THRESHOLD_GRID = tuple(round(0.05 * step, 2) for step in range(1, 20))


def runtime():
    import torch

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    return torch


def load_model(directory, labels=2):
    torch = runtime()
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        str(directory), num_labels=labels, local_files_only=True, ignore_mismatched_sizes=True
    )
    model.eval()
    return torch, tokenizer, model


def token_count(tokenizer, text):
    return len(tokenizer(text or "", add_special_tokens=False)["input_ids"])


def chunk_budget(tokenizer, answer, max_length=MAX_LENGTH):
    used = token_count(tokenizer, answer)
    room = max_length - 3 - used
    return max(MIN_CHUNK_TOKENS, min(CHUNK_TOKENS, room))


def document_chunks(tokenizer, document, size=CHUNK_TOKENS, overlap=OVERLAP):
    text = document or ""
    if not text.strip():
        return [{"index": 0, "start": 0, "end": len(text), "text": "", "tokens": 0}]
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    total = len(offsets)
    step = max(1, int(round(size * (1.0 - overlap))))
    chunks = []
    start = 0
    while start < total:
        end = min(start + size, total)
        left = int(offsets[start][0])
        right = int(offsets[end - 1][1])
        chunks.append(
            {
                "index": len(chunks),
                "start": left,
                "end": right,
                "text": text[left:right],
                "tokens": end - start,
            }
        )
        if end >= total:
            break
        start += step
    return chunks


def overlap_score(answer, chunk_text):
    wanted = {stem(token) for token in tokens(answer)}
    if not wanted:
        return 0.0
    have = {stem(token) for token in tokens(chunk_text)}
    return round(len(wanted & have) / len(wanted), 6)


def choose_chunks(tokenizer, pair, top=TOP_K, max_length=MAX_LENGTH):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    size = chunk_budget(tokenizer, answer, max_length)
    chunks = document_chunks(tokenizer, document, size)
    scored = []
    for chunk in chunks:
        scored.append((-overlap_score(answer, chunk["text"]), chunk["index"], chunk))
    scored.sort(key=lambda item: (item[0], item[1]))
    chosen = [chunk for _, _, chunk in scored[: max(1, top)]]
    chosen.sort(key=lambda item: item["index"])
    return chosen


def encode_pair(tokenizer, chunk_text, answer, max_length=MAX_LENGTH):
    encoded = tokenizer(
        chunk_text or "",
        answer or "",
        truncation="longest_first",
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    ids = [int(item) for item in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    separator = tokenizer.sep_token_id
    marks = [index for index, token in enumerate(ids) if token == separator]
    if len(marks) < 2:
        return None
    left = marks[0] + 1
    right = marks[-1]
    answer_tokens = []
    for index in range(left, right):
        start, end = offsets[index]
        if start == end:
            continue
        answer_tokens.append((index, start, end))
    if not answer_tokens:
        return None
    return {"input_ids": ids, "answer_left": left, "answer_right": right, "answer_tokens": answer_tokens}


def answer_labels(encoded, gold):
    labels = [IGNORE] * len(encoded["input_ids"])
    spans = gold or []
    for index, start, end in encoded["answer_tokens"]:
        label = 0
        for span in spans:
            if min(end, span["end"] + 1) - max(start, span["start"]) > 0:
                label = 1
                break
        labels[index] = label
    return labels


def make_row(tokenizer, pair, top=1, max_length=MAX_LENGTH):
    answer = pair.get("model_output") or ""
    if not answer:
        return None
    chunks = choose_chunks(tokenizer, pair, top=top, max_length=max_length)
    encoded = encode_pair(tokenizer, chunks[0]["text"] if chunks else "", answer, max_length)
    if encoded is None:
        return None
    return {
        "pair_id": pair.get("pair_id"),
        "input_ids": encoded["input_ids"],
        "attention_mask": [1] * len(encoded["input_ids"]),
        "labels": answer_labels(encoded, pair.get("spans") or []),
        "chunks": len(chunks),
    }


def examples_for(tokenizer, pairs, top=1, max_length=MAX_LENGTH):
    rows = []
    for pair in pairs:
        row = make_row(tokenizer, pair, top=top, max_length=max_length)
        if row is not None:
            rows.append(row)
    return rows


def collate(torch, rows, pad_id):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention = []
    labels = []
    for row in rows:
        padding = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * padding)
        attention.append(row["attention_mask"] + [0] * padding)
        labels.append(row["labels"] + [IGNORE] * padding)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def train(
    torch,
    model,
    tokenizer,
    rows,
    epochs=EPOCHS,
    batch=BATCH,
    accumulate=ACCUMULATE,
    head_rate=HEAD_RATE,
    encoder_rate=ENCODER_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup=WARMUP,
    seed=SEED,
    on_epoch=None,
    stop_loss=None,
    stop_after=5,
):
    head = []
    encoder = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "classifier" in name:
            head.append(parameter)
        else:
            encoder.append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": encoder_rate}, {"params": head, "lr": head_rate}],
        weight_decay=weight_decay,
    )
    micro = max(1, accumulate)
    steps_per_epoch = max(1, math.ceil(math.ceil(len(rows) / batch) / micro))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * warmup))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    history = []
    stopped = False
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        total = 0.0
        steps = 0
        chunks = list(range(0, len(order), batch))
        for position, start in enumerate(chunks):
            index = order[start : start + batch]
            input_ids, attention_mask, labels = collate(
                torch, [rows[item] for item in index], tokenizer.pad_token_id
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.view(-1, 2), labels.view(-1)) / micro
            loss.backward()
            total += float(loss.detach()) * micro
            if (position + 1) % micro == 0 or position + 1 == len(chunks):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1
        model.eval()
        rates = scheduler.get_last_lr()
        entry = {
            "epoch": epoch + 1,
            "loss": round(total / max(1, steps), 6),
            "lr_encoder": round(rates[0], 10),
            "lr_head": round(rates[-1], 10),
        }
        history.append(entry)
        if on_epoch is not None:
            on_epoch(entry)
        if stop_loss is not None and epoch + 1 >= stop_after and entry["loss"] >= stop_loss:
            stopped = True
            break
    model.eval()
    return {"history": history, "stopped_early": stopped, "steps_per_epoch": steps_per_epoch}


def save_checkpoint(model, tokenizer, directory):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return target


def pair_probabilities(tokenizer, model, torch, pair, top=TOP_K, max_length=MAX_LENGTH):
    answer = pair.get("model_output") or ""
    if not answer:
        return None
    chunks = choose_chunks(tokenizer, pair, top=top, max_length=max_length)
    encoded = []
    for chunk in chunks:
        item = encode_pair(tokenizer, chunk["text"], answer, max_length)
        if item is not None:
            encoded.append(item)
    if not encoded:
        return None
    width = max(len(item["input_ids"]) for item in encoded)
    pad = tokenizer.pad_token_id
    batch_ids = [item["input_ids"] + [pad] * (width - len(item["input_ids"])) for item in encoded]
    batch_mask = [
        [1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"])) for item in encoded
    ]
    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor(batch_ids, dtype=torch.long),
            attention_mask=torch.tensor(batch_mask, dtype=torch.long),
        )
    probabilities = torch.softmax(outputs.logits, dim=-1).tolist()
    scores = {}
    for row, item in enumerate(encoded):
        for position, (index, _, _) in enumerate(item["answer_tokens"]):
            value = probabilities[row][index][1]
            if value > scores.get(position, -1.0):
                scores[position] = value
    pieces = []
    for position, (_, start, end) in enumerate(encoded[0]["answer_tokens"]):
        pieces.append((start, end, round(float(scores.get(position, 0.0)), 6)))
    return {"answer": answer, "chunks": len(encoded), "pieces": pieces}


def spans_from(cached, threshold):
    answer = cached["answer"]
    spans = []
    current = None
    for start, end, score in cached["pieces"]:
        if score >= threshold:
            if current is None:
                current = {"start": start, "end": end - 1, "risk": score}
            else:
                current["end"] = end - 1
                current["risk"] = max(current["risk"], score)
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    result = []
    for span in spans:
        fragment = answer[span["start"] : span["end"] + 1]
        result.append(
            {
                "start": span["start"],
                "end": span["end"],
                "risk": round(span["risk"], 4),
                "type": predict_type(fragment, 0.0),
                "confidence": confidence(span["risk"]),
                "text": fragment,
            }
        )
    return result


def predictor_from_cache(cache, threshold):
    def predict(pair):
        cached = cache.get(pair.get("pair_id"))
        return spans_from(cached, threshold) if cached else []

    return predict


def evaluate(tokenizer, model, torch, pairs, limit=0, threshold=0.5, top=TOP_K):
    chosen = pairs[:limit] if limit else pairs
    cache = {}
    for pair in chosen:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair, top=top)
    return metrics(chosen, predictor_from_cache(cache, threshold)), cache


def scan_threshold(cache, pairs, grid=THRESHOLD_GRID):
    best = None
    history = []
    for threshold in grid:
        result = metrics(pairs, predictor_from_cache(cache, threshold))
        entry = {"threshold": threshold, "f1": result["span_level"]["f1"], "fpr": result["fpr"]}
        history.append(entry)
        if best is None or (entry["f1"], -threshold) > (best["f1"], -best["threshold"]):
            best = dict(entry)
    return best, history
