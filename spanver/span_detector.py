import json
import random
import time
from pathlib import Path

from .evaluate import metrics
from .model_features import DOC_CHUNK_CHARS, index_chunks, retrieve
from .module import confidence, predict_type

MAX_LENGTH = 512
BATCH = 8
EPOCHS = 2
RATE = 5e-5
SEED = 0
THREADS = 4
TOP_CHUNKS = 2
IGNORE = -100
LABEL_NAMES = ("supported", "hallucination")


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


def token_offsets(tokenizer, context, answer, max_length=MAX_LENGTH):
    encoding = tokenizer(
        context,
        answer,
        truncation="only_first",
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    offsets = encoding.pop("offset_mapping")[0]
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
    total = encoding["input_ids"].shape[1]
    start = max(0, total - answer_tokens - 1)
    return encoding, offsets, start


def span_labels(answer, gold, offsets, start, total):
    labels = [IGNORE] * total
    for index in range(start, total):
        left, right = offsets[index].tolist()
        if left == right:
            labels[index] = IGNORE
            continue
        label = 0
        for span in gold:
            overlap = min(right, span["end"] + 1) - max(left, span["start"])
            if overlap > 0:
                label = 1
                break
        labels[index] = label
    return labels


def chunk_for(answer, document, gold):
    pieces, stems = index_chunks(document, DOC_CHUNK_CHARS)
    if not pieces:
        return ""
    if gold:
        best = None
        position = 0
        for index, piece in enumerate(pieces):
            for span in gold:
                snippet = piece
                overlap = 0
                for token in snippet.split():
                    if token in answer:
                        overlap += 1
                if best is None or overlap > best[0]:
                    best = (overlap, index)
        position = best[1] if best else 0
        return pieces[position]
    selected = retrieve(answer, pieces, stems, TOP_CHUNKS)
    return pieces[selected[0]] if selected else pieces[0]


def examples_for(pairs, tokenizer, max_length=MAX_LENGTH):
    rows = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        if not answer or not document:
            continue
        context = chunk_for(answer, document, pair.get("spans") or [])
        encoding, offsets, start = token_offsets(tokenizer, context, answer, max_length)
        total = encoding["input_ids"].shape[1]
        labels = span_labels(answer, pair.get("spans") or [], offsets, start, total)
        rows.append(
            {
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "labels": labels,
            }
        )
    return rows


def collate(torch, rows, pad_id):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention = []
    labels = []
    for row in rows:
        padding = width - len(row["input_ids"])
        input_ids.append(torch.cat([row["input_ids"], torch.full((padding,), pad_id, dtype=torch.long)]))
        attention.append(torch.cat([row["attention_mask"], torch.zeros(padding, dtype=torch.long)]))
        labels.append(row["labels"] + [IGNORE] * padding)
    return (
        torch.stack(input_ids),
        torch.stack(attention),
        torch.tensor(labels, dtype=torch.long),
    )


def train(torch, model, tokenizer, rows, epochs=EPOCHS, batch=BATCH, rate=RATE, seed=SEED):
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    rng = random.Random(seed)
    order_source = list(range(len(rows)))
    history = []
    model.train()
    for epoch in range(epochs):
        rng.shuffle(order_source)
        total = 0.0
        steps = 0
        for position in range(0, len(order_source), batch):
            index = order_source[position : position + batch]
            batch_rows = [rows[i] for i in index]
            input_ids, attention_mask, labels = collate(torch, batch_rows, tokenizer.pad_token_id)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.view(-1, 2), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        history.append(round(total / max(1, steps), 6))
    model.eval()
    return history


def save_checkpoint(model, tokenizer, directory):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return target


def prefilter(answer, document, top=TOP_CHUNKS):
    pieces, stems = index_chunks(document, DOC_CHUNK_CHARS)
    if not pieces:
        return [""]
    chosen = retrieve(answer, pieces, stems, top)
    return [pieces[index] for index in chosen] if chosen else [pieces[0]]


THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def pair_probabilities(tokenizer, model, torch, pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    context = prefilter(answer, document)
    encoding, offsets, start = token_offsets(tokenizer, "\n".join(context), answer)
    if not answer:
        return None
    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    total = encoding["input_ids"].shape[1]
    answer_offset = offsets[start][0].item() if start < offsets.size(0) else 0
    pieces = []
    for index in range(start, total):
        left, right = offsets[index].tolist()
        if left == right:
            continue
        pieces.append((left - answer_offset, right - answer_offset, float(probabilities[index, 1])))
    return {"answer": answer, "pieces": pieces}


def spans_from(cached, threshold):
    answer = cached["answer"]
    spans = []
    current = None
    for left, right, score in cached["pieces"]:
        if score >= threshold:
            if current is None:
                current = {"start": left, "end": right, "risk": score}
            else:
                current["end"] = right
                current["risk"] = max(current["risk"], score)
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    result = []
    for span in spans:
        fragment = answer[span["start"] : span["end"]]
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


def evaluate(tokenizer, model, torch, pairs, limit=0, threshold=0.5):
    chosen = pairs[:limit] if limit else pairs
    cache = {}
    for pair in chosen:
        cache[pair.get("pair_id")] = pair_probabilities(tokenizer, model, torch, pair)
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
