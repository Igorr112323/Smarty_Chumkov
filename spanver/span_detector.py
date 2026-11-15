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
BATCH_POSITIVE = 8
BATCH_NEGATIVE = 8
EPOCH_BATCHES = 179
EPOCHS = 10
HEAD_RATE = 1e-3
ENCODER_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP = 0.1
SEED = 0
THREADS = 4
IGNORE = -100
LABEL_NAMES = ("supported", "hallucination")
THRESHOLD_GRID = tuple(round(0.05 * step, 2) for step in range(1, 20))
GATE_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
VAL_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


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
        "positive": bool(pair.get("spans")),
    }


def examples_for(tokenizer, pairs, top=1, max_length=MAX_LENGTH):
    rows = []
    for pair in pairs:
        row = make_row(tokenizer, pair, top=top, max_length=max_length)
        if row is not None:
            rows.append(row)
    return rows


def rows_by_id(rows):
    return {row["pair_id"]: row for row in rows}


def epoch_batches(
    positive_ids,
    negative_ids,
    batches=EPOCH_BATCHES,
    batch_positive=BATCH_POSITIVE,
    batch_negative=BATCH_NEGATIVE,
    seed=SEED,
):
    rng = random.Random(seed)
    positive_store = sorted(positive_ids)
    negative_store = sorted(negative_ids)
    rng.shuffle(positive_store)
    rng.shuffle(negative_store)
    positive_pool = list(positive_store)
    negative_pool = list(negative_store)
    plan = []
    for _ in range(batches):
        batch = []
        for count, pool, store in (
            (batch_positive, positive_pool, positive_store),
            (batch_negative, negative_pool, negative_store),
        ):
            if not store:
                continue
            taken = []
            while len(taken) < count:
                if not pool:
                    pool.extend(store)
                    rng.shuffle(pool)
                taken.append(pool.pop())
            batch.extend(taken)
        rng.shuffle(batch)
        plan.append(batch)
    return plan


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


def make_optimizer(torch, model, total_steps, head_rate=HEAD_RATE, encoder_rate=ENCODER_RATE, weight_decay=WEIGHT_DECAY, warmup=WARMUP):
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
    total = max(1, total_steps)
    warmup_steps = max(1, int(total * warmup))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total - step) / max(1, total - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler


def run_epoch(torch, model, tokenizer, table, plan, optimizer, scheduler, batch=BATCH, accumulate=ACCUMULATE):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    micro = max(1, accumulate)
    model.train()
    total = 0.0
    steps = 0
    optimizer.zero_grad()
    for position, batch_ids in enumerate(plan):
        rows = [table[pair_id] for pair_id in batch_ids if pair_id in table]
        if not rows:
            continue
        for start in range(0, len(rows), batch):
            piece = rows[start : start + batch]
            input_ids, attention_mask, labels = collate(torch, piece, tokenizer.pad_token_id)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.view(-1, 2), labels.view(-1)) / micro
            loss.backward()
            total += float(loss.detach()) * micro
            if (start + batch) >= len(rows):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1
    model.eval()
    rates = scheduler.get_last_lr()
    return {
        "loss": round(total / max(1, steps), 6),
        "steps": steps,
        "lr_encoder": round(rates[0], 10),
        "lr_head": round(rates[-1], 10),
    }


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


def is_gold_token(start, end, gold):
    for span in gold or []:
        if min(end, span["end"] + 1) - max(start, span["start"]) > 0:
            return True
    return False


def token_stats(cache, pairs, threshold):
    tp = fp = fn = 0
    flagged_negative = 0
    negatives = 0
    positive_scores = []
    negative_scores = []
    maxima = 0.0
    for pair in pairs:
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            continue
        gold = pair.get("spans") or []
        hit = False
        for start, end, score in cached["pieces"]:
            maxima = max(maxima, score)
            if gold:
                positive_scores.append(score)
            else:
                negative_scores.append(score)
            predicted = score >= threshold
            if predicted:
                hit = True
            if is_gold_token(start, end, gold):
                if predicted:
                    tp += 1
                else:
                    fn += 1
            elif predicted:
                fp += 1
        if not gold:
            negatives += 1
            if hit:
                flagged_negative += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pair_fpr": round(flagged_negative / negatives, 4) if negatives else 0.0,
        "mean_p_positive_pairs": round(sum(positive_scores) / len(positive_scores), 4) if positive_scores else 0.0,
        "mean_p_negative_pairs": round(sum(negative_scores) / len(negative_scores), 4) if negative_scores else 0.0,
        "max_p": round(maxima, 4),
    }


def gate_table(cache, pairs, fixed=0.5, grid=GATE_GRID):
    rows = [token_stats(cache, pairs, threshold) for threshold in grid]
    best = max(rows, key=lambda item: (item["f1"], -abs(item["threshold"] - fixed)))
    fixed_row = next((item for item in rows if abs(item["threshold"] - fixed) < 1e-9), None)
    if fixed_row is None:
        fixed_row = token_stats(cache, pairs, fixed)
    span = metrics(pairs, predictor_from_cache(cache, best["threshold"]))
    return {"rows": rows, "best": best, "fixed": fixed_row, "span_at_best": span["span_level"]}


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


def longest_common(answer, chunk_text, minimum=10):
    best = ""
    seen = set()
    limit = min(len(answer), 400)
    for start in range(limit):
        for end in range(start + len(best) + 1, min(len(answer), start + 200) + 1):
            piece = answer[start:end]
            if piece in seen:
                continue
            seen.add(piece)
            if piece in chunk_text:
                if len(piece) > len(best):
                    best = piece
            else:
                break
    return best if len(best) >= minimum else ""


def extended_diagnostics(tokenizer, cache, pairs):
    report = []
    positive = [pair for pair in pairs if pair.get("spans")]
    negative = [pair for pair in pairs if not pair.get("spans")]
    for pair in positive:
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            continue
        gold_scores = []
        other_scores = []
        for start, end, score in cached["pieces"]:
            if is_gold_token(start, end, pair.get("spans")):
                gold_scores.append(score)
            else:
                other_scores.append(score)
        median = 0.0
        if other_scores:
            ordered = sorted(other_scores)
            median = ordered[len(ordered) // 2]
        above = sum(1 for score in gold_scores if score > median)
        chunks = choose_chunks(tokenizer, pair, top=TOP_K)
        best_chunk = chunks[0] if chunks else {"index": 0, "tokens": 0, "text": ""}
        common = longest_common(pair.get("model_output") or "", best_chunk["text"])
        report.append(
            {
                "pair_id": pair.get("pair_id"),
                "gold_tokens": len(gold_scores),
                "gold_mean_p": round(sum(gold_scores) / len(gold_scores), 4) if gold_scores else 0.0,
                "other_mean_p": round(sum(other_scores) / len(other_scores), 4) if other_scores else 0.0,
                "gold_above_other_median": round(above / len(gold_scores), 4) if gold_scores else 0.0,
                "chunk_index": best_chunk["index"],
                "chunk_tokens": best_chunk["tokens"],
                "longest_common": len(common),
            }
        )
    negatives = []
    for pair in negative:
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            continue
        scores = [score for _, _, score in cached["pieces"]]
        negatives.append(
            {
                "pair_id": pair.get("pair_id"),
                "mean_p": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "max_p": round(max(scores), 4) if scores else 0.0,
            }
        )
    spreads = [item["gold_mean_p"] - item["other_mean_p"] for item in report]
    return {
        "positive": report,
        "negative": negatives,
        "gold_mean_p": round(sum(item["gold_mean_p"] for item in report) / len(report), 4) if report else 0.0,
        "other_mean_p": round(sum(item["other_mean_p"] for item in report) / len(report), 4) if report else 0.0,
        "negative_mean_p": round(sum(item["mean_p"] for item in negatives) / len(negatives), 4) if negatives else 0.0,
        "spread_min": round(min(spreads), 4) if spreads else 0.0,
        "spread_max": round(max(spreads), 4) if spreads else 0.0,
    }
