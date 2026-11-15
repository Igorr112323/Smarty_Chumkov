import torch
from torch import nn

from .module import (
    CAPITAL_RE,
    DIGIT_TOKEN_RE,
    NEGATION_WORDS,
    NUMERIC_WORDS,
    TOKEN_RE,
    UNCERTAINTY_WORDS,
    document_token_set,
    marker_offsets,
    stem,
)
from .span_detector import IGNORE, choose_chunks, encode_pair

FEATURE_COUNT = 5
EDGE = -1e9


def answer_features(answer, document):
    text = answer or ""
    document_tokens = document_token_set(document or "")
    markers = marker_offsets(text, UNCERTAINTY_WORDS)
    negations = marker_offsets(text, NEGATION_WORDS)
    rows = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        start, end = match.span()
        number = 1.0 if (DIGIT_TOKEN_RE.search(token) or lowered in NUMERIC_WORDS) else 0.0
        entity = 1.0 if (CAPITAL_RE.match(token) and start > 0) else 0.0
        uncertainty = 1.0 if any(left - 25 <= start <= right + 25 for left, right in markers) else 0.0
        negation = 1.0 if any(left - 12 <= start <= right + 12 for left, right in negations) else 0.0
        overlap = 1.0 if (lowered in document_tokens or token in document_tokens or stem(lowered) in document_tokens) else 0.0
        rows.append((start, end, [number, entity, uncertainty, negation, overlap]))
    return rows


def aligned(rows, start, end):
    values = [0.0] * FEATURE_COUNT
    for left, right, flags in rows:
        if min(end, right) - max(start, left) > 0:
            for index, flag in enumerate(flags):
                values[index] = max(values[index], flag)
    return values


def build_row(tokenizer, pair, max_length=512):
    answer = pair.get("model_output") or ""
    if not answer:
        return None
    chunks = choose_chunks(tokenizer, pair, top=1, max_length=max_length)
    encoded = encode_pair(tokenizer, chunks[0]["text"] if chunks else "", answer, max_length)
    if encoded is None:
        return None
    rows = answer_features(answer, pair.get("document_text") or "")
    total = len(encoded["input_ids"])
    labels = [IGNORE] * total
    answer_mask = [0] * total
    features = [[0.0] * FEATURE_COUNT for _ in range(total)]
    gold = pair.get("spans") or []
    for index, start, end in encoded["answer_tokens"]:
        answer_mask[index] = 1
        features[index] = aligned(rows, start, end + 1)
        labels[index] = 1 if any(min(end, span["end"] + 1) > max(start, span["start"]) for span in gold) else 0
    return {
        "pair_id": pair.get("pair_id"),
        "input_ids": encoded["input_ids"],
        "attention_mask": [1] * total,
        "labels": labels,
        "answer_mask": answer_mask,
        "features": features,
        "target": 1.0 if gold else float(pair.get("label", 0)),
    }


def build_table(tokenizer, pairs, max_length=512):
    table = {}
    for pair in pairs:
        row = build_row(tokenizer, pair, max_length)
        if row is not None:
            table[row["pair_id"]] = row
    return table


def collate(rows, pad_id):
    width = max(len(row["input_ids"]) for row in rows)
    batch = {"input_ids": [], "attention_mask": [], "labels": [], "answer_mask": [], "features": [], "targets": []}
    for row in rows:
        padding = width - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_id] * padding)
        batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
        batch["labels"].append(row["labels"] + [IGNORE] * padding)
        batch["answer_mask"].append(row["answer_mask"] + [0] * padding)
        batch["features"].append(row["features"] + [[0.0] * FEATURE_COUNT for _ in range(padding)])
        batch["targets"].append(row["target"])
    return {
        "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(batch["labels"], dtype=torch.long),
        "answer_mask": torch.tensor(batch["answer_mask"], dtype=torch.float),
        "features": torch.tensor(batch["features"], dtype=torch.float),
        "targets": torch.tensor(batch["targets"], dtype=torch.float),
    }


class FeatureModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(base.config.hidden_size + FEATURE_COUNT, base.config.hidden_size)

    def forward(self, input_ids, attention_mask, features):
        embeddings = self.base.bert.embeddings(input_ids)
        mixed = self.proj(torch.cat([embeddings, features.to(embeddings.dtype)], dim=-1))
        mask = (1.0 - attention_mask[:, None, None, :].to(mixed.dtype)) * torch.finfo(mixed.dtype).min
        hidden = self.base.bert.encoder(mixed, attention_mask=mask)[0]
        return self.base.classifier(hidden)


def load_feature_model(directory, labels=2):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    base = AutoModelForTokenClassification.from_pretrained(
        str(directory), num_labels=labels, local_files_only=True, ignore_mismatched_sizes=True
    )
    return tokenizer, FeatureModel(base)


def save_feature_model(model, tokenizer, directory, source):
    from pathlib import Path

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "source": str(source)}, target / "feature_model.pt")
    tokenizer.save_pretrained(str(target))
    return target


def pair_probabilities(tokenizer, model, pair, top=4, max_length=512):
    answer = pair.get("model_output") or ""
    if not answer:
        return None
    rows = answer_features(answer, pair.get("document_text") or "")
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
    batch_mask = [[1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"])) for item in encoded]
    batch_features = []
    for item in encoded:
        values = [[0.0] * FEATURE_COUNT for _ in range(width)]
        for index, start, end in item["answer_tokens"]:
            values[index] = aligned(rows, start, end + 1)
        batch_features.append(values)
    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor(batch_ids, dtype=torch.long),
            attention_mask=torch.tensor(batch_mask, dtype=torch.long),
            features=torch.tensor(batch_features, dtype=torch.float),
        )
        probabilities = torch.softmax(logits, dim=-1)[:, :, 1]
    best = {}
    for position, item in enumerate(encoded):
        for index, start, end in item["answer_tokens"]:
            value = float(probabilities[position][index])
            previous = best.get(start)
            if previous is None or value > previous[2]:
                best[start] = (start, end, value)
    pieces = [best[key] for key in sorted(best)]
    return {"answer": answer, "pieces": pieces}


def train(torch_module, model, table, plans, head_rate, encoder_rate, pad_id, accumulate=4, batch=4, weight_decay=0.01, warmup=0.1, pair_weight=0.3, smoothing=0.1, focal_gamma=0.0, class_weight=1.0):
    head = []
    encoder = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("proj") or "classifier" in name:
            head.append(parameter)
        else:
            encoder.append(parameter)
    optimizer = torch_module.optim.AdamW(
        [{"params": encoder, "lr": encoder_rate}, {"params": head, "lr": head_rate}], weight_decay=weight_decay
    )
    total = max(1, sum(len(plan) for plan in plans))
    warmup_steps = max(1, int(total * warmup))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total - step) / max(1, total - warmup_steps))

    scheduler = torch_module.optim.lr_scheduler.LambdaLR(optimizer, factor)
    micro = max(1, accumulate)
    history = []
    for epoch, plan in enumerate(plans, 1):
        model.train()
        token_total = 0.0
        pair_total = 0.0
        steps = 0
        optimizer.zero_grad()
        for batch_ids in plan:
            rows = [table[pair_id] for pair_id in batch_ids if pair_id in table]
            if not rows:
                continue
            for start in range(0, len(rows), batch):
                piece = collate(rows[start : start + batch], pad_id)
                logits = model(input_ids=piece["input_ids"], attention_mask=piece["attention_mask"], features=piece["features"])
                labels = piece["labels"]
                valid = labels != IGNORE
                token_loss = None
                if bool(valid.any()):
                    weights = torch_module.tensor([class_weight, 1.0], dtype=logits.dtype)
                    flat = torch_module.nn.functional.cross_entropy(
                        logits.view(-1, 2),
                        labels.view(-1),
                        ignore_index=IGNORE,
                        label_smoothing=smoothing,
                        weight=weights,
                        reduction="none",
                    )
                    keep = (labels.view(-1) != IGNORE).to(flat.dtype)
                    if focal_gamma > 0.0:
                        probabilities = torch_module.nn.functional.softmax(logits.view(-1, 2), dim=-1)
                        chosen = probabilities.gather(1, labels.view(-1, 1).clamp(min=0)).squeeze(1)
                        flat = flat * (1.0 - chosen).clamp(min=0.0) ** focal_gamma
                    token_loss = (flat * keep).sum() / keep.sum().clamp(min=1.0)
                diff = logits[:, :, 1] - logits[:, :, 0]
                masked = diff.masked_fill(piece["answer_mask"] == 0, EDGE)
                scores = masked.max(dim=1).values
                pair_loss = torch_module.nn.functional.binary_cross_entropy_with_logits(scores, piece["targets"])
                loss = pair_weight * pair_loss if token_loss is None else token_loss + pair_weight * pair_loss
                (loss / micro).backward()
                token_total += 0.0 if token_loss is None else float(token_loss.detach())
                pair_total += float(pair_loss.detach())
                if (start + batch) >= len(rows):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    steps += 1
        model.eval()
        rates = scheduler.get_last_lr()
        history.append(
            {
                "epoch": epoch,
                "token_loss": round(token_total / max(1, steps), 6),
                "pair_loss": round(pair_total / max(1, steps), 6),
                "lr_encoder": round(rates[0], 12),
                "lr_head": round(rates[-1], 12),
                "batches": len(plan),
            }
        )
    return history



def load_saved(directory, source):
    from pathlib import Path

    tokenizer, model = load_feature_model(source)
    saved = torch.load(Path(directory) / "feature_model.pt", map_location="cpu")
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return tokenizer, model


def batched_probabilities(tokenizer, model, pairs, top=4, max_length=512, batch=16):
    from .span_detector import choose_chunks, encode_pair

    items = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        if not answer:
            continue
        rows = answer_features(answer, pair.get("document_text") or "")
        for chunk in choose_chunks(tokenizer, pair, top=top, max_length=max_length):
            encoded = encode_pair(tokenizer, chunk["text"], answer, max_length)
            if encoded is not None:
                items.append((pair.get("pair_id"), answer, rows, encoded))
    results = {}
    pad = tokenizer.pad_token_id
    for start in range(0, len(items), batch):
        piece = items[start : start + batch]
        width = max(len(item[3]["input_ids"]) for item in piece)
        ids = [item[3]["input_ids"] + [pad] * (width - len(item[3]["input_ids"])) for item in piece]
        mask = [[1] * len(item[3]["input_ids"]) + [0] * (width - len(item[3]["input_ids"])) for item in piece]
        features = []
        for item in piece:
            values = [[0.0] * FEATURE_COUNT for _ in range(width)]
            for index, left, right in item[3]["answer_tokens"]:
                values[index] = aligned(item[2], left, right + 1)
            features.append(values)
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor(ids, dtype=torch.long),
                attention_mask=torch.tensor(mask, dtype=torch.long),
                features=torch.tensor(features, dtype=torch.float),
            )
            probabilities = torch.softmax(logits, dim=-1)[:, :, 1]
        for position, item in enumerate(piece):
            key, answer = item[0], item[1]
            best = results.setdefault(key, {"answer": answer, "best": {}})["best"]
            for index, left, right in item[3]["answer_tokens"]:
                value = float(probabilities[position][index])
                previous = best.get(left)
                if previous is None or value > previous[2]:
                    best[left] = (left, right, value)
    return {key: {"answer": value["answer"], "pieces": [value["best"][left] for left in sorted(value["best"])]} for key, value in results.items()}
