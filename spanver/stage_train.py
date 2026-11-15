import math
import random

from .span_detector import IGNORE, choose_chunks, encode_pair

PAIR_WEIGHT = 0.3
LABEL_SMOOTHING = 0.1
HEAD_RATE = 5e-4
ENCODER_RATE = 2e-5
STAGE_C_DIVIDER = 5.0
WEIGHT_DECAY = 0.01
WARMUP = 0.1
SEED = 0
BATCH = 4
ACCUMULATE = 4
OURS_IN_BATCH = 5
RUFACTS_IN_BATCH = 11
EDGE = -1e9


def overlaps(start, end, gold):
    for span in gold or []:
        if min(end, span["end"] + 1) - max(start, span["start"]) > 0:
            return True
    return False


def make_stage_row(tokenizer, pair, mode, max_length=512):
    answer = pair.get("model_output") or ""
    if not answer:
        return None
    chunks = choose_chunks(tokenizer, pair, top=1, max_length=max_length)
    encoded = encode_pair(tokenizer, chunks[0]["text"] if chunks else "", answer, max_length)
    if encoded is None:
        return None
    total = len(encoded["input_ids"])
    answer_mask = [0] * total
    labels = [IGNORE] * total
    gold = pair.get("spans") or []
    for index, start, end in encoded["answer_tokens"]:
        answer_mask[index] = 1
        if mode == "token":
            labels[index] = 1 if overlaps(start, end, gold) else 0
    target = 1.0 if gold else float(pair.get("label", 0))
    return {
        "pair_id": pair.get("pair_id"),
        "input_ids": encoded["input_ids"],
        "attention_mask": [1] * total,
        "labels": labels,
        "answer_mask": answer_mask,
        "target": target,
        "positive": bool(gold),
    }


def build_table(tokenizer, pairs, mode):
    table = {}
    for pair in pairs:
        row = make_stage_row(tokenizer, pair, mode)
        if row is not None:
            table[row["pair_id"]] = row
    return table


def collate_stage(torch, rows, pad_id):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention = []
    labels = []
    answer_mask = []
    targets = []
    for row in rows:
        padding = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * padding)
        attention.append(row["attention_mask"] + [0] * padding)
        labels.append(row["labels"] + [IGNORE] * padding)
        answer_mask.append(row["answer_mask"] + [0] * padding)
        targets.append(row["target"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "answer_mask": torch.tensor(answer_mask, dtype=torch.float),
        "targets": torch.tensor(targets, dtype=torch.float),
    }


def make_optimizer(torch, model, total_steps, head_rate=HEAD_RATE, encoder_rate=ENCODER_RATE):
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
        weight_decay=WEIGHT_DECAY,
    )
    total = max(1, total_steps)
    warmup_steps = max(1, int(total * WARMUP))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total - step) / max(1, total - warmup_steps))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def batch_loss(torch, model, batch, pair_weight=PAIR_WEIGHT):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits
    labels = batch["labels"]
    token_loss = None
    valid = labels != IGNORE
    if bool(valid.any()):
        token_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 2), labels.view(-1), ignore_index=IGNORE, label_smoothing=LABEL_SMOOTHING
        )
    diff = logits[:, :, 1] - logits[:, :, 0]
    masked = diff.masked_fill(batch["answer_mask"] == 0, EDGE)
    scores = masked.max(dim=1).values
    pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, batch["targets"])
    if token_loss is None:
        return pair_weight * pair_loss, 0.0, float(pair_loss.detach())
    return token_loss + pair_weight * pair_loss, float(token_loss.detach()), float(pair_loss.detach())


def run_stage(torch, model, tokenizer, table, plans, head_rate, encoder_rate, accumulate=ACCUMULATE, batch=BATCH):
    micro = max(1, accumulate)
    total_steps = max(1, sum(len(plan) for plan in plans))
    optimizer, scheduler = make_optimizer(torch, model, total_steps, head_rate, encoder_rate)
    history = []
    for epoch, plan in enumerate(plans, 1):
        model.train()
        token_total = 0.0
        pair_total = 0.0
        steps = 0
        optimizer.zero_grad()
        for position, batch_ids in enumerate(plan):
            rows = [table[pair_id] for pair_id in batch_ids if pair_id in table]
            if not rows:
                continue
            for start in range(0, len(rows), batch):
                piece = batch_rows = rows[start : start + batch]
                data = collate_stage(torch, batch_rows, tokenizer.pad_token_id)
                loss, token_value, pair_value = batch_loss(torch, model, data)
                (loss / micro).backward()
                token_total += token_value
                pair_total += pair_value
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


def chunks_of(ids, size, rng, limit=0):
    pool = list(ids)
    rng.shuffle(pool)
    plans = []
    for start in range(0, len(pool), size):
        piece = pool[start : start + size]
        if limit and len(plans) >= limit:
            break
        plans.append(piece)
    return plans


def stage_a_plans(ids, epochs, size=16, seed=SEED):
    return [chunks_of(ids, size, random.Random(seed + epoch)) for epoch in range(1, epochs + 1)]


def stage_b_plans(positive_ids, negative_ids, epochs, positive_per_batch=8, negative_per_batch=8, seed=SEED):
    plans = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(seed + 100 + epoch)
        positive = list(positive_ids)
        negative = list(negative_ids)
        rng.shuffle(positive)
        rng.shuffle(negative)
        count = min(len(positive) // positive_per_batch, len(negative) // negative_per_batch)
        batches = []
        for index in range(count):
            batch = positive[index * positive_per_batch : (index + 1) * positive_per_batch]
            batch += negative[index * negative_per_batch : (index + 1) * negative_per_batch]
            rng.shuffle(batch)
            batches.append(batch)
        plans.append(batches)
    return plans


def stage_c_plans(our_ids, helper_ids, epochs, ours_per_batch=OURS_IN_BATCH, helper_per_batch=RUFACTS_IN_BATCH, seed=SEED):
    plans = []
    for epoch in range(1, epochs + 1):
        rng = random.Random(seed + 200 + epoch)
        ours = list(our_ids)
        helpers = list(helper_ids)
        rng.shuffle(ours)
        rng.shuffle(helpers)
        count = max(1, math.ceil(len(ours) / ours_per_batch))
        ours_pool = list(ours)
        helper_pool = list(helpers)
        batches = []
        for _ in range(count):
            if len(ours_pool) < ours_per_batch:
                ours_pool.extend(ours)
                rng.shuffle(ours_pool)
            if len(helper_pool) < helper_per_batch:
                helper_pool.extend(helpers)
                rng.shuffle(helper_pool)
            batch = [ours_pool.pop() for _ in range(min(ours_per_batch, len(ours_pool)))]
            batch += [helper_pool.pop() for _ in range(min(helper_per_batch, len(helper_pool)))]
            rng.shuffle(batch)
            batches.append(batch)
        plans.append(batches)
    return plans
