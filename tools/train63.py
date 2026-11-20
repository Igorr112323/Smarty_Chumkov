"""ПРОМПТ 63, стадия 2: обучение посегментного отбора — один фолд, один сид.

Что обучается: `spanver.segment_model.CandidateModel` — rubert-tiny2 с двумя
головами (бинарная «есть галлюцинация» + 6 типов) поверх входа
`[CLS] вопрос [SEP] чанк [SEP] кандидат [SEP]`.

Loss: `BCE + 0.3 * CE(тип) + 0.3 * soft-IoU(границы)`. Soft-IoU — то, что
отличает схему от 52: там IoU не оптимизировался, а меряется он.

Протокол: разбиение `grouped_folds` по `doc_ids` (документное, без пересечения),
10 фолдов, 5 сидов, порог на внутренней сетке с ограничением `fpr_token <= 0,40`.

Порог выбирается по ВНУТРЕННЕМУ фолду; внешний фолд только меряется. Test 301
здесь не читается вовсе.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from spanver import lexfeat, segment_model
from spanver.evaluate import covered_tokens, match_spans, read_jsonl, token_positions
from tools.metrics59 import prf, spans_of
from tools.nested38 import file_digest, split
from tools.preflight_13 import id_digest
from tools.segments63 import candidates_with_unions, label_candidates
from tools.train_base33 import grouped_folds
from tools.train_scheme13_full import save_json

FOLDS = 10
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 8
BATCH = 16
MICRO = 4
MAX_LENGTH = 256
LR = 2e-3
CE_WEIGHT = 0.3
IOU_WEIGHT = 0.3
FPR_LIMIT = 0.40
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
TYPES = ["Excess", "Missing", "Oversight", "Partial", "Contradiction", "Unconfirmed"]
TYPE_INDEX = {name: index for index, name in enumerate(TYPES)}


def load_questions():
    from tools.fewshot56 import load_questions as loader

    return loader()


def question_of(pair, table):
    pair_id = pair["pair_id"]
    prefix = "sber" if pair_id.startswith("sber") else "rusci"
    digits = "".join(ch for ch in pair_id[len(prefix):] if ch.isdigit())
    rows = table.get(prefix) or []
    if not digits:
        return ""
    index = int(digits)
    return (rows[index] if index < len(rows) else "") or ""


def best_chunk(document, text, max_chars=1200):
    """Чанк документа с максимальным лексическим перекрытием с кандидатом (п. 2.2)."""
    if not document:
        return ""
    words = set(segment_model.content_words(text))
    if not words:
        return document[:max_chars]
    best = (0, 0)
    step = max(1, max_chars // 2)
    for start in range(0, max(1, len(document) - max_chars + 1), step):
        piece = document[start:start + max_chars]
        overlap = len(words & set(segment_model.content_words(piece)))
        if overlap > best[0]:
            best = (overlap, start)
    return document[best[1]:best[1] + max_chars]


def build_rows(tokenizer, pair, question, probabilities, pad_id=None):
    """Строки обучающей таблицы: по строке на кандидата."""
    answer = pair["model_output"] or ""
    document = pair.get("document_text") or ""
    candidates = candidates_with_unions(answer)
    marks = label_candidates(answer, candidates, spans_of(pair))
    question_words = segment_model.content_words(question)
    total = len(candidates)
    rows = []
    for rank, mark in enumerate(marks):
        text = answer[mark["start"]:mark["end"] + 1]
        chunk = best_chunk(document, text)
        head = tokenizer(question or "", add_special_tokens=False)["input_ids"][:80]
        middle = tokenizer(chunk, add_special_tokens=False)["input_ids"][:140]
        tail = tokenizer(text, add_special_tokens=False)["input_ids"][:120]
        ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id] + middle + \
              [tokenizer.sep_token_id] + tail + [tokenizer.sep_token_id]
        ids = ids[:MAX_LENGTH]
        features = segment_model.candidate_features(
            answer, mark, question, document, question_words,
            segment_model.document_token_set(document),
            [(tuple(piece[:2]), piece[2]) for piece in probabilities],
            {(mark["start"], mark["end"]): rank / max(1, total)},
        )
        rows.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "features": features,
            "label": 1.0 if mark["positive"] else 0.0,
            "type": TYPE_INDEX.get(mark["type"], 0),
            "weight": len(text) / 200.0,
            "start": mark["start"],
            "end": mark["end"],
            "pair_id": pair["pair_id"],
        })
    return rows


def collate(rows, pad_id):
    width = max(len(row["input_ids"]) for row in rows)
    ids, masks, features, labels, kinds, weights = [], [], [], [], [], []
    for row in rows:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * padding)
        masks.append(row["attention_mask"] + [0] * padding)
        features.append(row["features"])
        labels.append(row["label"])
        kinds.append(row["type"])
        weights.append(row["weight"])
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "features": torch.tensor(features, dtype=torch.float),
        "labels": torch.tensor(labels, dtype=torch.float),
        "types": torch.tensor(kinds, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float),
    }


def train_model(model, tokenizer, rows, seed, epochs=EPOCHS):
    """Обучение голов при замороженном энкодере — как вариант v2 схемы 52."""
    for parameter in model.base.bert.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.proj.parameters()) + list(model.binary.parameters()) + \
                 list(model.kind.parameters())
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01, lr=LR)
    total_steps = math.ceil(len(rows) / BATCH) * epochs
    warmup = max(1, int(total_steps * 0.1))

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    history = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(rows)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        model.train()
        total = count = 0.0
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            optimizer.zero_grad()
            for offset in range(0, len(batch), MICRO):
                part = [rows[index] for index in batch[offset:offset + MICRO]]
                piece = collate(part, tokenizer.pad_token_id)
                logits, kinds = model(piece["input_ids"], piece["attention_mask"], piece["features"])
                bce = F.binary_cross_entropy_with_logits(logits, piece["labels"])
                ce = F.cross_entropy(kinds, piece["types"])
                iou = segment_model.soft_iou_loss(logits, piece["labels"], piece["weights"])
                loss = bce + CE_WEIGHT * ce + IOU_WEIGHT * iou
                (loss * len(part) / len(batch)).backward()
                total += float(loss.detach()) * len(part)
                count += len(part)
            optimizer.step()
            scheduler.step()
        history.append({"epoch": epoch, "loss": total / max(1, count)})
    return history


def predict_pairs(model, tokenizer, pairs, questions, probabilities_of):
    """Вероятность каждого кандидата; возвращаются спаны по парам."""
    model.eval()
    result = {}
    with torch.no_grad():
        for pair in pairs:
            answer = pair["model_output"] or ""
            if not answer:
                result[pair["pair_id"]] = []
                continue
            question = questions.get(pair["pair_id"], "")
            document = pair.get("document_text") or ""
            rows = build_rows(tokenizer, pair, question,
                              probabilities_of.get(pair["pair_id"], []))
            if not rows:
                result[pair["pair_id"]] = []
                continue
            span_rows = []
            for start in range(0, len(rows), BATCH):
                part = rows[start:start + BATCH]
                piece = collate(part, tokenizer.pad_token_id)
                logits, kinds = model(piece["input_ids"], piece["attention_mask"], piece["features"])
                for row, value in zip(part, torch.sigmoid(logits).tolist()):
                    span_rows.append({"start": row["start"], "end": row["end"], "risk": value})
            result[pair["pair_id"]] = span_rows
    return result


def decide(span_rows, threshold, limit=3):
    """Решение по кандидатам: всё выше порога, не больше `limit` на пару."""
    chosen = [row for row in span_rows if row["risk"] >= threshold]
    chosen.sort(key=lambda row: -row["risk"])
    return [{"start": row["start"], "end": row["end"]} for row in chosen[:limit]]


def support_of(pairs, spans_by_id):
    """Счётчики токенов и пар в формате, совместимом с rule57.merge."""
    tp = fp = fn = negative_tokens = flagged_tokens = 0
    negative_pairs = flagged_pairs = 0
    for pair in pairs:
        answer = pair["model_output"] or ""
        gold_spans = spans_of(pair)
        gold = covered_tokens(answer, [(s["start"], s["end"]) for s in gold_spans])
        predicted = covered_tokens(answer, [(s["start"], s["end"])
                                            for s in spans_by_id.get(pair["pair_id"], [])])
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        negative_tokens += len(token_positions(answer)) - len(gold)
        flagged_tokens += len(predicted - gold)
        if not gold_spans:
            negative_pairs += 1
            if predicted:
                flagged_pairs += 1
    point = prf(tp, fp, fn)
    point["fpr_token"] = flagged_tokens / negative_tokens if negative_tokens else 0.0
    point["fpr_pair"] = flagged_pairs / negative_pairs if negative_pairs else 0.0
    return point


def pick_threshold(pairs, spans_by_id):
    """Порог: максимум token F1 при `fpr_token <= 0,40` — правило A."""
    best = None
    for threshold in THRESHOLDS:
        chosen = {pair_id: decide(rows, threshold) for pair_id, rows in spans_by_id.items()}
        point = support_of(pairs, chosen)
        if point["fpr_token"] > FPR_LIMIT:
            continue
        if best is None or (point["f1"], -threshold) > (best[1]["f1"], -best[0]):
            best = (threshold, point)
    return best


def score(pairs, spans_by_id, threshold):
    """Метрики на пороге, включая мягкий IoU (договорная метрика, п. 5)."""
    chosen = {pair_id: decide(rows, threshold) for pair_id, rows in spans_by_id.items()}
    token = support_of(pairs, chosen)
    predicted_total = sum(len(spans) for spans in chosen.values())
    gold_total = sum(len(spans_of(pair)) for pair in pairs)
    result = {
        "token": {"precision": token["precision"], "recall": token["recall"], "f1": token["f1"]},
        "fpr_token": token["fpr_token"],
        "fpr_pair": token["fpr_pair"],
        "predicted_spans": predicted_total,
        "gold_spans": gold_total,
        "threshold": threshold,
    }
    for name, bound in (("exact", 1.0), ("soft", 0.5)):
        hits = sum(match_spans(chosen[pair["pair_id"]], spans_of(pair), bound)[0] for pair in pairs)
        result[f"span_f1_{name}"] = (2 * hits / (predicted_total + gold_total)
                                     if predicted_total + gold_total else 0.0)
        result[f"span_tp_{name}"] = hits
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", required=True, type=int, choices=range(1, FOLDS + 1))
    parser.add_argument("--inner", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--prefix", default="seg63")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # детерминизм включаем мягко: строгий режим в torch 2.15 падает на
    # операциях, которых в схеме 52 не было (новые головы и soft-IoU)
    torch.use_deterministic_algorithms(True, warn_only=True)
    source = ROOT / "artifacts/ours33.jsonl"
    pairs = read_jsonl(source)
    questions_table = load_questions()
    questions = {pair["pair_id"]: question_of(pair, questions_table) for pair in pairs}

    folds = grouped_folds(pairs, FOLDS)
    outer = args.outer - 1
    inner = (outer + args.inner) % FOLDS
    if inner == outer:
        raise SystemExit("внутренний фолд совпал с внешним")
    training, tuning, held = split(folds, outer, inner)
    inner_tag = inner + 1

    path = Path(args.out) if args.out else \
        ROOT / f"artifacts/{args.prefix}_outer{args.outer}_inner{inner_tag}_seed{args.seed}.json"
    report = {
        "status": "в процессе",
        "variant": "scheme63-segment",
        "with_question": True,
        "question_input": True,
        "candidate_rule": "предложения/клаузы + объединения до 3 + весь ответ",
        "loss": {"bce": 1.0, "ce_type": CE_WEIGHT, "soft_iou": IOU_WEIGHT},
        "outer_fold": args.outer,
        "inner_fold": inner_tag,
        "seed": args.seed,
        "epochs": args.epochs,
        "git_sha": args.git_sha,
        "torch": str(torch.__version__),
        "gold_sha256": file_digest(source),
        "train_id_sha256": id_digest(training),
        "inner_id_sha256": id_digest(tuning),
        "outer_id_sha256": id_digest(held),
        "test301": False,
        "external_data": False,
    }
    save_json(path, report)

    started = time.perf_counter()
    tokenizer, base = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=args.seed)
    base.base.config._attn_implementation = "eager"
    model = segment_model.CandidateModel(base.base)
    # признаки «прежний сигнал» считаются моделью поставки без QUESTION_INPUT:
    # она уже обучена, веса зафиксированы, дообучения здесь нет
    probabilities = {}
    with torch.no_grad():
        for pair in pairs:
            scan = lexfeat.pair_probabilities(tokenizer, base, pair)
            # `pair_probabilities` отдаёт {"answer", "pieces"}; тренеру нужны pieces
            probabilities[pair["pair_id"]] = [] if scan is None else scan["pieces"]

    training_rows = []
    for pair in training:
        training_rows.extend(build_rows(tokenizer, pair, questions[pair["pair_id"]],
                                        probabilities.get(pair["pair_id"], [])))
    report["training_rows"] = len(training_rows)
    report["positive_rows"] = sum(1 for row in training_rows if row["label"] > 0.5)
    save_json(path, report)

    history = train_model(model, tokenizer, training_rows, args.seed, args.epochs)
    report["history"] = history

    inner_spans = predict_pairs(model, tokenizer, tuning, questions, probabilities)
    outer_spans = predict_pairs(model, tokenizer, held, questions, probabilities)
    picked = pick_threshold(tuning, inner_spans)
    report["threshold_point"] = picked[1] if picked else None
    report["threshold"] = picked[0] if picked else None
    if picked is None:
        report["status"] = "готово: нет допустимого порога"
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(path, report)
        print("нет допустимого порога", flush=True)
        return

    report["outer_at_threshold"] = score(held, outer_spans, picked[0])
    report["outer_candidates"] = {
        pair_id: [{"start": row["start"], "end": row["end"], "risk": row["risk"]}
                  for row in rows]
        for pair_id, rows in outer_spans.items()
    }
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    save_json(path, report)
    point = report["outer_at_threshold"]
    print(f"outer {args.outer} seed {args.seed}: порог {picked[0]} | token F1 {point['token']['f1']:.4f} | "
          f"soft span {point['span_f1_soft']:.4f} | exact {point['span_f1_exact']:.4f} | "
          f"fpr {point['fpr_token']:.4f} | {report['elapsed_s']} с", flush=True)


if __name__ == "__main__":
    main()
