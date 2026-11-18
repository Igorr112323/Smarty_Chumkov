"""ПРОМПТ 59: метрики прогона с вопросом на входе.

Считаются ровно те величины, что зафиксированы в `artifacts/protocol59.md`, и
ничего сверх них:

* **полная метка** (все шесть типов) — основная метрика, п. 0.1;
* **документные типы** (Contradiction + Unconfirmed + Excess) — вторая строка,
  п. 0.2, считается всегда и приводится рядом;
* **обе базовые линии** каждого набора, п. 0.4.

Почему отдельный модуль. `spanver/evaluate.metrics` считает token F1 по всем
меткам сразу и не умеет ни сузить метку до подмножества типов, ни посчитать
базовую линию тем же путём. Здесь оба уровня считаются одной и той же функцией
`token_counts` через `covered_tokens`/`token_positions` — так базовая линия и
цифра модели сопоставимы по построению, а не по обещанию.

Порог, декодер и правило отбора сюда не входят: это дело `train52_chunks.py`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, match_spans, token_positions

# 0.2: подмножество, сопоставимое с RAGTruth/LettuceDetect
DOCUMENT_TYPES = ("Contradiction", "Unconfirmed", "Excess")
# 0.1: всё остальное — метка, определимая только относительно вопроса
QUESTION_TYPES = ("Partial", "Missing", "Oversight")


def spans_of(pair, types=None):
    """Спаны пары, при необходимости только указанных типов."""
    spans = pair.get("spans") or []
    if types is None:
        return spans
    return [span for span in spans if span.get("type_src") in types]


def token_counts(answer, gold_intervals, predicted_intervals):
    """tp/fp/fn по токенам — единственный путь подсчёта в этом модуле."""
    gold = covered_tokens(answer, gold_intervals)
    predicted = covered_tokens(answer, predicted_intervals)
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def pooled(rows, key):
    """Pooled по парам: объединённые счётчики, а не среднее средних."""
    tp = sum(row["levels"][key]["tp"] for row in rows)
    fp = sum(row["levels"][key]["fp"] for row in rows)
    fn = sum(row["levels"][key]["fn"] for row in rows)
    return prf(tp, fp, fn)


def baseline_all_positive(pairs, types=None):
    """Базовая линия «all-positive»: все пары помечены как содержащие ошибку.

    Токенная часть при этом вырождается: правило «пометить всё» покрывает все
    токены ответа, поэтому token F1 = 2*|gold| / (|all| + |gold|), где |all| —
    все токены ответа. Считается тем же `covered_tokens`, что и модель.
    """
    tp = fp = fn = 0
    for pair in pairs:
        answer = pair.get("model_output") or ""
        positions = [(start, end) for start, end in token_positions(answer)]
        gold = covered_tokens(answer, [(s["start"], s["end"]) for s in spans_of(pair, types)])
        predicted = covered_tokens(answer, positions)
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    return prf(tp, fp, fn)


def score_pairs(pairs, predicted_by_id, types=None):
    """Метрики одного набора пар: по токенам и по парам."""
    rows = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        gold_spans = spans_of(pair, types)
        predicted = predicted_by_id.get(pair["pair_id"]) or []
        gold_intervals = [(s["start"], s["end"]) for s in gold_spans]
        predicted_intervals = [(s["start"], s["end"]) for s in predicted]
        token = prf(*token_counts(answer, gold_intervals, predicted_intervals))
        matched = match_spans(predicted, gold_spans, 1.0)[0]  # exact span F1
        rows.append({
            "pair_id": pair["pair_id"],
            "levels": {"token": token},
            "gold_spans": len(gold_spans),
            "predicted_spans": len(predicted),
            "exact_tp": matched,
        })
    token = pooled(rows, "token")
    predicted_total = sum(row["predicted_spans"] for row in rows)
    gold_total = sum(row["gold_spans"] for row in rows)
    exact_tp = sum(row["exact_tp"] for row in rows)
    with_gold = [row for row in rows if row["gold_spans"]]
    pair_tp = sum(1 for row in with_gold if row["exact_tp"])
    negatives = [row for row in rows if not row["gold_spans"]]
    flagged = sum(1 for row in negatives if row["predicted_spans"])
    return {
        "pairs": len(rows),
        "with_gold": len(with_gold),
        "token": token,
        "exact_span_f1": 2 * exact_tp / (predicted_total + gold_total) if predicted_total + gold_total else 0.0,
        "pair_f1": 2 * pair_tp / (len(with_gold) + (flagged + pair_tp)) if with_gold else 0.0,
        "member_f1": pair_tp / len(with_gold) if with_gold else 0.0,
        "fpr_pair": flagged / len(negatives) if negatives else 0.0,
    }


def evaluate(pairs, predicted_by_id, baseline_by_id=None):
    """Обе строки сразу: полная метка и документное подмножество.

    `baseline_by_id` (необязательно) — предсказания «all-positive» уже в виде
    спанов; по умолчанию базовая линия считается по всем токенам ответа.
    """
    result = {"pairs": len(pairs)}
    for name, types in (("full", None), ("document", DOCUMENT_TYPES), ("question", QUESTION_TYPES)):
        result[name] = score_pairs(pairs, predicted_by_id, types)
        result[name]["baseline"] = baseline_all_positive(pairs, types)
    if baseline_by_id is not None:
        result["baseline_rule"] = score_pairs(pairs, baseline_by_id)
    return result


def margin(model_f1, baseline_f1):
    """Отрыв модели от базовой линии того же набора."""
    return model_f1 - baseline_f1


def bootstrap_ci(pairs, predicted_by_id, key="full", rounds=10000, seed=59):
    """CI 95 % по парам: пересэмплирование пар с возвращением.

    Возвращается интервал для token F1 уровня `key`. Нужен, чтобы утверждение
    «отрыв больше нуля» опиралось на интервал, а не на точечное сравнение.
    """
    import random

    generator = random.Random(seed)
    counts = []
    prepared = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        types = {"full": None, "document": DOCUMENT_TYPES, "question": QUESTION_TYPES}[key]
        gold_intervals = [(s["start"], s["end"]) for s in spans_of(pair, types)]
        predicted = predicted_by_id.get(pair["pair_id"]) or []
        predicted_intervals = [(s["start"], s["end"]) for s in predicted]
        prepared.append(token_counts(answer, gold_intervals, predicted_intervals))
    size = len(prepared)
    for _ in range(rounds):
        tp = fp = fn = 0
        for index in range(size):
            a, b, c = prepared[generator.randrange(size)]
            tp += a
            fp += b
            fn += c
        counts.append(prf(tp, fp, fn)["f1"])
    counts.sort()
    low = counts[int(0.025 * rounds)]
    high = counts[min(rounds - 1, int(0.975 * rounds))]
    return {"low": low, "high": high, "rounds": rounds, "seed": seed, "key": key}
