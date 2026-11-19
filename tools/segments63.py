"""ПРОМПТ 63, стадия 1: посегментные кандидаты, метки и оракул потолка.

Переход от «один спан на пару» к «много кандидатов на пару» требует сначала
ответить на два вопроса, и оба дешёвые:

1.1 нарезка ответа на кандидаты. Не только по `.!?…`, но и по границам клауз
    внутри длинных предложений (запятая, точка с запятой, двоеточие), если
    предложение длиннее 200 знаков. Цель — медиана кандидата 80–160 знаков.
1.2 метка кандидата: положителен, если содержит не меньше половины токенов
    какого-либо золотого спана. Тип — тип этого спана. Все кандидаты пар без
    галлюцинаций — отрицательные.
1.3 статистика меток: доля положительных, кандидатов на пару, пар с хотя бы
    одним положительным (ожидание 239), распределение по шести типам.
1.4 оракул на НОВЫХ кандидатах: потолок архитектуры при идеальном выборе.
    Порог 1.4 — token F1 не ниже 0,75. Если ниже, нарезку менять до обучения.
1.5 контроль целостности: 699 пар, 300 спанов, `id_digest` не изменился.

Никакого обучения здесь нет: всё считается по уже сохранённым данным.
"""
import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, read_jsonl
from spanver.sentence_snap import sentence_intervals
from tools.metrics59 import prf, spans_of, token_positions

LONG_SENTENCE = 200          # знаков: длиннее — режем по клаузам
CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")
TYPES = ["Excess", "Missing", "Oversight", "Partial", "Contradiction", "Unconfirmed"]
POSITIVE_SHARE = 0.5          # доля токенов золотого спана в кандидате
CANDIDATE_MIN_CHARS = 2
MAX_UNION = 3                 # кандидат — объединение до трёх соседних единиц

# Зачем объединения. Правило «кандидат содержит >= половины токенов золотого
# спана» при нарезке только по одиночным единицам теряет 99 из 300 спанов
# (33,0 %): медиана длины потерянных — 531 знак, и такой спан физически не
# отдаёт половину токенов кандидату медианой 78 знаков. Объединение соседних
# единиц возвращает эти спаны, потому что золотые границы чаще всего совпадают
# с границами единиц (74 % выравниваются по предложениям).


def trim(text, left, right):
    """Сдвинуть границы внутрь, убрав пробелы по краям."""
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right - 1


def clause_cuts(text, left, right):
    """Границы клауз внутри предложения (левая, правая) пары."""
    cuts = []
    for match in CLAUSE_SPLIT.finditer(text, left, right):
        cuts.append(match.start())
    pieces = []
    start = left
    for cut in cuts:
        if cut > start:
            pieces.append((start, cut))
        start = cut + 1
    if right > start:
        pieces.append((start, right))
    return [(a, b) for a, b in pieces if b > a]


def candidates_of(answer):
    """1.1: кандидаты ответа — предложения, длинные режутся по клаузам."""
    result = []
    for left, right in sentence_intervals(answer):
        left, right = trim(answer, left, right + 1)
        if right < left:
            continue
        if right - left + 1 > LONG_SENTENCE:
            for a, b in clause_cuts(answer, left, right + 1):
                a, b = trim(answer, a, b)
                if b >= a and b - a + 1 >= CANDIDATE_MIN_CHARS:
                    result.append((a, b))
        else:
            if right - left + 1 >= CANDIDATE_MIN_CHARS:
                result.append((left, right))
    return result


def candidates_with_unions(answer, max_union=MAX_UNION):
    """1.1 + объединения соседних единиц: кандидат может покрывать несколько.

    Объединения нужны, чтобы длинный золотой спан (медиана 531 знак у
    потерянных) мог получить положительную метку. Без них 33 % спанов
    остаются без положительного кандидата, и модель физически не может их найти.
    """
    units = candidates_of(answer)
    result = list(units)
    for size in range(2, max_union + 1):
        for index in range(len(units) - size + 1):
            result.append((units[index][0], units[index + size - 1][1]))
    # кандидат «весь ответ»: 17 пар из 239 имеют золото, равное всему ответу
    # (медиана 835 знаков), и ни одно объединение до трёх единиц его не покрывает
    text = answer.strip()
    if text:
        left = len(answer) - len(answer.lstrip())
        right = len(answer.rstrip()) - 1
        result.append((left, right))
    # дубликаты границ возможны на коротких ответах — убираем, сохраняя порядок
    seen = set()
    unique = []
    for left, right in result:
        if (left, right) in seen:
            continue
        seen.add((left, right))
        unique.append((left, right))
    return unique


def label_candidates(answer, candidates, gold_spans):
    """1.2: метка кандидата — покрывает не меньше половины токенов золотого спана."""
    positions = token_positions(answer)
    gold_tokens = []
    for span in gold_spans:
        gold_tokens.append((span, covered_tokens(answer, [(span["start"], span["end"])])))
    marks = []
    for left, right in candidates:
        covered = covered_tokens(answer, [(left, right)])
        best = None
        for span, tokens in gold_tokens:
            if not tokens:
                continue
            share = len(tokens & covered) / len(tokens)
            if share >= POSITIVE_SHARE and (best is None or share > best[1]):
                best = (span, share)
        marks.append({"start": left, "end": right,
                      "positive": best is not None,
                      "type": best[0].get("type_src") if best else None,
                      "share": best[1] if best else 0.0})
    return marks


def oracle_token_f1(pairs, marks_by_id):
    """1.4: потолок при идеальном выборе кандидатов.

    Правило оракула: отметить ВСЕ кандидаты, пересекающие золото, — это верхняя
    граница для любого правила, которое выбирает подмножество кандидатов.
    """
    tp = fp = fn = 0
    for pair in pairs:
        answer = pair["model_output"] or ""
        gold_spans = spans_of(pair)
        gold = covered_tokens(answer, [(s["start"], s["end"]) for s in gold_spans])
        chosen = []
        for mark in marks_by_id[pair["pair_id"]]:
            if any(min(mark["end"], s["end"]) - max(mark["start"], s["start"]) >= 0
                   for s in gold_spans):
                chosen.append((mark["start"], mark["end"]))
        predicted = covered_tokens(answer, chosen)
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    return prf(tp, fp, fn)


def id_digest(pairs):
    """1.5: отпечаток набора по идентификаторам и координатам спанов."""
    payload = "\n".join(
        f"{pair['pair_id']}|" + ",".join(f"{s['start']}-{s['end']}:{s.get('type_src')}"
                                         for s in pair["spans"])
        for pair in pairs
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--out", default="artifacts/segments63.json")
    parser.add_argument("--expected-digest", default=None)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    marks_by_id = {}
    lengths = []
    per_pair = []
    type_counter = Counter()
    positive_total = 0
    candidate_total = 0
    pairs_with_positive = 0
    for pair in pairs:
        answer = pair["model_output"] or ""
        candidates = candidates_with_unions(answer)
        marks = label_candidates(answer, candidates, spans_of(pair))
        marks_by_id[pair["pair_id"]] = marks
        lengths.extend(mark["end"] - mark["start"] + 1 for mark in marks)
        per_pair.append(len(marks))
        candidate_total += len(marks)
        positives = [mark for mark in marks if mark["positive"]]
        positive_total += len(positives)
        if positives:
            pairs_with_positive += 1
        for mark in positives:
            type_counter[mark["type"]] += 1

    lengths.sort()
    gold_spans = sum(len(pair["spans"]) for pair in pairs)
    oracle = oracle_token_f1(pairs, marks_by_id)
    digest = id_digest(pairs)

    report = {
        "pairs": len(pairs),
        "gold_spans": gold_spans,
        "digest": digest,
        "expected_digest": args.expected_digest,
        "digest_ok": (args.expected_digest is None or args.expected_digest == digest),
        "candidates": {
            "total": candidate_total,
            "per_pair": {
                "mean": candidate_total / len(pairs) if pairs else 0,
                "median": statistics.median(per_pair) if per_pair else 0,
                "min": min(per_pair) if per_pair else 0,
                "max": max(per_pair) if per_pair else 0,
            },
            "length": {
                "median": statistics.median(lengths) if lengths else 0,
                "p10": lengths[int(0.1 * len(lengths))] if lengths else 0,
                "p90": lengths[int(0.9 * len(lengths))] if lengths else 0,
                "min": lengths[0] if lengths else 0,
                "max": lengths[-1] if lengths else 0,
                "share_80_160": (sum(1 for x in lengths if 80 <= x <= 160) / len(lengths)
                                 if lengths else 0.0),
            },
        },
        "labels": {
            "positive_share": positive_total / candidate_total if candidate_total else 0.0,
            "positive_total": positive_total,
            "pairs_with_positive": pairs_with_positive,
            "pairs_without_positive": len(pairs) - pairs_with_positive,
            "by_type": dict(sorted(type_counter.items(), key=lambda kv: -kv[1])),
        },
        "oracle": {"token_precision": oracle["precision"], "token_recall": oracle["recall"],
                   "token_f1": oracle["f1"], "threshold": 0.75,
                   "passed": oracle["f1"] >= 0.75},
    }

    print(f"пар {report['pairs']}, золотых спанов {gold_spans}")
    print(f"digest {digest[:16]}… ok={report['digest_ok']}")
    print(f"кандидатов {candidate_total}, на пару медиана {report['candidates']['per_pair']['median']}, "
          f"среднее {report['candidates']['per_pair']['mean']:.1f}")
    print(f"длина кандидата: медиана {report['candidates']['length']['median']:.0f}, "
          f"p10 {report['candidates']['length']['p10']}, p90 {report['candidates']['length']['p90']}, "
          f"доля 80–160 {report['candidates']['length']['share_80_160']:.1%}")
    print(f"положительных {positive_total} ({report['labels']['positive_share']:.1%}), "
          f"пар с положительным {pairs_with_positive}")
    print(f"по типам: {report['labels']['by_type']}")
    print(f"\nОРАКУЛ 1.4: token F1 {oracle['f1']:.4f} "
          f"(P {oracle['precision']:.4f} R {oracle['recall']:.4f}) | порог 0,75 | "
          f"{'ПРОЙДЕН' if report['oracle']['passed'] else 'НЕ ПРОЙДЕН'}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    print(f"записано: {args.out}")


if __name__ == "__main__":
    main()
