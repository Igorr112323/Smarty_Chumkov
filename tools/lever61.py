"""ПРОМПТ 61: множитель гистерезиса как рычаг. Локально, по сохранённым piece.

Вопрос, на который отвечает этот расчёт: **есть ли из чего выбирать.** При
`m = 0,5` (текущий A) прогонов на пару 0 или 1, и выбор спана невозможен —
это проверено на 699 парах. Соседнее значение `m = 0,75` по промпту 57 роняло
token F1 всего на 0,0011, то есть почти бесплатно. Здесь мерим, покупает ли
оно кандидатов.

Что считается (стадия 1 промпта 61):

1.1 распределение числа прогонов на пару для m ∈ {0,5, 0,6, 0,75};
1.2 token F1, precision, recall, FPR токенный при пороге, выбранном заново на
    внутренней сетке для каждого m;
1.3 exact span F1, pair F1;
1.4 фаза 2: выбор спана по лексической релевантности вопросу против выбора по
    вероятности — на парах, где прогонов больше одного;
1.5 сколько пар с 2+ прогонами, где gold-спаны попадают в разные прогоны: это
    потолок выигрыша от любого правила выбора.

Порог для каждого m выбирается на ВНУТРЕННЕЙ сетке (`inner_grid` отчёта A даёт
готовые счётчики только для m = 0,5, поэтому для остальных считаем сами по
внутренним парам, если их piece сохранены; иначе честно помечаем пробел).

Никакого обучения, никаких заданий.
"""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from spanver.snap import boundaries, snapped
from tools.cascade59 import relevance, share_of_question
from tools.metrics59 import covered_tokens, match_spans, prf, spans_of
from tools.rule57 import GATE, _runs

MULTIPLIERS = (0.5, 0.6, 0.75, 0.9)
START_GRID = [round(0.30 + 0.01 * i, 2) for i in range(51)]  # 0.30 .. 0.80
FPR_LIMIT = 0.40
MIN_CHARS = 2


def flags_with_multiplier(pieces, start, multiplier):
    """Флаги субтокенов: вход по `start`, удержание по `start * multiplier`."""
    keep = start * multiplier
    flags = []
    active = False
    for _, _, score in pieces:
        active = score >= (keep if active else start)
        flags.append(active)
    return flags


def runs_of(answer, pieces, start, multiplier, gate=GATE):
    """Спаны пары при данном множителе, с обрезкой до границ слов.

    Гейт 0,45 — часть пути A: если максимум вероятности по паре ниже него,
    пара не помечается вовсе. Без гейта метрики сдвигаются.
    """
    if not pieces:
        return []
    if gate is not None and max(piece[2] for piece in pieces) < gate:
        return []
    flags = flags_with_multiplier(pieces, start, multiplier)
    spans = _runs(answer, pieces, flags)
    spans = snapped(spans, answer, "expand")
    result = []
    for span in spans:
        if span["end"] - span["start"] + 1 < MIN_CHARS:
            continue
        covered = [piece[2] for piece in pieces
                   if min(span["end"], piece[1]) - max(span["start"], piece[0]) >= 0]
        span = dict(span)
        span["risk"] = max(covered) if covered else 0.0
        result.append(span)
    return result


def pick_by_risk(spans):
    """Прежний выбор A: спан с максимальной вероятностью."""
    if not spans:
        return None
    return max(spans, key=lambda span: (span["risk"], span["start"], span["end"]))


def pick_by_question(spans, answer, question):
    """Фаза 2: выбор по лексической релевантности, без модели и векторов."""
    if not spans:
        return None
    best = None
    for span in spans:
        value = share_of_question(question, answer[span["start"]:span["end"] + 1])
        if best is None or value > best[0]:
            best = (value, span)
    return best[1]


def evaluate(pairs, predictions, types=None):
    """Метрики варианта предсказаний — тем же путём, что свод промпта 59.

    Раньше здесь были собственные определения FPR и pair F1: FPR расходился
    (0,3437 против 0,3787), pair F1 тоже (0,2676 против 0,3032). Теперь метрики
    берутся из `summarize59.counts_for` и `metrics59.score_pairs`, чтобы таблица
    сравнивалась с A по одному определению.
    """
    from tools.metrics59 import score_pairs
    from tools.summarize59 import counts_for

    token = counts_for(pairs, predictions, types)
    point = score_pairs(pairs, predictions, types)
    return {
        "pairs": len(pairs),
        "token": {"precision": token["precision"], "recall": token["recall"],
                  "f1": token["f1"]},
        "fpr_token": token["fpr_token"],
        "span_f1": point["exact_span_f1"],
        "pair_f1": point["pair_f1"],
        "fpr_pair": point["fpr_pair"],
    }


def gold_runs_split(pair, pieces, start, multiplier):
    """Сколько нужно прогонов, чтобы покрыть все gold-спаны пары.

    Критерий 1.5: если gold-спаны попадают в РАЗНЫЕ прогоны, выбор одного из
    них теряет фрагмент, и любое правило выбора может его вернуть. Если все
    gold в одном прогоне, выбор не нужен.
    """
    spans = runs_of(pair["model_output"], pieces, start, multiplier)
    if len(spans) < 2:
        return False, len(spans)
    gold = [(s["start"], s["end"]) for s in spans_of(pair)]
    if not gold:
        return False, len(spans)
    hit_runs = set()
    for g_start, g_end in gold:
        for index, span in enumerate(spans):
            overlap = min(g_end, span["end"]) - max(g_start, span["start"]) + 1
            if overlap > 0:
                hit_runs.add(index)
    return len(hit_runs) > 1, len(spans)


def choose_threshold(inner_pairs, inner_preds, multiplier):
    """Порог на внутренней сетке: максимум token F1 при FPR <= 0,40."""
    best = None
    for start in START_GRID:
        predictions = {}
        for pair in inner_pairs:
            pieces = inner_preds.get(pair["pair_id"])
            if not pieces:
                continue
            spans = runs_of(pair["model_output"], pieces, start, multiplier)
            chosen = pick_by_risk(spans)
            predictions[pair["pair_id"]] = [] if chosen is None else [chosen]
        result = evaluate(inner_pairs, predictions)
        if result["fpr_token"] > FPR_LIMIT:
            continue
        if best is None or result["token"]["f1"] > best[1]["token"]["f1"]:
            best = (start, result)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--out", default="artifacts/lever61.json")
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    questions = load_questions()
    by_id = {pair["pair_id"]: pair for pair in pairs}
    reports = sorted(Path("artifacts").glob("q59_a_outer*_inner*.json"))
    # канонический маппинг свода: inner = outer % 10 + 1 (у части отчётов поле
    # primary выставлено неверно, поэтому берём по имени, а не по флагу)
    all_reports = {}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        all_reports[(report["outer_fold"], report["inner_fold"])] = report
    from tools.summarize59 import thresholds_for

    canon = thresholds_for(all_reports)  # порог по четырём внутренним сеткам
    primary = {}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        outer = report["outer_fold"]
        if report["inner_fold"] == outer % 10 + 1 and outer in canon:
            primary[outer] = (report, path.name)
    print(f"отчётов A: {len(reports)}, фолдов по каноническому маппингу: {len(primary)}")
    missing = [o for o in range(1, 11) if o not in primary]
    if missing:
        raise SystemExit(f"нет отчётов для фолдов: {missing}")

    table = {}
    for multiplier in MULTIPLIERS:
        counts = Counter()
        pairs_with_gold_split = 0
        pairs_multi = 0
        chosen_risk = {}
        chosen_question = {}
        per_fold = {}
        for outer, (report, name) in sorted(primary.items()):
            start = canon[outer]
            fold_pairs = [by_id[pid] for pid in report["outer_predictions"] if pid in by_id]
            preds = {pid: [tuple(x) for x in pieces]
                     for pid, pieces in report["outer_predictions"].items() if pid in by_id}
            for pair in fold_pairs:
                pieces = preds[pair["pair_id"]]
                spans = runs_of(pair["model_output"], pieces, start, multiplier)
                counts[len(spans)] += 1
                if len(spans) >= 2:
                    pairs_multi += 1
                    split, _ = gold_runs_split(pair, pieces, start, multiplier)
                    if split:
                        pairs_with_gold_split += 1
                by_risk = pick_by_risk(spans)
                text = question_of(pair, questions)
                by_question = pick_by_question(spans, pair["model_output"], text)
                chosen_risk[pair["pair_id"]] = [] if by_risk is None else [by_risk]
                chosen_question[pair["pair_id"]] = [] if by_question is None else [by_question]
            per_fold[outer] = start

        total = sum(counts.values())
        result = {
            "multiplier": multiplier,
            "runs_distribution": {str(k): v for k, v in sorted(counts.items())},
            "share_zero": counts[0] / total if total else 0.0,
            "share_one": counts[1] / total if total else 0.0,
            "share_two_plus": sum(v for k, v in counts.items() if k >= 2) / total if total else 0.0,
            "pairs_two_plus": pairs_multi,
            "pairs_gold_split": pairs_with_gold_split,
            "thresholds": per_fold,
        }
        all_pairs = [by_id[pid] for report, _ in primary.values()
                     for pid in report["outer_predictions"] if pid in by_id]
        result["risk_choice"] = evaluate(all_pairs, chosen_risk)
        result["question_choice"] = evaluate(all_pairs, chosen_question)
        table[f"{multiplier}"] = result
        print(f"m={multiplier}: прогонов {dict(sorted(counts.items()))} | "
              f"2+ {pairs_multi} | gold разложен {pairs_with_gold_split} | "
              f"token F1 {result['risk_choice']['token']['f1']:.4f} | "
              f"exact {result['risk_choice']['span_f1']:.4f}")

    payload = {"multipliers": list(MULTIPLIERS), "fpr_limit": FPR_LIMIT, "table": table}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nзаписано: {args.out}")


def load_questions():
    """Вопросы из исходных CSV — тот же путь, что в варианте B (fewshot56)."""
    from tools.fewshot56 import load_questions as loader

    return loader()


def question_of(pair, table=None):
    """Вопрос пары: `query_text` исходного CSV по индексу из pair_id.

    `fewshot56.question_for` режет `pair_id[4:]`, что верно для `sber000073`,
    но для `rusci000310` даёт `i000310`. Здесь индекс берётся как хвостовые
    цифры, а префикс — по началу идентификатора.
    """
    if not table:
        return ""
    pair_id = pair["pair_id"]
    prefix = "sber" if pair_id.startswith("sber") else "rusci"
    digits = "".join(ch for ch in pair_id[len(prefix):] if ch.isdigit())
    rows = table.get(prefix) or []
    if not digits:
        return ""
    index = int(digits)
    return (rows[index] if index < len(rows) else "") or ""


if __name__ == "__main__":
    main()
