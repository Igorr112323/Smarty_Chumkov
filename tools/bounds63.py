"""ПРОМПТ 63: границы вместо отбора — предложение с пиком вероятности.

Гипотеза задания: точная локализация упирается не в порог и не в отбор, а в
разрешение — предсказанный спан медианой 490–522 знака не может совпасть с
золотым 85-знаковым. Рычаг — не выбор из кандидатов, а обрезка области выше
порога до границ предложения.

Что проверяется (всё локально, 0 заданий, обучение не нужно):

5.1 предсказание = предложение, содержащее максимум вероятности, если максимум
    выше порога; иначе пара не помечается;
5.2 то же, но все предложения с пиком выше порога (не больше трёх на пару);
5.3 сравнение с A: exact span F1 ≥ 0,30 при fpr_token ≤ 0,40 и token F1 ≥ 0,49;
5.4 медиана длины предсказания в диапазоне 120–250 знаков.

Никаких новых параметров: порог прежний (пофолдовый из внутренней сетки),
`m = 0,5`, гистерезис не трогается. Меняется только способ вырезать фрагмент.

Ключевое обоснование, проверенное до расчёта: 74,0 % золотых спанов (222 из
300) точно совпадают с границами предложений — 36,7 % ровно с одним
предложением, 37,3 % с объединением соседних. Медиана длины предложения 103,5
знака, что попадает в целевой диапазон 120–250.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from spanver.sentence_snap import sentence_intervals
from tools.lever61 import runs_of
from tools.metrics59 import score_pairs, spans_of
from tools.summarize59 import counts_for, thresholds_for

MAX_SENTENCES = 3  # потолок из задания, п. 5.2


def peak_span(answer, pieces, start, multiplier=0.5):
    """Область выше порога путём A — источник пика вероятности."""
    spans = runs_of(answer, pieces, start, multiplier)
    if not spans:
        return None
    return max(spans, key=lambda span: (span["risk"], span["start"], span["end"]))


def sentence_with_peak(answer, pieces, start):
    """5.1: предложение, накрывающее максимум вероятности по всем субтокенам."""
    if not pieces:
        return []
    peak_piece = max(pieces, key=lambda piece: piece[2])
    if peak_piece[2] < start:
        return []
    peak_start = peak_piece[0]
    for left, right in sentence_intervals(answer):
        if left <= peak_start <= right:
            return [{"start": left, "end": right}]
    return []


def sentences_above_threshold(answer, pieces, start, limit=MAX_SENTENCES):
    """5.2: все предложения, где есть субтокен с вероятностью >= порога."""
    if not pieces:
        return []
    hot = [piece[0] for piece in pieces if piece[2] >= start]
    if not hot:
        return []
    chosen = []
    for left, right in sentence_intervals(answer):
        if any(left <= position <= right for position in hot):
            chosen.append({"start": left, "end": right})
            if len(chosen) >= limit:
                break
    return chosen


def predict(pair, pieces, start, mode):
    answer = pair["model_output"] or ""
    tuples = [tuple(piece) for piece in pieces]
    if mode == "sentence":
        return sentence_with_peak(answer, tuples, start)
    if mode == "sentences":
        return sentences_above_threshold(answer, tuples, start)
    raise ValueError(mode)


def evaluate_mode(reports, pairs_by_id, thresholds, mode):
    """Метрики режима по всем фолдам: pooled + распределение длин."""
    pairs = []
    predicted = {}
    for outer in sorted(thresholds):
        report = reports[(outer, outer % 10 + 1)]
        for pair_id, pieces in report["outer_predictions"].items():
            pair = pairs_by_id.get(pair_id)
            if pair is None:
                continue
            pairs.append(pair)
            predicted[pair_id] = predict(pair, pieces, thresholds[outer], mode)
    token = counts_for(pairs, predicted, None)
    point = score_pairs(pairs, predicted)
    lengths = sorted(span["end"] - span["start"] + 1
                     for spans in predicted.values() for span in spans)
    per_pair = [len(spans) for spans in predicted.values()]
    return {
        "mode": mode,
        "pairs": len(pairs),
        "token_precision": token["precision"],
        "token_recall": token["recall"],
        "token_f1": token["f1"],
        "fpr_token": token["fpr_token"],
        "exact_span_f1": point["exact_span_f1"],
        "pair_f1": point["pair_f1"],
        "fpr_pair": point["fpr_pair"],
        "marked_pairs": sum(1 for count in per_pair if count),
        "spans_total": len(lengths),
        "spans_per_marked": (len(lengths) / sum(1 for count in per_pair if count)
                             if any(per_pair) else 0.0),
        "length_median": statistics.median(lengths) if lengths else 0,
        "length_p90": lengths[int(0.9 * len(lengths))] if lengths else 0,
        "length_share_120_250": (sum(1 for x in lengths if 120 <= x <= 250) / len(lengths)
                                 if lengths else 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--out", default="artifacts/bounds63.json")
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    pairs_by_id = {pair["pair_id"]: pair for pair in pairs}
    reports = {}
    for path in sorted(Path("artifacts").glob("q59_a_outer*_inner*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[(report["outer_fold"], report["inner_fold"])] = report
    thresholds = thresholds_for(reports)
    print(f"отчётов {len(reports)}, пороги {[thresholds[o] for o in sorted(thresholds)]}")

    table = {"thresholds": thresholds, "modes": {}}
    # варианты «сигнала» для того же правила обрезки: какой сигнал указывает на
    # предложение. Добавлены, чтобы отличить «правило плохое» от «сигнал не находит».
    for mode in ("sentence", "sentences"):
        row = evaluate_mode(reports, pairs_by_id, thresholds, mode)
        table["modes"][mode] = row
        print(f"\n{mode}: token F1 {row['token_f1']:.4f} (P {row['token_precision']:.4f} "
              f"R {row['token_recall']:.4f}) | exact {row['exact_span_f1']:.4f} | "
              f"pair F1 {row['pair_f1']:.4f} | fpr {row['fpr_token']:.4f}/{row['fpr_pair']:.4f} | "
              f"длина med {row['length_median']:.0f} p90 {row['length_p90']} | "
              f"помечено {row['marked_pairs']} | спанов {row['spans_total']}")

    # опорная точка A на тех же порогах — для сравнения в одной таблице
    base_pairs, base_pred = [], {}
    for outer in sorted(thresholds):
        report = reports[(outer, outer % 10 + 1)]
        for pair_id, pieces in report["outer_predictions"].items():
            pair = pairs_by_id.get(pair_id)
            if pair is None:
                continue
            base_pairs.append(pair)
            span = peak_span(pair["model_output"], [tuple(p) for p in pieces], thresholds[outer])
            base_pred[pair_id] = [] if span is None else [span]
    token = counts_for(base_pairs, base_pred, None)
    point = score_pairs(base_pairs, base_pred)
    lengths = sorted(span["end"] - span["start"] + 1 for spans in base_pred.values() for span in spans)
    table["baseline_a"] = {
        "token_f1": token["f1"], "fpr_token": token["fpr_token"],
        "exact_span_f1": point["exact_span_f1"], "pair_f1": point["pair_f1"],
        "length_median": statistics.median(lengths) if lengths else 0,
    }
    print(f"\nA на тех же порогах: token F1 {token['f1']:.4f} | exact {point['exact_span_f1']:.4f} | "
          f"fpr {token['fpr_token']:.4f} | длина med {table['baseline_a']['length_median']:.0f}")

    # критерий 5.3 / 5.4
    checks = {}
    for mode, row in table["modes"].items():
        checks[mode] = {
            "5.3_exact_ge_0_30": row["exact_span_f1"] >= 0.30,
            "5.3_fpr_le_0_40": row["fpr_token"] <= 0.40,
            "5.3_token_ge_0_49": row["token_f1"] >= 0.49,
            "5.4_length_120_250": 120 <= row["length_median"] <= 250,
        }
        checks[mode]["all"] = all(checks[mode].values())
    table["checks"] = checks
    for mode, result in checks.items():
        print(f"{mode}: {result}")

    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nзаписано: {args.out}")


if __name__ == "__main__":
    main()
