"""ПРОМПТ 62: гейт как рычаг — проверка режима 0,45 локально, по 40 отчётам A.

Проверяются три объяснения находки «порог 0,45 даёт F1 0,644»:

1. это рабочий режим, а текущий 0,47–0,52 выбран неверно;
2. precision надута: гейт 0,45 пропускает почти всё, спаны длинные, перекрытие
   с золотом механическое;
3. три (на деле пять) отчёта нерепрезентативны — проверить на всех 40.

Что считается для каждого порога:

* token P/R/F1 — путём `summarize59.counts_for`, как в A;
* exact span F1, pair F1, FPR пары — `metrics59.score_pairs`;
* FPR токенный — `counts_for`;
* медиана и p90 длины предсказанного спана против золотой медианы 157 знаков;
* число помеченных пар.

Отчётность ведётся в двух разрезах, и это важно:

* `canon` — по одному отчёту на фолд (699 пар, честный набор без повторов);
* `all40` — все 40 отчётов как есть (каждая пара до четырёх раз, с piece от
  разных внутренних фолдов): так считались «351 пара» в задании, поэтому разрез
  нужен для сверки, но сравнивать с A можно только `canon`.

Порог при этом НЕ выбирается по внешним фолдам: здесь он задаётся вручную,
чтобы посмотреть таблицу. Настоящий выбор — `summarize59.pick_threshold` по
четырём внутренним сеткам с ограничением `fpr_token <= 0,40`; он и печатается
отдельной строкой как единственный законный.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.lever61 import pick_by_risk, runs_of
from tools.metrics59 import score_pairs
from tools.summarize59 import counts_for, thresholds_for

GATES = [round(0.40 + 0.01 * i, 2) for i in range(13)]  # 0.40 .. 0.52
FPR_LIMIT = 0.40
GOLD_MEDIAN = 157  # медиана длины золотого спана по разбору промпта 59


def load(paths):
    out = {}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        out[(report["outer_fold"], report["inner_fold"])] = report
    return out


def predict(report, pairs_by_id, start):
    result = {}
    for pair_id, pieces in report["outer_predictions"].items():
        pair = pairs_by_id.get(pair_id)
        if pair is None:
            continue
        chosen = pick_by_risk(runs_of(pair["model_output"], [tuple(x) for x in pieces], start, 0.5))
        result[pair_id] = [] if chosen is None else [chosen]
    return result


def row_for(reports, pairs_by_id, start):
    """Строка таблицы: метрики и распределение длин для одного порога."""
    pairs = []
    predicted = {}
    lengths = []
    for report in reports:
        for pair_id in report["outer_predictions"]:
            pair = pairs_by_id.get(pair_id)
            if pair is None:
                continue
            pairs.append(pair)
        predicted.update(predict(report, pairs_by_id, start))
    token = counts_for(pairs, predicted, None)
    point = score_pairs(pairs, predicted)
    for pair_id, spans in predicted.items():
        for span in spans:
            lengths.append(span["end"] - span["start"] + 1)
    lengths.sort()
    marked = sum(1 for spans in predicted.values() if spans)
    return {
        "threshold": start,
        "pairs": len(pairs),
        "token_precision": token["precision"],
        "token_recall": token["recall"],
        "token_f1": token["f1"],
        "fpr_token": token["fpr_token"],
        "exact_span_f1": point["exact_span_f1"],
        "pair_f1": point["pair_f1"],
        "fpr_pair": point["fpr_pair"],
        "marked_pairs": marked,
        "spans": len(lengths),
        "length_median": statistics.median(lengths) if lengths else 0,
        "length_p90": lengths[int(0.9 * len(lengths))] if lengths else 0,
        "length_max": lengths[-1] if lengths else 0,
        "length_share_over_400": sum(1 for x in lengths if x > 400) / len(lengths) if lengths else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--out", default="artifacts/gate62.json")
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    pairs_by_id = {pair["pair_id"]: pair for pair in pairs}
    reports = load(sorted(Path("artifacts").glob("q59_a_outer*_inner*.json")))
    print(f"отчётов A: {len(reports)}")

    canon = [reports[(outer, outer % 10 + 1)] for outer in range(1, 11) if (outer, outer % 10 + 1) in reports]
    print(f"канонических (по одному на фолд): {len(canon)}")

    table = {"gates": GATES, "fpr_limit": FPR_LIMIT, "gold_median": GOLD_MEDIAN,
             "canon": [], "all40": []}
    for start in GATES:
        canon_row = row_for(canon, pairs_by_id, start)
        all_row = row_for(list(reports.values()), pairs_by_id, start)
        table["canon"].append(canon_row)
        table["all40"].append(all_row)
        print(f"{start:.2f} | canon F1 {canon_row['token_f1']:.3f} P {canon_row['token_precision']:.3f} "
              f"R {canon_row['token_recall']:.3f} | fpr {canon_row['fpr_token']:.3f} | "
              f"длина med {canon_row['length_median']:.0f} p90 {canon_row['length_p90']}")

    # законный порог: по внутренним сеткам, с ограничением FPR
    legal = thresholds_for(reports)
    table["legal_thresholds"] = legal
    legal_pred = {}
    for outer in sorted(legal):
        report = reports[(outer, outer % 10 + 1)]
        legal_pred.update(predict(report, pairs_by_id, legal[outer]))
    token = counts_for(pairs, legal_pred, None)
    point = score_pairs(pairs, legal_pred)
    table["legal"] = {
        "thresholds": legal,
        "token_precision": token["precision"],
        "token_recall": token["recall"],
        "token_f1": token["f1"],
        "fpr_token": token["fpr_token"],
        "exact_span_f1": point["exact_span_f1"],
        "pair_f1": point["pair_f1"],
        "fpr_pair": point["fpr_pair"],
    }
    print(f"\nзаконный порог (внутренние сетки, FPR<=0,40): {[legal[o] for o in sorted(legal)]}")
    print(f"  token F1 {token['f1']:.6f} | fpr {token['fpr_token']:.6f} | exact {point['exact_span_f1']:.6f}")

    # медиана золотых спанов для сверки
    gold_lengths = [span["end"] - span["start"] + 1 for pair in pairs for span in pair["spans"]]
    table["gold_length_median"] = statistics.median(gold_lengths) if gold_lengths else 0
    table["gold_length_share_over_400"] = (sum(1 for x in gold_lengths if x > 400) / len(gold_lengths)
                                           if gold_lengths else 0.0)
    print(f"золото: медиана {table['gold_length_median']:.0f} знаков, "
          f"доля длиннее 400: {table['gold_length_share_over_400']:.1%}")

    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"записано: {args.out}")


if __name__ == "__main__":
    main()
