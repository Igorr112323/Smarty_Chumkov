"""ПРОМПТ 59: свод прогона A (без вопроса) и B (с вопросом).

Читает отчёты `artifacts/q59_{a,b}_outer{N}_inner{I}.json` (10 фолдов × 4
внутренних, как в схеме 56) и считает то, что зафиксировано в
`artifacts/protocol59.json`:

* token F1 pooled по всем внешним фолдам — основная метрика, п. 0.1;
* ту же величину на «документных» типах — вторая строка, п. 0.2;
* базовую линию своего набора и отрыв от неё, п. 0.4;
* CI 95 % bootstrap по парам, п. 1.11;
* вердикт по правилу 1.12.

Порог. Выбирается **только по внутренним фолдам** (объединённые счётчики,
FPR ≤ 0,40) и применяется к отложенному фолду — та же процедура, что дала
0,5119 в `decode56.json`. По test 301 не выбирается ничего: этот прогон его
не касается.

Отличие от `summarize59.py` (первая версия): там порог брался из
`threshold_point` отчёта, а в схеме 56 порогов было по одному на фолд и они
лежали в `decode56.json`. Здесь порог пересчитывается из `inner_grid` всех
внутренних фолдов, поэтому свод не зависит от того, что успел записать фолд.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, read_jsonl, token_positions
from tools.metrics59 import (DOCUMENT_TYPES, QUESTION_TYPES, bootstrap_ci, margin,
                             prf, score_pairs)
from tools.rule57 import FPR_LIMIT, merge, metrics, spans_from_pieces
from tools.train_base33 import grouped_folds

FOLDS = 10
INNER_OFFSETS = (1, 2, 3, 4)


def inner_shift(outer):
    """Номера внутренних фолдов внешнего фолда (1-based), как в схеме 56."""
    return [((outer - 1 + offset) % FOLDS) + 1 for offset in INNER_OFFSETS]


def load_reports(prefix):
    """Отчёты фолдов: (outer, inner) -> отчёт."""
    found = {}
    for path in sorted(ROOT.glob(f"artifacts/{prefix}_outer*_inner*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "готово":
            continue
        found[int(data["outer_fold"]), int(data["inner_fold"])] = data
    return found


def pick_threshold(grids):
    """Порог по объединённым счётчикам внутренних фолдов при FPR ≤ 0,40.

    Возвращает точку сетки с максимальным token F1 среди допустимых. Ровно то
    правило, которым получены пороги 0,45–0,50 в `decode56.json`.
    """
    if not grids:
        return None
    best = None
    for index in range(len(grids[0])):
        rows = [grid[index] for grid in grids]
        if any(not row["support"]["negative_tokens"] for row in rows):
            continue
        point = metrics(merge([row["support"] for row in rows]), 0, 0)
        if point["fpr_token"] > FPR_LIMIT:
            continue
        start = grids[0][index]["start"]
        if best is None or (point["token_f1"], -start) > (best["token_f1"], -best["start"]):
            best = {"start": start, "token_f1": point["token_f1"],
                    "fpr_token": point["fpr_token"], "index": index}
    return best


def fold_threshold(reports, outer):
    """Порог внешнего фолда: по его четырём внутренним, не по внешнему."""
    grids = []
    for inner in inner_shift(outer):
        report = reports.get((outer, inner))
        if report is None or not report.get("inner_grid"):
            return None
        grids.append(report["inner_grid"])
    return pick_threshold(grids)


def predict_fold(report, pairs_by_id, start):
    """Предсказания внешнего фолда на пороге его внутренних фолдов."""
    predicted = {}
    for pair_id, pieces in report["outer_predictions"].items():
        pair = pairs_by_id.get(pair_id)
        if pair is None:
            raise ValueError(f"пара {pair_id} не из набора")
        predicted[pair_id] = spans_from_pieces(pair["model_output"], pieces, start,
                                               start / 2, report.get("gate", 0.45),
                                               1, 0.10)
    return predicted


def counts_for(pairs, predicted_by_id, types):
    """Счётчики токенов для baseline и для подсчёта CI."""
    tp = fp = fn = negative = flagged = 0
    for pair in pairs:
        answer = pair["model_output"]
        gold_spans = [s for s in pair["spans"] if types is None or s.get("type_src") in types]
        gold = covered_tokens(answer, [(s["start"], s["end"]) for s in gold_spans])
        predicted = covered_tokens(answer, [(s["start"], s["end"])
                                            for s in predicted_by_id.get(pair["pair_id"], [])])
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        positions = token_positions(answer)
        negative += len(positions) - len(gold)
        flagged += len(predicted - gold)
    point = prf(tp, fp, fn)
    point["fpr_token"] = flagged / negative if negative else 0.0
    return point


def baseline(pairs, types):
    """Базовая линия «пометить всё»: покрыть все токены ответа."""
    marks = {}
    for pair in pairs:
        marks[pair["pair_id"]] = [{"start": start, "end": end, "type": None}
                                  for start, end in token_positions(pair["model_output"])]
    return counts_for(pairs, marks, types)


def score_run(reports, pairs, with_bootstrap=True, rounds=10000):
    """Полный свод по десяти фолдам: pooled, документные типы, базлайны, CI."""
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    predicted_all, thresholds, outside = {}, {}, []
    for outer in range(1, FOLDS + 1):
        report = reports.get((outer, outer % FOLDS + 1))
        if report is None:
            outside.append(f"нет отчёта внешнего фолда {outer}")
            continue
        point = fold_threshold(reports, outer)
        if point is None:
            outside.append(f"фолд {outer}: нет допустимого порога по внутренним")
            continue
        thresholds[outer] = point["start"]
        for pair_id, spans in predict_fold(report, pairs_by_id, point["start"]).items():
            predicted_all[pair_id] = spans
    result = {
        "thresholds": thresholds,
        "folds_scored": sorted(thresholds),
        "missing": outside,
        "pairs_scored": len(predicted_all),
    }
    if outside:
        result["status"] = "неполный прогон"
        return result
    if len(predicted_all) != len(pairs):
        raise ValueError(f"покрыто {len(predicted_all)} пар из {len(pairs)}")

    # п. 0.1 / 0.2: полная метка и документные типы одним и тем же путём
    scored = score_pairs(pairs, predicted_all)
    full = counts_for(pairs, predicted_all, None)
    document = counts_for(pairs, predicted_all, DOCUMENT_TYPES)
    question = counts_for(pairs, predicted_all, QUESTION_TYPES)
    result.update({
        "status": "готово",
        "full": {"token": full, "baseline": baseline(pairs, None),
                 "exact_span_f1": scored["exact_span_f1"], "pair_f1": scored["pair_f1"],
                 "fpr_pair": scored["fpr_pair"]},
        "document": {"types": list(DOCUMENT_TYPES), "token": document,
                     "baseline": baseline(pairs, DOCUMENT_TYPES)},
        "question_types": {"types": list(QUESTION_TYPES), "token": question,
                           "baseline": baseline(pairs, QUESTION_TYPES)},
    })
    result["margin"] = margin(full["f1"], result["full"]["baseline"]["f1"])
    result["margin_document"] = margin(document["f1"], result["document"]["baseline"]["f1"])
    if with_bootstrap:
        result["ci"] = bootstrap_ci(pairs, predicted_all, "full", rounds=rounds)
        result["ci_document"] = bootstrap_ci(pairs, predicted_all, "document", rounds=rounds)
    return result


def control_check(run_a):
    """Критерий 1.5: контроль воспроизводит 0,5119 и FPR 0,3896."""
    f1 = run_a["full"]["token"]["f1"]
    fpr = run_a["full"]["token"]["fpr_token"]
    return {
        "token_f1": f1, "expected": 0.5118790496760259, "tolerance": 0.002,
        "token_f1_ok": abs(f1 - 0.5118790496760259) <= 0.002,
        "fpr_token": fpr, "fpr_expected": 0.3895509245670678, "fpr_tolerance": 0.002,
        "fpr_ok": abs(fpr - 0.3895509245670678) <= 0.002,
    }


def verdict(run_b):
    """Правило 1.12, зафиксированное до прогона."""
    full = run_b["full"]["token"]
    f1, precision = full["f1"], full["precision"]
    fpr = run_b["full"]["fpr_pair"]
    low = run_b["ci"]["low"] if run_b.get("ci") else None
    base = run_b["full"]["baseline"]["f1"]
    criteria = {
        "1.6_f1_ge_0_55": f1 >= 0.55,
        "1.7_fpr_pair_le_0_40": fpr <= 0.40,
        "1.8_precision_ge_0_45": precision >= 0.45,
        "1.9_margin_ge_0_05": run_b["margin"] >= 0.05,
        "1.9_ci_low_above_baseline": bool(low is not None and low > base),
    }
    if f1 >= 0.55 and precision >= 0.45 and fpr <= 0.40:
        outcome, nxt = "подтверждена", "стадия 2, п. 2.2"
    elif f1 > 0.52:
        outcome, nxt = "частично", "стадия 2, п. 2.1"
    else:
        outcome, nxt = "опровергнута", "причина в другом"
    return {"outcome": outcome, "next": nxt, "criteria": criteria,
            "all_criteria_met": all(criteria.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="q59_a")
    parser.add_argument("--b", default="q59_b")
    parser.add_argument("--pairs", default="artifacts/ours33.jsonl")
    parser.add_argument("--out", default="artifacts/question59.json")
    parser.add_argument("--rounds", type=int, default=10000)
    parser.add_argument("--only", default=None, choices=("a", "b"))
    args = parser.parse_args()

    source = ROOT / args.pairs
    pairs = read_jsonl(source)
    output = {
        "status": "неполный прогон",
        "protocol": "artifacts/protocol59.json",
        "pairs": len(pairs),
        "pairs_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "folds": FOLDS,
        "inner_offsets": list(INNER_OFFSETS),
        "test301": False,
        "selection_by_test": False,
        "primary_metric": "token F1 на полной метке (все 6 типов), pooled",
        "secondary_metric": "token F1 на Contradiction+Unconfirmed+Excess, pooled",
    }
    runs = {"a": ("A_without_question", args.a), "b": ("B_with_question", args.b)}
    for key, (name, prefix) in runs.items():
        if args.only and args.only != key:
            continue
        reports = load_reports(prefix)
        output[f"{name}_reports"] = len(reports)
        if not reports:
            continue
        if any(bool(report.get("with_question")) != (key == "b") for report in reports.values()):
            raise ValueError(f"{name}: в отчётах другой флаг with_question")
        output[name] = score_run(reports, pairs, rounds=args.rounds)

    a = output.get("A_without_question")
    b = output.get("B_with_question")
    if a and a.get("status") == "готово":
        output["control"] = control_check(a)
    if a and b and a.get("status") == "готово" and b.get("status") == "готово":
        output["verdict"] = verdict(b)
        output["delta"] = {
            "full_token_f1": b["full"]["token"]["f1"] - a["full"]["token"]["f1"],
            "document_token_f1": b["document"]["token"]["f1"] - a["document"]["token"]["f1"],
        }
        output["status"] = "готово"

    target = ROOT / args.out
    target.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"],
                      "a_reports": output.get("A_without_question_reports"),
                      "b_reports": output.get("B_with_question_reports")}, ensure_ascii=False))
    return 0 if output["status"] == "готово" else 2


if __name__ == "__main__":
    raise SystemExit(main())
