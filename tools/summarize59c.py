"""ПРОМПТ 59C: свод прогона двухступенчатого отбора и проверка критериев C.1-C.7.

Отчёты `artifacts/q59c_outer{N}_inner{I}.json` содержат `outer_predictions` в
том же формате, что A и B (список спанов на пару), поэтому метрики считаются тем
же кодом (`metrics59`), а разброс — той же процедурой, что в своде 59.

Критерии зафиксированы в `artifacts/protocol59c.json` до прогона.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.metrics59 import DOCUMENT_TYPES, bootstrap_ci, score_pairs
from tools.summarize59 import baseline_of, counts_for, fold_threshold, load_reports, thresholds_for
from tools.train_base33 import grouped_folds

FOLDS = 10

# зафиксировано до прогона, artifacts/protocol59c.json
CRITERIA = {
    "C.1_fpr_token_le_0_3896": 0.3896,
    "C.2_token_f1_ge_0_5165": 0.5165,
    "C.3_exact_span_f1_ge_0_2581": 0.2581,
    "C.4_pair_f1_ge_0_3316": 0.3316,
    "C.5_precision_ge_0_45": 0.45,
    "C.6_margin_ge_0_02_and_ci_low_above_baseline": 0.02,
    "C.7_folds_over_fpr_le_4": 4,
}
BASE_A = {"token_f1": 0.508888, "fpr_token": 0.378691, "exact_span_f1": 0.234086,
          "pair_f1": 0.303191, "precision": 0.442003, "folds_over_fpr": 4}
BASE_B = {"token_f1": 0.516522, "fpr_token": 0.407631, "exact_span_f1": 0.258065,
          "pair_f1": 0.331606, "precision": 0.436775, "folds_over_fpr": 6}


def fold_rows(reports, pairs_by_id, thresholds):
    """Метрики каждого внешнего фолда: pooled не публикуется без разброса."""
    import statistics

    rows = []
    for outer in sorted(thresholds):
        report = reports.get((outer, outer % FOLDS + 1))
        if report is None:
            continue
        fold_pairs = [pairs_by_id[pid] for pid in report["outer_predictions"]]
        predicted = report["outer_predictions"]
        token = counts_for(fold_pairs, predicted, None)
        scored = score_pairs(fold_pairs, predicted)
        rows.append({
            "fold": outer,
            "token_f1": token["f1"],
            "fpr_token": token["fpr_token"],
            "exact_span_f1": scored["exact_span_f1"],
            "pair_f1": scored["pair_f1"],
            "precision": token["precision"],
        })
    f1 = [row["token_f1"] for row in rows]
    fpr = [row["fpr_token"] for row in rows]
    worst = max(rows, key=lambda row: row["fpr_token"])
    return {
        "per_fold": rows,
        "folds_over_fpr": sum(1 for value in fpr if value > 0.40),
        "max_fold": {"fold": worst["fold"], "fpr_token": worst["fpr_token"]},
        "std": {"token_f1": statistics.pstdev(f1), "fpr_token": statistics.pstdev(fpr)},
        "min_fold_token_f1": min(f1),
        "max_fold_token_f1": max(f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="q59c")
    parser.add_argument("--pairs", default="artifacts/ours33.jsonl")
    parser.add_argument("--out", default="artifacts/question59c.json")
    parser.add_argument("--rounds", type=int, default=10000)
    args = parser.parse_args()

    report = {
        "status": "неполный прогон",
        "protocol": "artifacts/protocol59c.json",
        "pairs": args.pairs,
        "test301": False,
        "selection_by_test": False,
        "criteria_thresholds": CRITERIA,
        "comparison_base": {"A": BASE_A, "B": BASE_B,
                            "note": "0,5119 не цитировать как контроль: другой набор"},
    }
    source = ROOT / args.pairs
    pairs = read_jsonl(source)
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    reports = load_reports(args.prefix)
    report["reports"] = len(reports)
    if len(reports) != 40:
        report["missing"] = f"отчётов {len(reports)} из 40"
        (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                     encoding="utf-8")
        print(json.dumps({"status": report["status"], "reports": len(reports)}))
        return 2

    thresholds = thresholds_for(reports)
    # отчёты C уже содержат спаны в формате pair_id -> [{start, end}], декодер
    # применять не нужно (в отличие от A/B, где лежали вероятности)
    predicted_all = {}
    for outer in sorted(thresholds):
        predicted_all.update(reports[outer, outer % FOLDS + 1]["outer_predictions"])
    if predicted_all and not isinstance(next(iter(predicted_all.values())), list):
        raise SystemExit("формат outer_predictions в отчётах C не спаны")

    token = counts_for(pairs, predicted_all, None)
    document = counts_for(pairs, predicted_all, DOCUMENT_TYPES)
    scored = score_pairs(pairs, predicted_all)
    baseline = baseline_of(pairs)
    ci = bootstrap_ci(pairs, predicted_all, "full", rounds=args.rounds)
    folds = fold_rows(reports, pairs_by_id, thresholds)

    result = {
        "status": "готово",
        "token_f1": token["f1"],
        "token_precision": token["precision"],
        "token_recall": token["recall"],
        "fpr_token": token["fpr_token"],
        "fpr_pair": scored["fpr_pair"],
        "exact_span_f1": scored["exact_span_f1"],
        "pair_f1": scored["pair_f1"],
        "document_token_f1": document["f1"],
        "baseline": baseline,
        "margin": token["f1"] - baseline,
        "ci": ci,
        "folds": folds,
        "thresholds": thresholds,
    }
    low = ci["low"]
    checks = {
        "C.1": result["fpr_token"] <= CRITERIA["C.1_fpr_token_le_0_3896"],
        "C.2": result["token_f1"] >= CRITERIA["C.2_token_f1_ge_0_5165"],
        "C.3": result["exact_span_f1"] >= CRITERIA["C.3_exact_span_f1_ge_0_2581"],
        "C.4": result["pair_f1"] >= CRITERIA["C.4_pair_f1_ge_0_3316"],
        "C.5": result["token_precision"] >= CRITERIA["C.5_precision_ge_0_45"],
        "C.6": result["margin"] >= 0.02 and low > baseline,
        "C.7": folds["folds_over_fpr"] <= 4,
    }
    result["checks"] = checks
    result["all_criteria_met"] = all(checks.values())
    result["main_goal"] = checks["C.1"] and checks["C.3"]
    report.update(result)

    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    print(json.dumps({"status": "готово", "token_f1": round(result["token_f1"], 6),
                      "exact_span_f1": round(result["exact_span_f1"], 6),
                      "fpr_token": round(result["fpr_token"], 6),
                      "checks": checks, "main_goal": result["main_goal"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
