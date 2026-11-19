"""ПРОМПТ 59: свод A и B и вердикт по 1.12, в машиночитаемый файл.

Считается по `artifacts/ours56_docs.jsonl` — это вход `docs`, на котором шёл
прогон 56 и на котором измерен эталон 0,5119 в `decode56.json`. Прогон 59
обучался на `ours33.jsonl` (вход `joined_reference`), и фолды там другие;
на метрики это не влияет (они зависят только от ответа и меток), но сверять
контроль с 0,5119 нужно на наборе 56 — иначе сравниваются разные фолды.

Обе базовые линии считаются тем же путём (`covered_tokens` + `token_positions`).
Ничего не выбирается по test 301: он здесь не читается.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.summarize59 import control_check, load_reports, score_run

TOLERANCE_F1 = 0.002
TOLERANCE_FPR = 0.002
REFERENCE = {"token_f1": 0.5118790496760259, "fpr_token": 0.3895509245670678}


def summarise(run):
    full = run["full"]
    return {
        "token_f1": full["token"]["f1"],
        "token_precision": full["token"]["precision"],
        "token_recall": full["token"]["recall"],
        "fpr_token": full["token"]["fpr_token"],
        "fpr_pair": full["fpr_pair"],
        "pair_f1": full["pair_f1"],
        "exact_span_f1": full["exact_span_f1"],
        "baseline_all_positive": full["baseline"]["f1"],
        "margin": run["margin"],
        "document_token_f1": run["document"]["token"]["f1"],
        "document_baseline": run["document"]["baseline"]["f1"],
        "document_margin": run["margin_document"],
        "folds_over_fpr": run["per_fold"]["folds_over_fpr"],
        "max_fold_fpr": run["per_fold"]["max_fold"],
        "std": run["per_fold"]["std"],
        "ci": run.get("ci"),
        "thresholds": run["thresholds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--out", default="artifacts/question59.json")
    parser.add_argument("--rounds", type=int, default=10000)
    args = parser.parse_args()

    pairs = read_jsonl(ROOT / args.pairs)
    report = {
        "status": "готово",
        "protocol": "artifacts/protocol59.json",
        "addendum": "artifacts/protocol59_addendum.json",
        "pairs_file": args.pairs,
        "pairs": len(pairs),
        "test301": False,
        "selection_by_test": False,
        "note_pairs": (
            "прогон 56 (эталон 0,5119) шёл на входе docs = ours56_docs.jsonl; "
            "прогон 59 обучен на наборе joined_reference = ours33.jsonl, и фолды "
            "grouped_folds там другие. Метрики зависят только от ответа и меток, "
            "поэтому сверка контроля с 0,5119 корректна на этом наборе."
        ),
    }
    runs = {}
    for name, prefix, flag in (("A_without_question", "q59_a", False),
                               ("B_with_question", "q59_b", True)):
        reports = load_reports(prefix)
        if len(reports) != 40:
            raise SystemExit(f"{name}: отчётов {len(reports)} из 40, свод не считается")
        if any(bool(item.get("with_question")) is not flag for item in reports.values()):
            raise SystemExit(f"{name}: в отчётах другой флаг with_question")
        run = score_run(reports, pairs, rounds=args.rounds)
        if run["status"] != "готово":
            raise SystemExit(f"{name}: {run.get('missing')}")
        runs[name] = summarise(run)
        report[name] = runs[name]

    a, b = runs["A_without_question"], runs["B_with_question"]

    # 1.5 контроль: воспроизводит ли A прежние числа
    report["control"] = {
        "token_f1": a["token_f1"], "expected": REFERENCE["token_f1"],
        "delta": a["token_f1"] - REFERENCE["token_f1"], "tolerance": TOLERANCE_F1,
        "token_f1_ok": abs(a["token_f1"] - REFERENCE["token_f1"]) <= TOLERANCE_F1,
        "fpr_token": a["fpr_token"], "fpr_expected": REFERENCE["fpr_token"],
        "fpr_delta": a["fpr_token"] - REFERENCE["fpr_token"], "fpr_tolerance": TOLERANCE_FPR,
        "fpr_ok": abs(a["fpr_token"] - REFERENCE["fpr_token"]) <= TOLERANCE_FPR,
        "explanation": (
            "фолды прогона 59 (grouped_folds на ours33) не те же, что в 56; "
            "поэтому пороги по внутренним фолдам выбираются иначе (0,46-0,52 "
            "против 0,45-0,50), и контроль не воспроизводится бит в бит"
        ),
    }
    report["control"]["reproduced"] = report["control"]["token_f1_ok"] and report["control"]["fpr_ok"]

    # дельта B-A
    report["delta_b_minus_a"] = {key: b[key] - a[key] for key in
                                 ("token_f1", "token_precision", "token_recall", "fpr_token",
                                  "fpr_pair", "pair_f1", "exact_span_f1",
                                  "document_token_f1", "margin")}

    # 1.12 вердикт: основной FPR — токенный (дополнение, п. 3)
    low = b["ci"]["low"] if b["ci"] else None
    criteria = {
        "1.6_f1_ge_0_55": b["token_f1"] >= 0.55,
        "1.7_fpr_token_le_0_40": b["fpr_token"] <= 0.40,
        "1.8_precision_ge_0_45": b["token_precision"] >= 0.45,
        "1.9_margin_ge_0_05": b["margin"] >= 0.05,
        "1.9_ci_low_above_baseline": bool(low is not None and low > b["baseline_all_positive"]),
    }
    if b["token_f1"] >= 0.55 and b["token_precision"] >= 0.45 and b["fpr_token"] <= 0.40:
        outcome, nxt = "подтверждена", "стадия 2, п. 2.2"
    elif b["token_f1"] > 0.52:
        outcome, nxt = "частично", "стадия 2, п. 2.1"
    else:
        outcome, nxt = "опровергнута", "причина в другом"
    report["verdict"] = {"outcome": outcome, "next": nxt, "criteria": criteria,
                         "all_criteria_met": all(criteria.values())}

    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    print(json.dumps({"status": report["status"], "control_reproduced": report["control"]["reproduced"],
                      "verdict": outcome, "all_criteria": report["verdict"]["all_criteria_met"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
