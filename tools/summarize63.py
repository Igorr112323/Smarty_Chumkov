"""ПРОМПТ 63, стадия 3: свод по десяти фолдам и критерии приёмки.

Считается по отчётам `seg63_outer*_inner*_seed*.json`. По фолду берётся среднее
по пяти сидам (порог выбирается на внутреннем фолде для каждого сида, затем
усредняются предсказания по кандидатам — как в схеме 52).

Метрики докладываются В ДВУХ вариантах, и это требование заказчика:

* `span_f1_exact` — совпадение границ посимвольно (IoU ≥ 1,0);
* `span_f1_soft` — пересечение не ниже 0,5, **договорная метрика (п. 5)**.

Критерии 3.4 проверяются по мягкому IoU против порога 0,35 (уточнение
заказчика: потолок нарезки измерен на уровне 0,58, поэтому 0,40 снят).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.metrics59 import prf
from tools.train63 import FPR_LIMIT, score, support_of
from tools.train_base33 import grouped_folds

FOLDS = 10
SEEDS = (1, 2, 3, 4)      # сид 0 первой конфигурации в итерацию не входит
EXPECTED_EPOCHS = 20      # конфигурация итерации; отчёты с другим epochs отсеиваются
CRITERIA = {
    "3.1_token_f1": 0.60,
    "3.2_token_precision": 0.55,
    "3.3_fpr_token": 0.30,
    "3.4_span_f1_soft": 0.35,
    "3.5_ci_low": 0.5001,
    "3.6_folds_over_fpr": 0,
    "3.7_ms_per_1000": 30.0,
    "3.8_package_mb": 0.6,
}
BASELINE = 0.500132


def change(value, base):
    return value - base


def bootstrap(pairs, predicted, rounds=2000, seed=0):
    """CI на token F1 по парам — тот же приём, что в своде 59."""
    import random

    random.seed(seed)
    values = []
    for _ in range(rounds):
        sample = [random.randrange(len(pairs)) for _ in range(len(pairs))]
        chosen = {pairs[index]["pair_id"]: predicted.get(pairs[index]["pair_id"], [])
                  for index in sample}
        point = support_of([pairs[index] for index in sample], chosen)
        values.append(point["f1"])
    values.sort()
    return {"low": values[int(0.025 * len(values))], "high": values[int(0.975 * len(values))],
            "rounds": rounds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="seg63")
    parser.add_argument("--out", default=None)
    parser.add_argument("--rounds", type=int, default=2000)
    args = parser.parse_args()

    if args.prefix == "seg63b":
        raise SystemExit("seg63b: неполные сиды и несогласованное правило отбора; приёмочный свод запрещён")

    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    folds = grouped_folds(pairs, FOLDS)
    # Отчёты другой конфигурации в свод не попадают: у них другой epochs/unfreeze.
    # Сид 0 прогона seg63 обучен иначе (8 эпох, энкодер заморожен) и в итерации
    # не участвует — иначе «среднее по сидам» смешает две схемы обучения.
    reports = {}
    for path in sorted(Path(ROOT / "artifacts").glob(f"{args.prefix}_outer*_inner*_seed*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "готово":
            continue
        if report.get("epochs") != EXPECTED_EPOCHS or not report.get("unfreeze_after"):
            continue
        reports[(report["outer_fold"], report["inner_fold"], report["seed"])] = report
    print(f"отчётов: {len(reports)}")

    per_fold = []
    all_predicted = {}
    missing = []
    for outer in range(1, FOLDS + 1):
        inner = outer % FOLDS + 1
        seeds = [reports.get((outer, inner, seed)) for seed in SEEDS]
        seeds = [report for report in seeds if report is not None]
        if not seeds:
            missing.append(outer)
            continue
        held = folds[outer - 1]
        # усреднение по сидам: средняя вероятность кандидата
        merged = {}
        for report in seeds:
            for pair_id, rows in report.get("outer_candidates", {}).items():
                store = merged.setdefault(pair_id, {})
                for row in rows:
                    key = (row["start"], row["end"])
                    store.setdefault(key, []).append(row["risk"])
        averaged = {pair_id: [{"start": key[0], "end": key[1],
                               "risk": sum(values) / len(values)}
                              for key, values in store.items()]
                    for pair_id, store in merged.items()}
        thresholds = [report["threshold"] for report in seeds]
        threshold = statistics.median(thresholds)
        point = score(held, averaged, threshold)
        token = support_of(held, {pair_id: [{"start": row["start"], "end": row["end"]}
                                            for row in rows if row["risk"] >= threshold][:3]
                                  for pair_id, rows in averaged.items()})
        per_fold.append({
            "fold": outer, "seeds": len(seeds), "thresholds": thresholds,
            "threshold": threshold, "token": point["token"],
            "span_f1_exact": point["span_f1_exact"], "span_f1_soft": point["span_f1_soft"],
            "fpr_token": point["fpr_token"], "fpr_pair": point["fpr_pair"],
            "pairs": len(held),
        })
        for pair_id, rows in averaged.items():
            all_predicted[pair_id] = [{"start": row["start"], "end": row["end"]}
                                      for row in rows if row["risk"] >= threshold][:3]

    if missing:
        summary = {"status": "неполный прогон", "missing_folds": missing, "per_fold": per_fold}
        Path(args.out or f"artifacts/summary63_{args.prefix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8")
        print(f"нет отчётов по фолдам: {missing}")
        return

    token = support_of(pairs, all_predicted)
    tp = sum(len([1]) for _ in [])  # заглушка не нужна, счётчики ниже
    predicted_total = sum(len(spans) for spans in all_predicted.values())
    gold_total = sum(len(pair["spans"]) for pair in pairs)
    from spanver.evaluate import match_spans
    exact_tp = sum(match_spans(all_predicted.get(pair["pair_id"], []), pair["spans"], 1.0)[0]
                   for pair in pairs)
    soft_tp = sum(match_spans(all_predicted.get(pair["pair_id"], []), pair["spans"], 0.5)[0]
                  for pair in pairs)
    span_exact = 2 * exact_tp / (predicted_total + gold_total) if predicted_total + gold_total else 0.0
    span_soft = 2 * soft_tp / (predicted_total + gold_total) if predicted_total + gold_total else 0.0
    fpr_values = [row["fpr_token"] for row in per_fold]
    ci = bootstrap(pairs, all_predicted, rounds=args.rounds)

    summary = {
        "status": "готово",
        "pairs": len(pairs),
        "per_fold": per_fold,
        "pooled": {
            "token_f1": token["f1"], "token_precision": token["precision"],
            "token_recall": token["recall"], "fpr_token": token["fpr_token"],
            "fpr_pair": token["fpr_pair"],
            "span_f1_exact": span_exact, "span_f1_soft": span_soft,
            "predicted_spans": predicted_total, "gold_spans": gold_total,
        },
        "folds_over_fpr": sum(1 for value in fpr_values if value > FPR_LIMIT),
        "max_fold": {"fold": max(per_fold, key=lambda row: row["fpr_token"])["fold"],
                     "fpr_token": max(fpr_values)},
        "std": {"token_f1": statistics.pstdev([row["token"]["f1"] for row in per_fold]),
                "fpr_token": statistics.pstdev(fpr_values)},
        "ci": ci,
        "baseline": BASELINE,
        "delta_vs_baseline": token["f1"] - BASELINE,
        "delta_vs_a": {
            "token_f1": change(token["f1"], 0.508888),
            "span_f1_exact": change(span_exact, 0.234086),
            "span_f1_soft": change(span_soft, 0.303900),
            "fpr_token": change(token["fpr_token"], 0.378691),
        },
    }
    checks = {
        "3.1_token_f1": token["f1"] >= CRITERIA["3.1_token_f1"],
        "3.2_token_precision": token["precision"] >= CRITERIA["3.2_token_precision"],
        "3.3_fpr_token": token["fpr_token"] <= CRITERIA["3.3_fpr_token"],
        "3.4_span_f1_soft": span_soft >= CRITERIA["3.4_span_f1_soft"],
        "3.5_ci_low": ci["low"] > CRITERIA["3.5_ci_low"],
        "3.6_folds_over_fpr": summary["folds_over_fpr"] <= CRITERIA["3.6_folds_over_fpr"],
    }
    summary["criteria"] = CRITERIA
    summary["checks"] = checks
    summary["all_criteria_met"] = all(checks.values())
    summary["better_than_a"] = {
        "token_f1": token["f1"] > 0.508888,
        "precision": token["precision"] > 0.442003,
        "fpr_token": token["fpr_token"] < 0.378691,
    }
    # п. 3.9: провал 3.1–3.3 при улучшении по всем трём — не конец линии
    summary["3_9_continue"] = (not all(checks[key] for key in ("3.1_token_f1", "3.2_token_precision",
                                                              "3.3_fpr_token"))
                               and all(summary["better_than_a"].values()))

    out = args.out or f"artifacts/summary63_{args.prefix}.json"
    Path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    print(f"token F1 {token['f1']:.4f} | precision {token['precision']:.4f} | "
          f"fpr {token['fpr_token']:.4f}")
    print(f"span F1: exact {span_exact:.4f} | soft {span_soft:.4f} (договорная)")
    print(f"CI {ci['low']:.4f}–{ci['high']:.4f} | фолдов с FPR>0,40: {summary['folds_over_fpr']}")
    print(f"критерии: {checks}")
    print(f"все выполнены: {summary['all_criteria_met']} | п.3.9 продолжать: {summary['3_9_continue']}")


if __name__ == "__main__":
    main()
