"""ПРОМПТ 58, стадия 3.3: свести десять пофолдовых замеров в один отчёт.

Вход — `artifacts/final58_fold1..10.json`, по одному на фолд. Каждый содержит
счётчики по двум наборам (301 и 167 чистых), пофолдовые точки и построчные
результаты для bootstrap.

Счётчики фолдов объединяются (pooled), а не усредняются: фолды имеют разную
поддержку. CI считается по парам с возвращением: для каждой пары берётся её
token F1, усреднённый по десяти фолдам, затем 10 000 бутстреп-выборок пар.

Базлайн берётся из самих пофолдовых файлов и обязан совпасть между фолдами —
это проверяется, а не предполагается.
"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESAMPLES = 10000
SEED = 58
KEYS = ("token_tp", "token_fp", "token_fn", "negative_tokens", "flagged_negative_tokens",
        "span_tp", "span_fp", "span_fn")


def bootstrap_ci(values, resamples=RESAMPLES, seed=SEED):
    if not values:
        return None
    generator = random.Random(seed)
    samples = sorted(statistics.mean(values[generator.randrange(len(values))] for _ in values)
                     for _ in range(resamples))
    return [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]


def main():
    paths = sorted(ROOT.glob("artifacts/final58_fold*.json"))
    if not paths:
        raise SystemExit("нет пофолдовых файлов final58_fold*.json")
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(folds) != 10:
        raise SystemExit(f"ожидалось 10 фолдов, найдено {len(folds)}")

    # базлайн обязан быть одинаковым во всех фолдах — проверяем
    bases = {key: {json.dumps(fold["baselines"][key]["all_positive_token_f1"]) for fold in folds}
             for key in ("all_301", "clean_167")}
    baseline_consistent = all(len(values) == 1 for values in bases.values())

    report = {
        "status": "готово",
        "stage": "58 стадия 3",
        "entry": folds[0]["entry"],
        "folds": len(folds),
        "config": folds[0]["config"],
        "thresholds": [fold["fold_results"][0]["threshold"] for fold in folds],
        "baseline_consistent_across_folds": baseline_consistent,
        "baseline_source": "посчитан в каждом фолде тем же covered_tokens/token_positions на тех же парах",
        "test301_evaluated": True,
        "test301_usage": "единственный финальный замер; по test не выбиралось ничего",
    }

    for subset, name in (("all", "all_301"), ("clean", "clean_167")):
        totals = {key: 0 for key in KEYS}
        pair_tp = pair_fp = pair_fn = 0
        values = []
        pairs_count = None
        for fold in folds:
            for item in fold["fold_results"]:
                if item["subset"] != subset:
                    continue
                pairs_count = len(fold["per_pair"])
                for key in KEYS:
                    totals[key] += item["point"]["support"].get(key, 0)
        # построчные значения: по каждой паре среднее по фолдам
        by_pair = {}
        for fold in folds:
            for pair_id, rows_ in fold["per_pair"].items():
                for row in rows_:
                    if row["subset"] != subset:
                        continue
                    by_pair.setdefault(pair_id, []).append(row)
        for pair_id, rows_ in by_pair.items():
            values.append(statistics.mean(row["token_f1"] for row in rows_))
            pair_tp += sum(row["pair_tp"] for row in rows_)
            pair_fp += sum(row["pair_fp"] for row in rows_)
            pair_fn += sum(row["pair_fn"] for row in rows_)

        tp, fp, fn = totals["token_tp"], totals["token_fp"], totals["token_fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        span_denom = 2 * totals["span_tp"] + totals["span_fp"] + totals["span_fn"]
        pair_denom = 2 * pair_tp + pair_fp + pair_fn
        base = folds[0]["baselines"][name]
        report[name] = {
            "pairs": len(by_pair),
            "evaluations": len(folds) * len(by_pair),
            "token_f1": f1,
            "token_precision": precision,
            "token_recall": recall,
            "fpr_token": totals["flagged_negative_tokens"] / totals["negative_tokens"]
            if totals["negative_tokens"] else None,
            "span_exact_f1": 2 * totals["span_tp"] / span_denom if span_denom else 0.0,
            "pair_f1": 2 * pair_tp / pair_denom if pair_denom else 0.0,
            "pair_counts": {"tp": pair_tp, "fp": pair_fp, "fn": pair_fn},
            "baseline": base,
            "margin_token": f1 - base["all_positive_token_f1"],
            "margin_pair": (2 * pair_tp / pair_denom if pair_denom else 0.0)
            - base["all_positive_pair_f1"],
            "ci95_token_f1_over_pairs": bootstrap_ci(values),
            "ci_unit": "пары, среднее по фолдам, 10 000 бутстреп-выборок",
            "counts": totals,
            "per_fold_token_f1": [item["point"]["token_f1"] for fold in folds
                                  for item in fold["fold_results"] if item["subset"] == subset],
        }
        report[name]["per_fold_std"] = statistics.pstdev(report[name]["per_fold_token_f1"])

    # скорость берётся из отдельного замера, если он есть
    speed_path = ROOT / "artifacts/final58_speed.json"
    report["speed"] = (json.loads(speed_path.read_text(encoding="utf-8"))
                       if speed_path.exists() else None)

    (ROOT / "artifacts/final58.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for name in ("all_301", "clean_167"):
        item = report[name]
        print(f"{name}: пар {item['pairs']}, F1 {item['token_f1']}, P {item['token_precision']}, "
              f"R {item['token_recall']}, FPR {item['fpr_token']}, span {item['span_exact_f1']}, "
              f"pair F1 {item['pair_f1']}")
        print(f"    базлайн {item['baseline']['all_positive_token_f1']}, "
              f"отрыв {item['margin_token']:+.6f}, CI {item['ci95_token_f1_over_pairs']}")


if __name__ == "__main__":
    main()
