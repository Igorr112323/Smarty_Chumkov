import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")


def summarize(rows):
    return {name: {"mean": round(statistics.mean(row[name] for row in rows), 4), "std": round(statistics.pstdev(row[name] for row in rows), 4)} for name in METRICS}


def main():
    paths = [ROOT / f"artifacts/nested36_fold_{index}.json" for index in range(1, 6)]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.exists()]
    result = {"status": "неполный прогон", "folds_expected": 5, "folds_available": [row.get("fold") for row in rows], "folds_complete": [row.get("fold") for row in rows if row.get("status") == "готово"], "baseline": "CV 0.5138 получена с внешними негативами; новые числа без них, прямое сравнение ограничено", "test301": "ранее израсходован; повтор не проводился", "std": "population ddof=0"}
    if len(rows) == 5 and all(row.get("status") == "готово" for row in rows):
        if [row["fold"] for row in rows] != list(range(1, 6)) or any(row["fold_ids_sha256"] != rows[0]["fold_ids_sha256"] for row in rows):
            raise ValueError("несовместимые фолды")
        if any([seed["seed"] for seed in row["seeds"]] != [0, 1, 2] or row["ensemble"]["models_per_pair"] != 3 for row in rows):
            raise ValueError("неверный состав ансамбля")
        result.update({"status": "готово", "nested_cv_seed0": summarize([row["seeds"][0]["outer_metrics"] for row in rows]), "seed_bagging_3": summarize([row["ensemble"]["outer_metrics"] for row in rows]), "fold_ids_sha256": rows[0]["fold_ids_sha256"], "thresholds": [row["threshold"] for row in rows], "ensemble_caveat": "Порог выбран на вероятностях seed 0, применён к среднему трёх моделей; отдельной калибровки ансамбля не было."})
    target = ROOT / "artifacts/nested36_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "готово" else 2


if __name__ == "__main__":
    raise SystemExit(main())
