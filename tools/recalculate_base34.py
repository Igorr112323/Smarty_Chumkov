import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")


def main():
    original = json.loads((ROOT / "artifacts/actions33_cv_report_base3.json").read_text(encoding="utf-8"))
    values = {name: [row["tuning"]["metrics"][name] for row in original["rows"]] for name in METRICS}
    result = {"status": "математическая проверка исходных метрик; эксперимент не принят как валидный", "fold_count": len(original["rows"]), "std_convention": "population, ddof=0", "metrics": {name: {"fold_values": numbers, "mean_exact": statistics.mean(numbers), "std_population_exact": statistics.pstdev(numbers), "std_sample_exact": statistics.stdev(numbers) if len(numbers) > 1 else None, "original_report": original["summary"][name], "matches_original_rounded": {"mean": round(statistics.mean(numbers), 4) == original["summary"][name]["mean"], "std": round(statistics.pstdev(numbers), 4) == original["summary"][name]["std"]}} for name, numbers in values.items()}}
    path = ROOT / "artifacts/base34_recalculated.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
