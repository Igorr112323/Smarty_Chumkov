"""ПРОМПТ 56, критерий 2.8: свести пофолдовые замеры test 301 в два отчёта.

Вход — `artifacts/eval56_fold<N>_docs.json` и `..._joined.json`, по одному на фолд.
Счётчики объединяются (pooled), а не усредняются: фолды имеют разную поддержку.

Отдаются ровно те две цифры, которые требует промпт, для каждого входа:

  * все 301 пара;
  * чистая подвыборка (пары, чьи документы не встречаются в train+val), n указан.

Никакого отбора по test нет: пороги взяты из манифестов, выбранных на внутренних
фолдах. Скрипт только складывает уже посчитанное.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEYS = ("token_tp", "token_fp", "token_fn", "negative_tokens", "flagged_negative_tokens",
        "span_tp", "span_fp", "span_fn", "negative_pairs", "flagged_negative_pairs")


def merge(points):
    totals = {key: 0 for key in KEYS}
    for point in points:
        # pooled-блок несёт счётчики в поле counts (в per-fold — в support)
        support = point.get("counts") or point.get("support") or {}
        for key in KEYS:
            totals[key] += support.get(key, 0)
    tp, fp, fn = totals["token_tp"], totals["token_fp"], totals["token_fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    span_denom = 2 * totals["span_tp"] + totals["span_fp"] + totals["span_fn"]
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr_token": totals["flagged_negative_tokens"] / totals["negative_tokens"]
        if totals["negative_tokens"] else None,
        "fpr_pair": totals["flagged_negative_pairs"] / totals["negative_pairs"]
        if totals["negative_pairs"] else None,
        "span_exact_f1": 2 * totals["span_tp"] / span_denom if span_denom else 0.0,
        "counts": totals,
        "folds": len(points),
    }


def main():
    for entry in ("docs", "joined"):
        paths = sorted(ROOT.glob(f"artifacts/eval56_fold*_{entry}.json"))
        if not paths:
            print(f"нет файлов для входа {entry}")
            continue
        folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        clean_sizes = sorted({fold["clean_pairs"] for fold in folds})
        report = {
            "status": "готово",
            "entry": "docs" if entry == "docs" else "joined_reference",
            "folds": len(folds),
            "test_pairs": folds[0]["test_pairs"],
            "clean_pairs": folds[0]["clean_pairs"],
            "clean_pairs_variants": clean_sizes,
            "all_pairs": merge([fold["all_pairs"]["pooled"] for fold in folds]),
            "clean_subset": merge([fold["clean_subset"]["pooled"] for fold in folds]),
            "per_fold": [{"fold": fold.get("fold", index + 1),
                          "all_pairs_token_f1": fold["all_pairs"]["pooled"]["token_f1"],
                          "clean_token_f1": fold["clean_subset"]["pooled"]["token_f1"]}
                         for index, fold in enumerate(folds)],
            "test301": True,
            "usage": "финальный замер; ни порог, ни конфигурация по test не выбирались",
        }
        report["spread_over_folds"] = {
            name: statistics.pstdev([row[name] for row in report["per_fold"]])
            for name in ("all_pairs_token_f1", "clean_token_f1")} if len(folds) > 1 else {}
        target = ROOT / f"artifacts/eval56_test_{entry}.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{report['entry']}: все {report['test_pairs']} пар token F1 "
              f"{report['all_pairs']['token_f1']}; чистые {report['clean_pairs']} пар "
              f"{report['clean_subset']['token_f1']}")


if __name__ == "__main__":
    main()
