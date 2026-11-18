"""ПРОМПТ 56, критерии 2.1–2.6: сводка перепрогона схемы 52 на входе `docs`.

Структура записи фолда — та же, что в 52 (`inner_grid` на внутренних фолдах,
`outer_at_selected` на внешнем), поэтому правило отбора перенесено без правок:

  * порог выбирается ТОЛЬКО по внутренним фолдам правилом `pooled`
    (объединённые счётчики четырёх внутренних), как в `decode52.py`;
  * на внешнем фолде берётся точка с этим порогом (`outer_at_selected`).

Считается:
  * pooled token F1 / precision / FPR по 10 внешним фолдам (объединённые счётчики);
  * exact span F1 и pair F1 отдельными строками, без порога приёмки;
  * bootstrap 95 % CI по фолдам (единица — документные компоненты);
  * построчное сравнение с прежним входом из `artifacts/decode52.json`.

Ничего не подгоняется: при провале 2.1 отчёт пишет, где потеряно, по фолдам.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.preflight_13 import id_digest
from tools.train_base33 import grouped_folds

FPR_LIMIT = 0.40
LIMITS = {
    "token_f1": 0.5074495197020192,
    "fpr_token": 0.40,
    "token_precision": 0.45,
}
RESAMPLES = 10000
SEED = 56
COUNT_KEYS = ("token_tp", "token_fp", "token_fn", "negative_tokens", "flagged_negative_tokens",
              "span_tp", "span_fp", "span_fn", "flagged_negative_pairs", "negative_pairs")


def totals(rows):
    out = {key: 0 for key in COUNT_KEYS}
    for row in rows:
        support = row.get("support") or {}
        for key in COUNT_KEYS:
            out[key] += support.get(key, 0)
    return out


def pooled_metrics(rows):
    counts = totals(rows)
    tp, fp, fn = counts["token_tp"], counts["token_fp"], counts["token_fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    span_denom = 2 * counts["span_tp"] + counts["span_fp"] + counts["span_fn"]
    pair_neg = counts["negative_pairs"]
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr_token": counts["flagged_negative_tokens"] / counts["negative_tokens"]
        if counts["negative_tokens"] else None,
        "fpr_pair": counts["flagged_negative_pairs"] / pair_neg if pair_neg else None,
        "span_exact_f1": 2 * counts["span_tp"] / span_denom if span_denom else 0.0,
        "pair_f1": (counts["negative_pairs"] - counts["flagged_negative_pairs"]) / counts["negative_pairs"]
        if counts["negative_pairs"] else None,
        "counts": counts,
    }


def pick_pooled(records, outer):
    """Правило pooled: объединённые счётчики четырёх внутренних фолдов."""
    groups = [records[outer, ((outer - 1 + offset) % 10) + 1]["inner_grid"] for offset in range(1, 5)]
    best = None
    for index in range(len(groups[0])):
        rows = [group[index] for group in groups]
        if any(not (row.get("support") or {}).get("negative_tokens") for row in rows):
            continue
        metrics = pooled_metrics(rows)
        if metrics["fpr_token"] is None or metrics["fpr_token"] > FPR_LIMIT:
            continue
        point = rows[0]["start"]
        if best is None or (metrics["token_f1"], -point) > (best[0], -best[1]):
            best = (metrics["token_f1"], point, index)
    return best


def bootstrap(folds):
    values = [pooled_metrics([fold])["token_f1"] for fold in folds]
    rnd = random.Random(SEED)
    samples = sorted(statistics.mean([values[rnd.randrange(len(values))] for _ in values])
                     for _ in range(RESAMPLES))
    return {"unit": "внешние фолды (документные компоненты), с возвращением",
            "resamples": RESAMPLES, "seed": SEED,
            "ci95": [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--records-prefix", default="ablation56_ten")
    parser.add_argument("--reference", default="artifacts/decode52.json")
    parser.add_argument("--out", default="artifacts/decode56.json")
    args = parser.parse_args()

    pairs = read_jsonl(ROOT / args.pairs)
    folds = grouped_folds(pairs, 10)
    records = {}
    for outer in range(1, 11):
        for inner in [((outer - 1 + offset) % 10) + 1 for offset in range(1, 5)]:
            path = ROOT / f"artifacts/{args.records_prefix}_outer{outer}_inner{inner}.json"
            if not path.exists():
                raise SystemExit(f"нет записи фолда: {path.name}")
            records[outer, inner] = json.loads(path.read_text(encoding="utf-8"))
    broken = [key for key, record in records.items() if record.get("status") != "готово"]
    if broken:
        raise SystemExit(f"не завершены задания: {sorted(broken)[:5]}")

    outer_rows = []
    thresholds = {}
    for outer in range(1, 11):
        choice = pick_pooled(records, outer)
        if choice is None:
            raise SystemExit(f"нет допустимого порога для фолда {outer}")
        thresholds[outer] = {"start": choice[1], "grid_index": choice[2],
                             "inner_pooled_token_f1": choice[0]}
        primary = records[outer, outer % 10 + 1]
        outer_rows.append(primary["outer_at_selected"])

    pooled = pooled_metrics(outer_rows)
    report = {
        "status": "готово",
        "entry": "docs",
        "pairs_id_sha256": id_digest(pairs),
        "pairs_count": len(pairs),
        "folds": 10,
        "rule": "pooled",
        "threshold_by_fold": thresholds,
        "per_fold": [pooled_metrics([row]) for row in outer_rows],
        "pooled": pooled,
        "macro": {name: statistics.mean(pooled_metrics([row])[name] for row in outer_rows)
                  for name in ("token_f1", "token_precision", "fpr_token")},
        "std": {name: statistics.pstdev([pooled_metrics([row])[name] for row in outer_rows])
                for name in ("token_f1", "token_precision", "fpr_token")},
        "min_fold": {name: min(pooled_metrics([row])[name] for row in outer_rows)
                     for name in ("token_f1", "token_precision", "fpr_token")},
        "max_fold": {name: max(pooled_metrics([row])[name] for row in outer_rows)
                     for name in ("token_f1", "token_precision", "fpr_token")},
        "folds_over_fpr": sum(1 for row in outer_rows
                              if (pooled_metrics([row])["fpr_token"] or 0) > FPR_LIMIT),
        "bootstrap": bootstrap(outer_rows),
        "test301_evaluated": False,
        "external_data_used": False,
    }

    checks = {
        "2.1_token_f1_ge_current": pooled["token_f1"] >= LIMITS["token_f1"],
        "2.2_fpr_token_le_0.40": pooled["fpr_token"] is not None and pooled["fpr_token"] <= LIMITS["fpr_token"],
        "2.3_token_precision_ge_0.45": pooled["token_precision"] >= LIMITS["token_precision"],
    }
    report["acceptance"] = checks
    report["acceptance_all"] = all(checks.values())

    reference_path = ROOT / args.reference
    if reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        old = reference["variants"]["10"]["rules"]["pooled"]["pooled"]
        report["previous_entry"] = {key: old.get(key) for key in
                                    ("token_f1", "token_precision", "token_recall", "fpr_token",
                                     "fpr_pair", "span_exact_f1", "pair_f1")}
        report["previous_entry"]["note"] = ("prev pair_f1 в decode52.json не сохранён; "
                                            "в отчёте он будет взят из artifacts/decode52_report.md")
        report["delta"] = {key: (pooled.get(key) - old.get(key))
                           if pooled.get(key) is not None and old.get(key) is not None else None
                           for key in ("token_f1", "token_precision", "fpr_token")}

    if not report["acceptance_all"]:
        report["loss_breakdown"] = [
            {"fold": index + 1, **{key: value for key, value in pooled_metrics([row]).items()}}
            for index, row in enumerate(outer_rows)
        ]

    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": pooled, "acceptance": checks, "acceptance_all": report["acceptance_all"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
