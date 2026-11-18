"""ПРОМПТ 56, критерии 2.1–2.6: сводка перепрогона схемы 52 на входе `docs`.

Путь расчёта повторяет `decode52.py` один в один:

  1. порог выбирается ТОЛЬКО по внутренним фолдам правилом `pooled`
     (объединённые счётчики четырёх внутренних фолдов, FPR <= 0.40);
  2. на внешнем фолде берутся сохранённые `outer_predictions` и тот же самый
     декодер (`spanver.snap.snapped`, клаузный expand) с порогами
     start=порог, extend=порог/2, gate=0.45;
  3. счётчики суммируются по 10 фолдам (pooled) и по фолдам отдельно (macro/std).

Ничего не подгоняется. При провале 2.1 отчёт пишет разбивку по фолдам.
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
from tools.decode52 import FPR_LIMIT, evaluate, exact_overlap
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
GATE = 0.45


def totals(rows):
    keys = ("token_tp", "token_fp", "token_fn", "negative_tokens", "flagged_negative_tokens")
    out = {key: 0 for key in keys}
    for row in rows:
        for key in keys:
            out[key] += row["support"].get(key, 0)
    return out


def pooled_metrics(rows):
    counts = totals(rows)
    tp, fp, fn = counts["token_tp"], counts["token_fp"], counts["token_fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    span_tp = sum(row.get("span_counts", {}).get("exact_tp", 0) for row in rows)
    span_pred = sum(row.get("span_counts", {}).get("predicted", 0) for row in rows)
    span_gold = sum(row.get("span_counts", {}).get("gold", 0) for row in rows)
    pair_tp = sum(row.get("support", {}).get("pair_tp", 0) for row in rows)
    pair_fp = sum(row.get("support", {}).get("pair_fp", 0) for row in rows)
    pair_fn = sum(row.get("support", {}).get("pair_fn", 0) for row in rows)
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr_token": counts["flagged_negative_tokens"] / counts["negative_tokens"]
        if counts["negative_tokens"] else None,
        "span_exact_f1": 2 * span_tp / (span_pred + span_gold) if span_pred + span_gold else 0.0,
        "pair_f1": 2 * pair_tp / (2 * pair_tp + pair_fp + pair_fn) if pair_tp + pair_fp + pair_fn else 0.0,
        "counts": counts,
    }


def pick_pooled(inner_grids):
    best = None
    for index in range(len(inner_grids[0])):
        rows = [grid[index] for grid in inner_grids]
        if any(not (row.get("support") or {}).get("negative_tokens") for row in rows):
            continue
        tp = sum(row["support"]["token_tp"] for row in rows)
        fp = sum(row["support"]["token_fp"] for row in rows)
        fn = sum(row["support"]["token_fn"] for row in rows)
        flagged = sum(row["support"]["flagged_negative_tokens"] for row in rows)
        negatives = sum(row["support"]["negative_tokens"] for row in rows)
        fpr = flagged / negatives if negatives else None
        if fpr is None or fpr > FPR_LIMIT:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        point = rows[0]["start"]
        if best is None or (f1, -point) > (best[0], -best[1]):
            best = (f1, point, index)
    return best


def bootstrap(folds):
    values = [pooled_metrics([row])["token_f1"] for row in folds]
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

    per_fold = []
    thresholds = {}
    for outer in range(1, 11):
        grids = [records[outer, ((outer - 1 + offset) % 10) + 1]["inner_grid"] for offset in range(1, 5)]
        choice = pick_pooled(grids)
        if choice is None:
            raise SystemExit(f"нет допустимого порога для фолда {outer}")
        thresholds[outer] = {"start": choice[1], "inner_pooled_token_f1": choice[0],
                             "grid_index": choice[2]}
        primary = records[outer, outer % 10 + 1]
        if not primary.get("primary"):
            raise SystemExit(f"фолд {outer}: нет внешних вероятностей основного внутреннего фолда")
        held = folds[outer - 1]
        if set(primary["outer_predictions"]) != {pair["pair_id"] for pair in held}:
            raise SystemExit(f"фолд {outer}: состав пар не совпадает с сохранённым фолдом")
        point = evaluate(held, primary["outer_predictions"], choice[1], GATE)
        point.update(exact_overlap(held, primary["outer_predictions"], choice[1], GATE))
        per_fold.append(point)

    pooled = pooled_metrics(per_fold)
    report = {
        "status": "готово",
        "entry": "docs",
        "pairs_id_sha256": id_digest(pairs),
        "pairs": len(pairs),
        "folds": 10,
        "rule": "pooled",
        "threshold_by_fold": thresholds,
        "per_fold": [pooled_metrics([row]) for row in per_fold],
        "pooled": pooled,
        "macro": {name: statistics.mean(pooled_metrics([row])[name] for row in per_fold)
                  for name in ("token_f1", "token_precision", "fpr_token")},
        "std": {name: statistics.pstdev([pooled_metrics([row])[name] for row in per_fold])
                for name in ("token_f1", "token_precision", "fpr_token")},
        "min_fold": {name: min(pooled_metrics([row])[name] for row in per_fold)
                     for name in ("token_f1", "token_precision", "fpr_token")},
        "max_fold": {name: max(pooled_metrics([row])[name] for row in per_fold)
                     for name in ("token_f1", "token_precision", "fpr_token")},
        "folds_over_fpr": sum(1 for row in per_fold
                              if (pooled_metrics([row])["fpr_token"] or 0) > FPR_LIMIT),
        "bootstrap": bootstrap(per_fold),
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
                                     "fpr_pair", "span_exact_f1")}
        report["delta"] = {key: (pooled.get(key) - old.get(key))
                           if pooled.get(key) is not None and old.get(key) is not None else None
                           for key in ("token_f1", "token_precision", "fpr_token")}

    if not report["acceptance_all"]:
        report["loss_breakdown"] = [
            {"fold": index + 1, **pooled_metrics([row])} for index, row in enumerate(per_fold)
        ]

    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": {key: value for key, value in pooled.items() if key != "counts"},
                      "acceptance": checks, "acceptance_all": report["acceptance_all"]},
                     ensure_ascii=False, indent=1))
    # провал приёмки не роняет задание: потолок, скорость и test должны быть измерены,
    # а решение по 2.1–2.3 принимается по отчёту, а не кодом возврата
    if not report["acceptance_all"]:
        print("ВНИМАНИЕ: приёмка 2.1–2.3 не выполнена целиком, см. artifacts/decode56.json",
              file=sys.stderr)


if __name__ == "__main__":
    main()
