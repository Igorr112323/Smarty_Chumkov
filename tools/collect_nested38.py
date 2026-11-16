import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.nested_cv35 import evaluate, verify_fast
from tools.nested38 import GRID, file_digest, target_inner
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.summarize_nested36 import METRICS
from tools.sweep_ragtruth30 import within_fpr_limits
from tools.train_base33 import grouped_folds


def select_worst(rows):
    if len(rows) != 4:
        raise ValueError("нужны четыре внутренних фолда")
    if any([row["start"] for row in grid] != list(GRID) for grid in rows):
        raise ValueError("сетки внутренних фолдов не совпадают")
    candidates = []
    for index, threshold in enumerate(GRID):
        group = [grid[index] for grid in rows]
        allowed = all(within_fpr_limits(row) and row["support"]["negative_pairs"] and row["support"]["negative_tokens"] for row in group)
        if allowed:
            candidates.append({"start": threshold, "extend": threshold / 2, "mean_inner_token_f1_exact": statistics.mean(row["token_f1_exact"] for row in group), "max_inner_fpr_pair": max(row["fpr_pair"] for row in group), "max_inner_fpr_token": max(row["fpr_token"] for row in group)})
    return max(candidates, key=lambda row: (row["mean_inner_token_f1_exact"], -row["start"])) if candidates else None


def summarize(rows):
    return {name: {"mean": round(statistics.mean(row[name] for row in rows), 4), "std": round(statistics.pstdev(row[name] for row in rows), 4)} for name in METRICS}


def main():
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS:
        raise ValueError("неверный набор train/val")
    gold_sha = file_digest(ROOT / "artifacts/ours33.jsonl")
    folds = grouped_folds(pairs, 5)
    records = {}
    for outer in range(1, 6):
        for inner in range(1, 6):
            if outer == inner:
                continue
            path = ROOT / f"artifacts/nested38_outer_{outer}_inner_{inner}.json"
            if path.exists():
                records[outer, inner] = json.loads(path.read_text(encoding="utf-8"))
    output = {"status": "неполный прогон", "jobs_expected": 20, "jobs_available": len(records), "jobs_complete": sum(row.get("status") == "готово" for row in records.values()), "gold_train_val_sha256": gold_sha, "selection": "для каждого внешнего фолда максимум среднего token F1 по 4 внутренним; оба FPR <= 0.40 в каждом внутреннем; без доступа к внешним меткам при выборе", "warning": "Повторное использование внешних фолдов после предыдущих экспериментов; независимого подтверждения нет. Худший FPR на внутренних фолдах не гарантирует FPR на внешних.", "folds": []}
    target = ROOT / "artifacts/nested38_summary.json"
    if len(records) == 20 and all(row.get("status") == "готово" for row in records.values()):
        shas = {row["git_sha"] for row in records.values()}
        if len(shas) != 1 or any(row["gold_train_val_sha256"] != gold_sha or row["fold_ids_sha256"] != [id_digest(fold) for fold in folds] for row in records.values()):
            raise ValueError("данные или версия разных заданий не совпадают")
        for outer in range(1, 6):
            group = [records[outer, inner] for inner in range(1, 6) if inner != outer]
            choice = select_worst([row["inner_grid"] for row in group])
            if choice is None:
                output["status"] = f"STOP: нет допустимого порога во внутренних фолдах для внешнего {outer}"
                break
            primary = records[outer, target_inner(outer - 1) + 1]
            if not primary["outer_primary"] or [seed["seed"] for seed in primary["seeds"]] != list(range(5)):
                raise ValueError("неверные основные модели")
            held = folds[outer - 1]
            probabilities = primary["outer_predictions"]
            if set(probabilities) != {row["pair_id"] for row in held}:
                raise ValueError("отсутствуют вероятности внешнего фолда")
            cache = {row["pair_id"]: {"answer": row["model_output"], "pieces": probabilities[row["pair_id"]]} for row in held}
            actual = evaluate(held, cache, {row["pair_id"]: choice["start"] for row in held})
            verify_fast(held, cache, choice["start"], actual)
            output["folds"].append({"outer_fold": outer, "primary_inner_fold": primary["inner_fold"], "threshold": choice, "metrics": actual, "inner_fprs": [{"inner_fold": row["inner_fold"], "fpr_pair": row["inner_grid"][GRID.index(choice["start"])]["fpr_pair"], "fpr_token": row["inner_grid"][GRID.index(choice["start"])]["fpr_token"]} for row in group]})
        if len(output["folds"]) == 5:
            values = [row["metrics"] for row in output["folds"]]
            output.update({"status": "готово", "git_sha": shas.pop(), "fold_ids_sha256": [id_digest(fold) for fold in folds], "summary": summarize(values), "max_outer_fpr_pair": max(row["fpr_pair"] for row in values), "max_outer_fpr_token": max(row["fpr_token"] for row in values), "outer_folds_fpr_pair_over_040": sum(row["fpr_pair"] > 0.4 for row in values), "outer_folds_fpr_token_over_040": sum(row["fpr_token"] > 0.4 for row in values)})
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0 if output["status"] == "готово" else 2


if __name__ == "__main__":
    raise SystemExit(main())
