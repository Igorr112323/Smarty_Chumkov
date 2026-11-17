"""ШАГ A/C: сверка нового прогона с 51 и проверка критерия стадии 3.

Сравнение идёт на одних и тех же внешних фолдах, поэтому в таблицу попадают
только те фолды, для которых есть и 51, и 52. AUC считается по сохранённым в
отчёте 52 вероятностям (outer_predictions), чтобы не запускать инференс снова.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.train_base33 import grouped_folds


def auc(positives, negatives, seed=48, limit=4000):
    if not positives or not negatives:
        return None
    generator = random.Random(seed)
    sample_pos = generator.sample(positives, min(limit, len(positives)))
    sample_neg = generator.sample(negatives, min(limit, len(negatives)))
    wins = ties = 0
    for positive in sample_pos:
        for negative in sample_neg:
            if positive > negative:
                wins += 1
            elif positive == negative:
                ties += 1
    return (wins + 0.5 * ties) / (len(sample_pos) * len(sample_neg))


def char_auc(pairs, pieces_by_id):
    positives, negatives = [], []
    for pair in pairs:
        mask = set()
        for span in pair["spans"]:
            mask |= set(range(span["start"], span["end"] + 1))
        for start, end, score in pieces_by_id.get(pair["pair_id"], []):
            chars = set(range(start, end + 1))
            share = len(chars & mask) / max(1, len(chars))
            (positives if share >= 0.5 else negatives).append(score)
    return auc(positives, negatives), len(positives), len(negatives)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/compare52.json")
    args = parser.parse_args()
    pairs = {row["pair_id"]: row for row in read_jsonl(ROOT / "artifacts/ours33.jsonl")}
    folds = grouped_folds(list(pairs.values()), 5)
    reference = json.loads((ROOT / "artifacts/decode52.json").read_text(encoding="utf-8"))["variants"]["5"]["rules"]

    rows = []
    for outer in (1, 2):
        path = ROOT / f"artifacts/ablation52_outer{outer}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        held = folds[outer - 1]
        held_pairs = [pairs[row["pair_id"]] for row in held]
        pieces = report["outer_predictions"]
        value, positives, negatives = char_auc(held_pairs, pieces)
        new = report["outer_at_selected"]
        old_max = reference["max"]["folds"][outer - 1]
        old_pooled = reference["pooled"]["folds"][outer - 1]
        rows.append({
            "outer": outer,
            "pairs": len(held_pairs),
            "auc_chars_52": value,
            "auc_positives": positives,
            "auc_negatives": negatives,
            "new_token_f1": new["token_f1_exact"],
            "new_fpr_token": new["fpr_token"],
            "new_fpr_pair": new["fpr_pair"],
            "new_span_f1": new["span_f1_exact"],
            "new_threshold": report["threshold_point"]["start"],
            "old51_max_token_f1": old_max["token_f1"],
            "old51_max_fpr_token": old_max["fpr_token"],
            "old51_pooled_token_f1": old_pooled["token_f1"],
            "old51_pooled_fpr_token": old_pooled["fpr_token"],
            "old51_pooled_span_exact": old_pooled["span_exact_f1"],
        })
    report = {
        "status": "готово",
        "scope": "фолды 1 и 2, 269 пар, схемы 51 и 52 сравниваются на одних и тех же парах",
        "baseline_auc_chars_51": 0.6538235435724603,
        "all_positive_token_f1": 0.5001320461281141,
        "rows": rows,
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
