"""Стадия 1: пересчёт метрик при новом правиле отбора порога.

Читает только закоммиченные артефакты (совокупные счётчики внутренних фолдов
и внешние вероятности). Метки внешних фолдов используются ИСКЛЮЧИТЕЛЬНО для
оценки, никогда для выбора.

Правила отбора (все — только по внутренним фолдам):
  max     — как в tools/collect_nested38.py:26 (FPR <= 0.40 на КАЖДОМ из 4 внутренних)
  primary — только основной внутренний фолд
  pooled  — объединённые счётчики 4 внутренних (основное правило стадии 1)
  lofo    — единый порог, выбранный по внутренним фолдам остальных 9

Ограничение: gate и нормировка шкалы НЕ пересчитываются здесь, потому что
внутренние вероятности не сохранены — сохранены только агрегаты inner_grid,
посчитанные при GATE = 0.45. Их пересчёт требует нового инференса (стадия 3),
и это записано в artifacts/decode52_request.json.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, read_jsonl, token_positions
from spanver.snap import snapped
from tools.sweep_ragtruth30 import compute, prepare

GRID = tuple(round(number / 100, 2) for number in range(10, 96))
FPR_LIMIT = 0.40
METRICS = ("token_precision", "token_recall", "token_f1", "fpr_token", "span_exact_f1",
           "span_overlap_f1", "span_iou_f1", "pair_f1", "fpr_pair", "type_accuracy")


def inner_groups(records, outer, fold_count):
    """Четыре заранее заданных внутренних фолда для данного внешнего."""
    return [records[outer, ((outer - 1 + offset) % fold_count) + 1]["inner_grid"] for offset in range(1, 5)]


def pick_max(groups):
    best = None
    for index, threshold in enumerate(GRID):
        rows = [grid[index] for grid in groups]
        if any(not row["support"]["negative_pairs"] or not row["support"]["negative_tokens"] for row in rows):
            continue
        if max(row["fpr_token"] for row in rows) <= FPR_LIMIT and max(row["fpr_pair"] for row in rows) <= FPR_LIMIT:
            score = statistics.mean(row["token_f1_exact"] for row in rows)
            if best is None or (score, -threshold) > (best[0], -best[1]):
                best = (score, threshold)
    return best[1] if best else None


def pick_primary(groups):
    best = None
    for index, threshold in enumerate(GRID):
        row = groups[0][index]
        if not row["support"]["negative_pairs"] or not row["support"]["negative_tokens"]:
            continue
        if row["fpr_token"] <= FPR_LIMIT and row["fpr_pair"] <= FPR_LIMIT:
            if best is None or (row["token_f1_exact"], -threshold) > (best[0], -best[1]):
                best = (row["token_f1_exact"], threshold)
    return best[1] if best else None


def pooled_score(rows):
    support = [row["support"] for row in rows]
    tp = sum(item["token_tp"] for item in support)
    fp = sum(item["token_fp"] for item in support)
    fn = sum(item["token_fn"] for item in support)
    negatives = sum(item["negative_tokens"] for item in support)
    negative_pairs = sum(item["negative_pairs"] for item in support)
    flagged = sum(item["flagged_negative_tokens"] for item in support)
    flagged_pairs = sum(item["flagged_negative_pairs"] for item in support)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "fpr_token": flagged / negatives if negatives else None,
            "fpr_pair": flagged_pairs / negative_pairs if negative_pairs else None}


def pick_pooled(groups):
    best = None
    for index, threshold in enumerate(GRID):
        rows = [grid[index] for grid in groups]
        if any(not row["support"]["negative_pairs"] or not row["support"]["negative_tokens"] for row in rows):
            continue
        value = pooled_score(rows)
        if value["fpr_token"] <= FPR_LIMIT and value["fpr_pair"] <= FPR_LIMIT:
            if best is None or (value["f1"], -threshold) > (best[0], -best[1]):
                best = (value["f1"], threshold)
    return best[1] if best else None


def pick_lofo(records, outer, fold_count):
    """Единый порог по внутренним фолдам ВСЕХ внешних, кроме данного."""
    best = None
    for index, threshold in enumerate(GRID):
        rows = [record["inner_grid"][index] for key, record in records.items() if key[0] != outer]
        if any(not row["support"]["negative_pairs"] or not row["support"]["negative_tokens"] for row in rows):
            continue
        value = pooled_score(rows)
        if value["fpr_token"] <= FPR_LIMIT and value["fpr_pair"] <= FPR_LIMIT:
            if best is None or value["f1"] > best[0]:
                best = (value["f1"], threshold)
    return best[1] if best else None


def evaluate(pairs, raw, threshold, gate):
    cache = {pair_id: {"pieces": pieces} for pair_id, pieces in raw.items()}
    result = compute(prepare(pairs, cache), threshold, threshold / 2, gate)
    support = result["support"]
    return {
        "token_precision": result["token"]["precision"],
        "token_recall": result["token"]["recall"],
        "token_f1": result["token_f1_exact"],
        "fpr_token": result["fpr_token"],
        "fpr_pair": result["fpr_pair"],
        "pair_f1": result["pair"]["f1"],
        "span_exact_f1": None,
        "span_overlap_f1": None,
        "span_iou_f1": None,
        "type_accuracy": None,
        "support": support,
    }


def exact_overlap(pairs, raw, threshold, gate):
    """exact/overlap/IoU-0.5 span F1 и точность типа — по той же предсказанной маске."""
    from tools.sweep_ragtruth30 import predict_spans
    from spanver.snap import boundaries

    from tools.diagnose46 import without_snap

    exact = overlap = iou_hit = predicted_spans = gold_spans = 0
    correct = matched = 0
    for pair in pairs:
        answer = pair["model_output"]
        pieces = raw[pair["pair_id"]]
        row = {"answer": answer, "pieces": pieces, "gate_threshold": gate,
               "start_threshold": threshold, "extend_threshold": threshold / 2}
        spans = snapped(without_snap(row), answer, "expand")
        gold = pair["spans"]
        predicted_spans += len(spans)
        gold_spans += len(gold)
        used = set()
        for span in spans:
            left, right = span["start"], span["end"]
            best_iou, best_index = 0.0, None
            for index, truth in enumerate(gold):
                if index in used:
                    continue
                intersection = min(right, truth["end"]) - max(left, truth["start"]) + 1
                if intersection <= 0:
                    continue
                union = max(right, truth["end"]) - min(left, truth["start"]) + 1
                value = intersection / union
                if value > best_iou:
                    best_iou, best_index = value, index
            if best_index is None:
                continue
            if best_iou >= 0.5:
                used.add(best_index)
                iou_hit += 1
                matched += 1
                if gold[best_index].get("type") == span.get("type"):
                    correct += 1
                if best_iou == 1.0:
                    exact += 1
                overlap += 1
    def f1(tp):
        return 2 * tp / (predicted_spans + gold_spans) if predicted_spans + gold_spans else 0.0
    return {"span_exact_f1": f1(exact), "span_overlap_f1": f1(overlap), "span_iou_f1": f1(iou_hit),
            "type_accuracy": correct / matched if matched else 0.0,
            "span_counts": {"exact_tp": exact, "overlap_tp": overlap, "iou_tp": iou_hit,
                            "predicted": predicted_spans, "gold": gold_spans}}


def baselines(pairs):
    gold_tokens = negative_tokens = 0
    positive_pairs = spans = whole_hit = 0
    for pair in pairs:
        answer = pair["model_output"]
        positions = token_positions(answer)
        gold = covered_tokens(answer, [(s["start"], s["end"]) for s in pair["spans"]])
        gold_tokens += len(gold)
        negative_tokens += len(positions) - len(gold)
        spans += len(pair["spans"])
        if pair["spans"]:
            positive_pairs += 1
            whole = (0, len(answer) - 1)
            intersection = min(whole[1], max(s["end"] for s in pair["spans"])) - max(0, min(s["start"] for s in pair["spans"])) + 1
            if any(min(whole[1], s["end"]) - max(0, s["start"]) + 1 > 0 for s in pair["spans"]):
                union_span = max(whole[1], max(s["end"] for s in pair["spans"])) - min(0, min(s["start"] for s in pair["spans"])) + 1
                if any((min(whole[1], s["end"]) - max(0, s["start"]) + 1) / max(1, union_span) >= 0.5 for s in pair["spans"]):
                    whole_hit += 1
    count = len(pairs)
    return {
        "all_positive_token_f1": 2 * gold_tokens / (2 * gold_tokens + negative_tokens),
        "all_positive_token_precision": gold_tokens / (gold_tokens + negative_tokens),
        "all_positive_pair_f1": 2 * positive_pairs / (positive_pairs + count),
        "whole_answer_span_iou_f1": 2 * whole_hit / (count + spans),
        "gold_tokens": gold_tokens,
        "negative_tokens": negative_tokens,
        "positive_pairs": positive_pairs,
        "pairs": count,
        "spans": spans,
    }


def components(rows):
    """Связные компоненты пар по общим doc_id — единица бутстрэпа."""
    from tools.preflight_13 import our_keys

    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for key in our_keys(row):
            if key in seen:
                left, right = root(index), root(seen[key])
                if left != right:
                    parent[left] = right
            else:
                seen[key] = index
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(root(index), []).append(row)
    return list(groups.values())


def bootstrap(groups, repeats=10000, seed=48):
    rng = random.Random(seed)
    f1_values, fpr_values = [], []
    for _ in range(repeats):
        sample = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        tp = fp = fn = negatives = flagged = 0
        for group in sample:
            for row in group:
                support = row["support"]
                tp += support["token_tp"]
                fp += support["token_fp"]
                fn += support["token_fn"]
                negatives += support["negative_tokens"]
                flagged += support["flagged_negative_tokens"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        fpr_values.append(flagged / negatives if negatives else 0.0)
    f1_values.sort()
    fpr_values.sort()
    low, high = int(0.025 * repeats), int(0.975 * repeats)
    return {"resamples": repeats, "seed": seed, "components": len(groups),
            "token_f1_ci95": [f1_values[low], f1_values[high]],
            "token_f1_width": f1_values[high] - f1_values[low],
            "fpr_token_ci95": [fpr_values[low], fpr_values[high]]}


def run_variant(name, records, outer_pairs, fold_count, gate, picker):
    from tools.diagnose46 import without_snap
    folds = []
    all_rows = []
    for outer in range(1, fold_count + 1):
        record = records[outer, outer % fold_count + 1]
        if not record["primary"]:
            raise ValueError("нет внешних вероятностей основного внутреннего фолда")
        pairs = outer_pairs[outer]
        if set(record["outer_predictions"]) != {pair["pair_id"] for pair in pairs}:
            raise ValueError("состав пар не совпадает с сохранённым фолдом")
        threshold = picker(outer, records, fold_count)
        if threshold is None:
            raise ValueError(f"{name}: нет допустимого порога для внешнего фолда {outer}")
        value = evaluate(pairs, record["outer_predictions"], threshold, gate)
        extra = exact_overlap(pairs, record["outer_predictions"], threshold, gate)
        value.update(extra)
        value["fold"] = outer
        value["threshold"] = threshold
        value["extend"] = threshold / 2
        folds.append({key: item for key, item in value.items() if key not in ("support", "span_counts")})
        all_rows.append(value)
    totals = [row["support"] for row in all_rows]
    tp = sum(item["token_tp"] for item in totals)
    fp = sum(item["token_fp"] for item in totals)
    fn = sum(item["token_fn"] for item in totals)
    negatives = sum(item["negative_tokens"] for item in totals)
    flagged = sum(item["flagged_negative_tokens"] for item in totals)
    negative_pairs = sum(item["negative_pairs"] for item in totals)
    flagged_pairs = sum(item["flagged_negative_pairs"] for item in totals)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    per_pair_rows = []
    for outer in range(1, fold_count + 1):
        record = records[outer, outer % fold_count + 1]
        threshold = folds[outer - 1]["threshold"]
        for pair in outer_pairs[outer]:
            answer = pair["model_output"]
            pieces = record["outer_predictions"][pair["pair_id"]]
            row = {"answer": answer, "pieces": pieces, "gate_threshold": gate,
                   "start_threshold": threshold, "extend_threshold": threshold / 2}
            spans = snapped(without_snap(row), answer, "expand")
            gold = covered_tokens(answer, [(s["start"], s["end"]) for s in pair["spans"]])
            predicted = covered_tokens(answer, [(s["start"], s["end"]) for s in spans])
            per_pair_rows.append({**pair, "support": {
                "token_tp": len(gold & predicted), "token_fp": len(predicted - gold), "token_fn": len(gold - predicted),
                "negative_tokens": len(token_positions(answer)) - len(gold),
                "flagged_negative_tokens": len(predicted - gold),
            }})
    counts = {key: sum(row["span_counts"][key] for row in all_rows)
              for key in ("exact_tp", "overlap_tp", "iou_tp", "predicted", "gold")}
    def span_f1(key):
        denominator = counts["predicted"] + counts["gold"]
        return 2 * counts[key] / denominator if denominator else 0.0
    return {
        "rule": name,
        "folds": folds,
        "pooled": {
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr_token": flagged / negatives if negatives else 0.0,
            "fpr_pair": flagged_pairs / negative_pairs if negative_pairs else 0.0,
            "span_exact_f1": span_f1("exact_tp"),
            "span_overlap_f1": span_f1("overlap_tp"),
            "span_iou_f1": span_f1("iou_tp"),
            "span_counts": counts,
        },
        "macro": {key: statistics.mean(row[key] for row in folds if row[key] is not None) for key in METRICS},
        "std": {key: statistics.pstdev([row[key] for row in folds if row[key] is not None]) for key in METRICS},
        "min_fold": {key: min(row[key] for row in folds if row[key] is not None) for key in METRICS},
        "max_fold": {key: max(row[key] for row in folds if row[key] is not None) for key in METRICS},
        "folds_over_fpr": sum(1 for row in folds if row["fpr_token"] > FPR_LIMIT),
        "bootstrap": bootstrap(components(per_pair_rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/decode52.json")
    parser.add_argument("--pairs", default="artifacts/ours33.jsonl")
    parser.add_argument("--records", default="ablation51_ten", help="префикс файлов записей фолдов")
    parser.add_argument("--fold-count", type=int, default=0, help="0 — оба набора (10 и 5)")
    args = parser.parse_args()
    pairs = read_jsonl(ROOT / args.pairs)
    if len(pairs) != 699:
        raise ValueError("ожидались 699 пар train/val")
    orders = {
        10: ROOT / f"artifacts/{args.records}_summary.json",
        5: ROOT / "artifacts/ablation47_v5_summary.json",
    }
    if args.fold_count:
        orders = {args.fold_count: orders[args.fold_count]}
    report = {"status": "в процессе", "scope": "699 out-of-fold пар train/val, адаптивный выбор на внешних фолдах, не независимая оценка",
              "test301_evaluated": False, "external_data_used": False, "fpr_limit": FPR_LIMIT,
              "gate": 0.45, "grid_start": list(GRID), "variants": {}}
    for fold_count, summary_path in orders.items():
        folds = __import__("tools.train_base33", fromlist=["grouped_folds"]).grouped_folds(pairs, fold_count)
        outer_pairs = {index + 1: bucket for index, bucket in enumerate(folds)}
        records = {}
        for outer in range(1, fold_count + 1):
            for offset in range(1, 5):
                inner = ((outer - 1 + offset) % fold_count) + 1
                path = ROOT / f"artifacts/{args.records}_outer{outer}_inner{inner}.json" if fold_count == 10 else \
                    ROOT / f"artifacts/ablation47_v5_outer{outer}_inner{inner}.json"
                records[outer, inner] = json.loads(path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["status"] != "готово":
            raise ValueError("сводка не завершена")
        slot = {"fold_count": fold_count, "pairs_per_fold": [len(outer_pairs[i]) for i in range(1, fold_count + 1)],
                "repository_rule": summary["summary"], "baselines": baselines(pairs), "rules": {}}
        pickers = {
            "max": lambda outer, rec, n: pick_max(inner_groups(rec, outer, n)),
            "primary": lambda outer, rec, n: pick_primary(inner_groups(rec, outer, n)),
            "pooled": lambda outer, rec, n: pick_pooled(inner_groups(rec, outer, n)),
            "lofo": lambda outer, rec, n: pick_lofo(rec, outer, n),
        }
        for name, picker in pickers.items():
            slot["rules"][name] = run_variant(name, records, outer_pairs, fold_count, 0.45, picker)
        report["variants"][str(fold_count)] = slot
    report["status"] = "готово"
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for fold_count in [str(key) for key in orders]:
        print(f"=== {fold_count} фолдов ===")
        for name, value in report["variants"][fold_count]["rules"].items():
            print(f"{name:8s} pooled F1 {value['pooled']['token_f1']:.4f} P {value['pooled']['token_precision']:.4f} "
                  f"R {value['pooled']['token_recall']:.4f} FPR {value['pooled']['fpr_token']:.4f} "
                  f"macro {value['macro']['token_f1']:.4f} std {value['std']['token_f1']:.4f} "
                  f"min {value['min_fold']['token_f1']:.4f} CI [{value['bootstrap']['token_f1_ci95'][0]:.4f};{value['bootstrap']['token_f1_ci95'][1]:.4f}]")
    print(f"отчёт: {args.out}")


if __name__ == "__main__":
    main()
