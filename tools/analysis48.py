import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, match_spans, token_positions
from spanver.snap import snapped
from tools.diagnose46 import without_snap
from tools.preflight_13 import components, id_digest, our_keys


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def pair_counts(row):
    text, gold, predicted = row["answer"], row["gold_spans"], row["predicted_spans"]
    gt = covered_tokens(text, [(s["start"], s["end"]) for s in gold])
    pt = covered_tokens(text, [(s["start"], s["end"]) for s in predicted])
    matched = match_spans(predicted, gold)[0]
    return [len(gt & pt), len(pt - gt), len(gt - pt), len(token_positions(text)) - len(gt),
            match_spans(predicted, gold, 1.0)[0], match_spans(predicted, gold, 1e-15)[0],
            matched, len(predicted), len(gold), int(bool(gold) and bool(matched)),
            int(not gold and bool(predicted)), int(bool(gold) and not matched)]


def ratio(a, b):
    return a / b if b else 0.0


def measures(c):
    tp, fp, fn, negatives, exact, overlap, iou, pred, gold, pair_tp, pair_fp, pair_fn = c
    return {"token_precision": ratio(tp, tp + fp), "token_recall": ratio(tp, tp + fn), "token_f1": ratio(2 * tp, 2 * tp + fp + fn), "fpr_token": ratio(fp, negatives),
            "span_exact_f1": ratio(2 * exact, pred + gold), "span_overlap_f1": ratio(2 * overlap, pred + gold), "span_iou_f1": ratio(2 * iou, pred + gold),
            "span_precision": ratio(iou, pred), "span_recall": ratio(iou, gold),
            "pair_f1": ratio(2 * pair_tp, 2 * pair_tp + pair_fp + pair_fn), "pair_precision": ratio(pair_tp, pair_tp + pair_fp), "pair_recall": ratio(pair_tp, pair_tp + pair_fn)}


def totals(rows):
    value = [0] * 12
    for row in rows:
        value = add(value, pair_counts(row))
    return value


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def bootstrap(groups, seed=48, repeats=10000):
    blocks = [totals(group)[:4] for group in groups]
    r = random.Random(seed)
    f1, fpr = [], []
    for _ in range(repeats):
        total = [0, 0, 0, 0]
        for _ in range(len(blocks)):
            block = blocks[r.randrange(len(blocks))]
            for i in range(4):
                total[i] += block[i]
        tp, fp, fn, neg = total
        f1.append(ratio(2 * tp, 2 * tp + fp + fn))
        fpr.append(ratio(fp, neg))
    result = {}
    for name, values in (("token_f1", f1), ("fpr_token", fpr)):
        lo, hi = quantile(values, 0.025), quantile(values, 0.975)
        result[name] = {"ci_95": [lo, hi], "width": hi - lo}
    return result


def analyze(variant):
    art = ROOT / "artifacts"
    manifest = load(art / "predictions40_manifest.json")
    summary = load(art / f"ablation47_{variant}_summary.json")
    reference = load(art / "nested38_summary.json")
    folds = []
    for index, source in enumerate(manifest["outputs"]):
        content = (ROOT / source["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError("источник фолда не совпадает по SHA")
        rows = [json.loads(line) for line in content.splitlines() if line]
        if id_digest(rows) != reference["fold_ids_sha256"][index]:
            raise ValueError("идентификаторы фолда не совпали")
        fold, inner = index + 1, (index + 1) % 5 + 1
        record = load(art / f"ablation47_{variant}_outer{fold}_inner{inner}.json")
        if set(record["outer_predictions"]) != {row["pair_id"] for row in rows}:
            raise ValueError("неполные предсказания")
        threshold = summary["folds"][index]["threshold"]
        for row in rows:
            row.update({"pieces": record["outer_predictions"][row["pair_id"]], "gate_threshold": 0.45, "start_threshold": threshold["start"], "extend_threshold": threshold["extend"]})
            row["predicted_spans"] = snapped(without_snap(row), row["answer"], "expand")
        folds.append(rows)
    rows = sum(folds, [])
    if len(rows) != 699 or len({r["pair_id"] for r in rows}) != 699:
        raise ValueError("неверная совокупность")
    docsets = [{d for row in fold for d in row["doc_ids"]} for fold in folds]
    if any(docsets[i] & docsets[j] for i in range(5) for j in range(i + 1, 5)):
        raise ValueError("межфолдовое пересечение doc_id")
    jobs = 0
    for outer in range(5):
        for inner in range(5):
            if outer == inner:
                continue
            record = load(art / f"ablation47_{variant}_outer{outer+1}_inner{inner+1}.json")
            training = [row for i, fold in enumerate(folds) if i not in (outer, inner) for row in fold]
            if record["status"] != "готово" or [seed["seed"] for seed in record["seeds"]] != [0, 1, 2, 3, 4] or record["gold_sha256"] != reference["gold_train_val_sha256"] or record["fold_ids_sha256"] != reference["fold_ids_sha256"] or record["train_id_sha256"] != id_digest(training) or record["inner_id_sha256"] != id_digest(folds[inner]) or record["outer_id_sha256"] != id_digest(folds[outer]):
                raise ValueError(f"протокол задания не совпал {outer+1}/{inner+1}")
            jobs += 1
    groups = components(rows, our_keys)
    fold_metrics = [measures(totals(fold)) for fold in folds]
    macro = {key: statistics.mean(row[key] for row in fold_metrics) for key in fold_metrics[0]}
    pooled = measures(totals(rows))
    official = load(art / f"ablation47_{variant}_pooled.json")
    if abs(pooled["token_f1"] - official["pooled_token_f1_exact"]) > 1e-15 or abs(pooled["fpr_token"] - official["pooled_fpr_token"]) > 1e-15:
        raise ValueError("точные счётчики не совпадают")
    ci = bootstrap(groups)
    for name in ci:
        ci[name]["point"] = pooled[name]
    return {"variant": variant, "scope": "699 CV out-of-fold; адаптивный выбор и фиксированные прогнозы", "macro_exact_from_counts": macro, "pooled_exact_from_counts": pooled, "fold_metrics_exact": fold_metrics, "fold_counts": [totals(fold) for fold in folds], "pooled_counts_order": ["token_tp", "token_fp", "token_fn", "token_negative", "span_exact_tp", "span_overlap_tp", "span_iou_tp", "span_pred", "span_gold", "pair_tp", "pair_fp", "pair_fn"], "pooled_counts": totals(rows), "bootstrap": {"resamples": 10000, "seed": 48, "unit": "связные компоненты по общим doc_ids, с возвращением", "groups": len(groups), "interval": "percentile 2.5% и 97.5%, линейная интерполяция", "metrics": ci, "caveat": "условный интервал по фиксированным out-of-fold прогнозам, не учитывает вариативность обучения и адаптивный выбор"}, "provenance": {"pairs": len(rows), "distinct_pair_ids": len({r['pair_id'] for r in rows}), "distinct_doc_ids": len(set().union(*docsets)), "train_jobs_checked": jobs, "models_checked": jobs * 5, "all_train_inner_outer_id_hashes_match": True, "cross_fold_doc_id_overlap": 0, "extra_original_list_available": False, "extra_pair_or_document_overlap_verified": False, "note": "Все 100 обучений используют только три из пяти фолдов 699 train/val; фактический список extra 200 не найден, поэтому адресно проверить его pair_id/doc_id невозможно. Отдельная проверка и оценка extra не выполнялись по решению пользователя."}, "test301_evaluated": False, "external_data_used": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v4", "v5"), required=True)
    args = parser.parse_args()
    output = analyze(args.variant)
    path = ROOT / f"artifacts/analysis48_{args.variant}.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
