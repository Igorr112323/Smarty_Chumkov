import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, lexfeat
from spanver.evaluate import metrics, read_jsonl
from spanver.snap import snapped
from tools.preflight_13 import id_digest, our_keys
from tools.prepare_ours33 import EXPECTED_IDS
from tools.sweep_hyst import flags_with
from tools.sweep_ragtruth30 import compute, prepare, within_fpr_limits
from tools.train_base33 import grouped_folds, train
from tools.train_cv_feat import token_fpr
from tools.train_scheme13_full import save_json

GRID = tuple(round(index * 0.05, 2) for index in range(2, 20))
GATE = 0.45
EPOCHS = 3
METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")


def partition(folds, index):
    inner_index = (index + 1) % len(folds)
    outer = folds[index]
    inner = folds[inner_index]
    training = [pair for position, bucket in enumerate(folds) if position not in (index, inner_index) for pair in bucket]
    sets = [{key for row in bucket for key in our_keys(row)} for bucket in (training, inner, outer)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("пересечение документов между обучением, внутренним и внешним фолдами")
    return training, inner, outer, inner_index


def choose(inner, cache):
    prepared = prepare(inner, cache)
    rows = [compute(prepared, threshold, threshold / 2, GATE) for threshold in GRID]
    if any(not row["support"]["negative_pairs"] or not row["support"]["negative_tokens"] for row in rows):
        raise ValueError("невозможно оценить оба FPR без отрицательной поддержки")
    eligible = [row for row in rows if within_fpr_limits(row)]
    if not eligible:
        return None, rows
    return max(eligible, key=lambda row: (row["token_f1_exact"], -row["start"])), rows


def predict(value, start):
    pieces = value["pieces"]
    if max((piece[2] for piece in pieces), default=0.0) < GATE:
        return []
    flags = flags_with(pieces, start, start / 2, 0.0)
    return snapped(decode.spans_from_flags(value["answer"], pieces, flags, 2), value["answer"], "expand")


def evaluate(pairs, cache, thresholds):
    predictions = {row["pair_id"]: predict(cache[row["pair_id"]], thresholds[row["pair_id"]]) for row in pairs}
    def predictor(row):
        return predictions[row["pair_id"]]
    result = metrics(pairs, predictor)
    summary = {f"{level}_{name}": result[f"{level}_level"][name] for level in ("token", "span") for name in ("precision", "recall", "f1")}
    summary.update({"pair_f1": result["pair_level"]["f1"], "fpr_pair": result["fpr"], "fpr_token": token_fpr(pairs, predictor), "type_accuracy": result["type_accuracy"], "type_correct": result["type_correct"], "matched_spans": result["span_level"]["tp"], "predicted_pairs": sum(bool(value) for value in predictions.values()), "predicted_spans": sum(map(len, predictions.values()))})
    return summary


def verify_fast(pairs, cache, threshold, expected):
    fast = compute(prepare(pairs, cache), threshold, threshold / 2, GATE)
    for level in ("token", "span"):
        for name in ("precision", "recall", "f1"):
            if fast[level][name] != expected[f"{level}_{name}"]:
                raise ValueError("быстрые метрики не совпали с evaluator")
    if fast["pair"]["f1"] != expected["pair_f1"] or fast["fpr_pair"] != expected["fpr_pair"] or fast["fpr_token"] != expected["fpr_token"]:
        raise ValueError("метрики пар или FPR не совпали с evaluator")


def averaged(pair, caches, selected):
    entries = [caches[index][pair["pair_id"]] for index in selected]
    reference = entries[0]
    offsets = [(piece[0], piece[1]) for piece in reference["pieces"]]
    for item in entries[1:]:
        if item["answer"] != reference["answer"] or [(p[0], p[1]) for p in item["pieces"]] != offsets:
            raise ValueError("не совпадают смещения токенов между моделями")
    return {"answer": reference["answer"], "pieces": [[left, right, sum(item["pieces"][index][2] for item in entries) / len(entries)] for index, (left, right) in enumerate(offsets)]}


def mean_std(rows):
    return {metric: {"mean": round(statistics.mean(row[metric] for row in rows), 4), "std": round(statistics.pstdev(row[metric] for row in rows), 4)} for metric in METRICS}


def probabilities(tokenizer, model, pairs):
    cache = {}
    for begin in range(0, len(pairs), 16):
        cache.update(lexfeat.batched_probabilities(tokenizer, model, pairs[begin:begin + 16], top=4, batch=4))
    if len(cache) != len(pairs):
        raise ValueError("не все пары получили вероятности")
    return cache


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    args = parser.parse_args()
    clock = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS or len({row["pair_id"] for row in pairs}) != 699:
        raise ValueError("ожидались все 699 наших пар train/val")
    folds = grouped_folds(pairs, 5)
    index = args.fold - 1
    training, inner, outer, inner_index = partition(folds, index)
    report = {"status": "в процессе", "fold": args.fold, "inner_fold": inner_index + 1, "train_pairs": len(training), "inner_pairs": len(inner), "outer_pairs": len(outer), "fold_sizes": [len(fold) for fold in folds], "fold_ids_sha256": [id_digest(fold) for fold in folds], "train_id_sha256": id_digest(training), "inner_id_sha256": id_digest(inner), "outer_id_sha256": id_digest(outer), "seeds": [], "config": {"pairs": 699, "outer_folds": 5, "train_folds": 3, "inner_folds": 1, "epochs": EPOCHS, "model": "rubert-tiny2", "features": 5, "loss": "focal gamma=2 + pair loss 0.3", "class_weight": 1.0, "grid": list(GRID), "extend": "start/2", "selection": "max exact token F1 при FPR_pair <= 0.40 и FPR_token <= 0.40; tie -> min start; без fallback", "top_chunks": 4, "pair_gate": GATE, "snap": "expand", "std": "population ddof=0", "external_data": False, "test301": False}}
    target = ROOT / f"artifacts/nested36_fold_{args.fold}.json"
    save_json(target, report)
    outer_caches = []
    threshold = None
    for seed in (0, 1, 2):
        started = time.perf_counter()
        print(f"фолд {args.fold}/5, seed {seed}: обучение {len(training)}, внутренняя {len(inner)}, внешняя {len(outer)}", flush=True)
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(seed)
        history = train(torch, model, tokenizer, training, EPOCHS, seed * 1000)
        if seed == 0:
            inner_cache = probabilities(tokenizer, model, inner)
            selected, grid = choose(inner, inner_cache)
            report["inner_grid"] = grid
            if selected is None:
                report["status"] = "STOP: нет допустимого порога на внутреннем фолде"
                save_json(target, report)
                print(report["status"], flush=True)
                return 2
            threshold = selected["start"]
            inner_metric = evaluate(inner, inner_cache, {item["pair_id"]: threshold for item in inner})
            verify_fast(inner, inner_cache, threshold, inner_metric)
            report["threshold"] = {"start": threshold, "extend": threshold / 2, "inner": selected, "inner_metrics": inner_metric, "eligible_count": sum(within_fpr_limits(item) for item in grid)}
            del inner_cache
        outer_cache = probabilities(tokenizer, model, outer)
        thresholds = {item["pair_id"]: threshold for item in outer}
        result = evaluate(outer, outer_cache, thresholds)
        verify_fast(outer, outer_cache, threshold, result)
        outer_caches.append(outer_cache)
        report["seeds"].append({"seed": seed, "shuffle_seed": seed * 1000, "history": history, "outer_metrics": result, "seconds": round(time.perf_counter() - started, 3)})
        report["elapsed_s"] = round(time.perf_counter() - clock, 3)
        save_json(target, report)
        print(f"фолд {args.fold}, seed {seed}: порог {threshold}/{threshold / 2}, внешний token F1 {result['token_f1']}, span F1 {result['span_f1']}", flush=True)
        del model, tokenizer
    blended = {pair["pair_id"]: averaged(pair, outer_caches, (0, 1, 2)) for pair in outer}
    report["ensemble"] = {"models_per_pair": 3, "threshold_from": "seed 0, внутренний фолд", "outer_metrics": evaluate(outer, blended, thresholds), "averaging": "вероятности субтокенов после max по чанкам"}
    verify_fast(outer, blended, threshold, report["ensemble"]["outer_metrics"])
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - clock, 3)
    save_json(target, report)
    print(f"фолд {args.fold}/5 готов: ансамбль token F1 {report['ensemble']['outer_metrics']['token_f1']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
