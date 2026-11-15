import argparse
import bisect
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, match_spans, token_positions
from spanver.snap import boundaries
from tools.preflight_13 import digest, id_digest
from tools.train_scheme13_a1 import read_pairs
from tools.train_scheme13_full import load_state, save_json

CHECKPOINTS = {"bce": "artifacts/checkpoints/stage_a_bce_w11_e1", "focal": "artifacts/checkpoints/stage_a_full_e2"}
GRID = tuple(round(index * 0.05, 2) for index in range(1, 20))
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def predict_spans(answer, pieces, start_threshold, extend_threshold, edges):
    active = False
    run_start = run_end = None
    raw = []
    for left, right, score in pieces:
        flag = score >= (extend_threshold if active else start_threshold)
        if flag:
            if run_start is None:
                run_start = left
            run_end = right - 1
            active = True
        else:
            if run_start is not None:
                raw.append((run_start, run_end))
            run_start = run_end = None
            active = False
    if run_start is not None:
        raw.append((run_start, run_end))
    result = []
    for left, right in raw:
        if len(answer[left:right + 1]) < 2:
            continue
        before = bisect.bisect_right(edges, left) - 1
        after = bisect.bisect_left(edges, right + 1)
        span_left = edges[before] if before >= 0 else 0
        span_right = edges[after] - 1 if after < len(edges) else len(answer) - 1
        result.append({"start": max(0, span_left), "end": min(len(answer) - 1, max(span_left, span_right))})
    return result


def prepare(pairs, cache):
    result = []
    for pair in pairs:
        answer = pair["model_output"]
        spans = pair.get("spans") or []
        positions = token_positions(answer)
        labels = covered_tokens(answer, [(span["start"], span["end"]) for span in spans])
        mask = sum(1 << index for index in labels)
        result.append({"pair_id": pair["pair_id"], "answer": answer, "gold": spans, "pieces": cache[pair["pair_id"]]["pieces"], "edges": boundaries(answer), "starts": [item[0] for item in positions], "ends": [item[1] for item in positions], "gold_mask": mask, "all_mask": (1 << len(positions)) - 1})
    return result


def compute(prepared, start, extend, gate=None):
    tp = fp = fn = pair_tp = pair_fp = pair_fn = negative_pairs = negative_flagged = 0
    token_tp = token_fp = token_fn = negative_tokens = negative_tokens_flagged = 0
    for row in prepared:
        pieces = row["pieces"]
        if gate is not None and max((piece[2] for piece in pieces), default=0.0) < gate:
            predicted = []
        else:
            predicted = predict_spans(row["answer"], pieces, start, extend, row["edges"])
        matched, _, _ = match_spans(predicted, row["gold"])
        tp += matched
        fp += len(predicted) - matched
        fn += len(row["gold"]) - matched
        if row["gold"]:
            if matched:
                pair_tp += 1
            else:
                pair_fn += 1
        else:
            negative_pairs += 1
            if predicted:
                pair_fp += 1
                negative_flagged += 1
        predicted_mask = 0
        for span in predicted:
            first = bisect.bisect_left(row["ends"], span["start"])
            last = bisect.bisect_right(row["starts"], span["end"])
            if last > first:
                predicted_mask |= ((1 << (last - first)) - 1) << first
        gold_mask = row["gold_mask"]
        token_tp += (predicted_mask & gold_mask).bit_count()
        token_fp += (predicted_mask & ~gold_mask).bit_count()
        token_fn += (gold_mask & ~predicted_mask).bit_count()
        negative_tokens += (row["all_mask"] & ~gold_mask).bit_count()
        negative_tokens_flagged += (predicted_mask & ~gold_mask).bit_count()

    def stats(a, b, c):
        precision = a / (a + b) if a + b else 0.0
        recall = a / (a + c) if a + c else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    return {"start": start, "extend": extend, "token": stats(token_tp, token_fp, token_fn), "span": stats(tp, fp, fn), "span_f1_exact": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, "pair": stats(pair_tp, pair_fp, pair_fn), "fpr_pair": round(negative_flagged / negative_pairs, 4) if negative_pairs else 0.0, "fpr_token": round(negative_tokens_flagged / negative_tokens, 4) if negative_tokens else 0.0, "support": {"span_tp": tp, "span_fp": fp, "span_fn": fn, "negative_pairs": negative_pairs, "negative_tokens": negative_tokens, "flagged_negative_pairs": negative_flagged, "flagged_negative_tokens": negative_tokens_flagged}}


def within_fpr_limits(row):
    support = row["support"]
    return support["flagged_negative_pairs"] * 5 <= support["negative_pairs"] * 2 and support["flagged_negative_tokens"] * 5 <= support["negative_tokens"] * 2


def percentile(values, q):
    if not values:
        return None
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def distribution(prepared):
    classes = {"positive": [], "negative": []}
    for row in prepared:
        for left, right, score in row["pieces"]:
            positive = any(max(left, span["start"]) < min(right, span["end"] + 1) for span in row["gold"])
            classes["positive" if positive else "negative"].append(float(score))
    result = {}
    for name, values in classes.items():
        values.sort()
        result[name] = {"count": len(values), "mean": sum(values) / len(values) if values else None, "quantiles": {f"{int(q * 100)}%": percentile(values, q) for q in QUANTILES}}
    positive = classes["positive"]
    negative = classes["negative"]
    if positive and negative:
        concordant = sum(bisect.bisect_left(negative, value) + (bisect.bisect_right(negative, value) - bisect.bisect_left(negative, value)) / 2 for value in positive)
        result["subword_roc_auc"] = concordant / (len(positive) * len(negative))
    return result


def load_predictions(model_name, heldout):
    import torch
    from spanver import lexfeat

    folder = ROOT / CHECKPOINTS[model_name]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    meta = {"model": model_name, "holdout_id_sha256": id_digest(heldout), "parts_sha256": [part["sha256"] for part in manifest["training_state_parts"]], "lexfeat_sha256": digest(ROOT / "spanver/lexfeat.py"), "span_detector_sha256": digest(ROOT / "spanver/span_detector.py"), "top": 4, "batch": 4}
    path = ROOT / f"artifacts/threshold30_{model_name}_scores.jsonl"
    cache = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            first = json.loads(handle.readline())
            if first != {"meta": meta}:
                raise ValueError("кэш относится к другим весам или разбиению")
            for line in handle:
                item = json.loads(line)
                if item["pair_id"] in cache:
                    raise ValueError("повтор pair_id в кэше")
                cache[item["pair_id"]] = {"answer": item["answer"], "pieces": item["pieces"]}
    else:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
    missing = [pair for pair in heldout if pair["pair_id"] not in cache]
    if missing:
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
        model.base.config._attn_implementation = "eager"
        torch.set_num_threads(2)
        state = load_state(folder, manifest["training_state_parts"], torch)
        model.load_state_dict(state["model"], strict=True)
        del state
        model.eval()
        with path.open("a", encoding="utf-8") as handle, torch.no_grad():
            for start in range(0, len(missing), 16):
                group = missing[start:start + 16]
                batch = lexfeat.batched_probabilities(tokenizer, model, group, top=4, batch=4)
                if len(batch) != len(group):
                    raise ValueError("не все пары получили вероятности")
                for pair in group:
                    item = batch[pair["pair_id"]]
                    if item["answer"] != pair["model_output"]:
                        raise ValueError("ответ в кэше не совпадает")
                    handle.write(json.dumps({"pair_id": pair["pair_id"], **item}, ensure_ascii=False) + "\n")
                    cache[pair["pair_id"]] = item
                handle.flush()
                if (start // 16 + 1) % 10 == 0:
                    print(f"{model_name}: инференс {len(cache)}/{len(heldout)}", flush=True)
    ids = {pair["pair_id"] for pair in heldout}
    if set(cache) != ids:
        raise ValueError("состав кэша расходится с holdout")
    return cache, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    _, heldout = read_pairs()
    preflight = json.loads((ROOT / "artifacts/preflight_13.json").read_text(encoding="utf-8"))
    if len(heldout) != 1458 or id_digest(heldout) != preflight["rag_holdout"]["id_sha256"]:
        raise ValueError("разбиение не совпадает")
    cache, meta = load_predictions(args.model, heldout)
    prepared = prepare(heldout, cache)
    report = {"status": "расчёт", "model": args.model, "meta": meta, "samples": len(heldout), "histogram_unit": "субтокены ответа, после объединения чанков по максимуму; разметка по пересечению с gold", "histogram": distribution(prepared), "pair_gate": None, "snap": "expand", "min_chars": 2, "simple": [], "hysteresis": []}
    output = ROOT / f"artifacts/threshold30_{args.model}.json"
    save_json(output, report)
    for mode in ("simple", "hysteresis"):
        for threshold in GRID:
            extend = threshold if mode == "simple" else threshold / 2
            report[mode].append(compute(prepared, threshold, extend))
            report["seconds_so_far"] = round(time.perf_counter() - started, 3)
            save_json(output, report)
            print(f"{args.model} {mode} {threshold:.2f}: span {report[mode][-1]['span']['f1']:.4f}", flush=True)
    report["previous_decoder_now"] = compute(prepared, 0.25, 0.25, gate=0.45)
    for mode in ("simple", "hysteresis"):
        rows = report[mode]
        eligible = [row for row in rows if within_fpr_limits(row)]
        report[mode + "_best_unrestricted"] = max(rows, key=lambda row: (row["span_f1_exact"], -row["start"]))
        report[mode + "_best_fpr_limited"] = max(eligible, key=lambda row: (row["span_f1_exact"], -row["start"])) if eligible else None
    report["status"] = "готово"
    report["seconds"] = round(time.perf_counter() - started, 3)
    save_json(output, report)
    print(json.dumps({"model": args.model, "simple_best": report["simple_best_unrestricted"], "simple_limited": report["simple_best_fpr_limited"], "hysteresis_best": report["hysteresis_best_unrestricted"], "hysteresis_limited": report["hysteresis_best_fpr_limited"], "seconds": report["seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
