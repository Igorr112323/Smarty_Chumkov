import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from spanver.evaluate import evaluate_pairs, read_jsonl
from spanver.model_features import (
    ATTENTION_FEATURES,
    HEAD_FEATURES,
    MAX_MODEL_CANDIDATES,
    MODEL_SEED,
    MODEL_THREADS,
    TOP_SEGMENTS,
    SEGMENT_TOKENS,
    cache_key,
    load_model,
    pair_features,
    read_cache,
    write_cache,
)
from spanver.model_score import (
    COMBINE,
    THRESHOLD_GRID,
    save_head,
    TRAIN_LIMIT,
    ai_share,
    label_rows,
    make_predictor,
    save_table,
    scan_threshold,
    score_rows,
    train_head,
    training_subset,
)
from spanver.eval_report import package_info
from spanver.evaluate import metrics
from spanver.splits import load_splits, ordered_pairs

GOLD_SHA256 = "fddd62ce05baf028b64cb6db81b89053bdc5444a540c6e18e34e318577c8b32b"
CALIB_SHA256 = "afbfd35f85731e8acc9cb0a584974f481c4ac6e8b1fe5c0ceb368d0e9fdef989"
STRESS_CHARS = 2000000
STRESS_RUNS = 3
PROGRESS_STEP = 20


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fail(message):
    print(message)
    raise SystemExit(2)


def load_pairs(data, extra):
    pairs = read_jsonl(Path(data))
    extra_path = Path(extra)
    if extra_path.exists():
        pairs.extend(read_jsonl(extra_path))
    return pairs


def stress_document(pairs, chars=STRESS_CHARS):
    parts = []
    total = 0
    for pair in pairs:
        document = pair.get("document_text") or ""
        if not document:
            continue
        parts.append(document + " ")
        total += len(document) + 1
        if total >= chars:
            break
    return "".join(parts)[:chars]


def run_calibration(bundle, head, data):
    import make_calibration

    target = ROOT / "artifacts" / "calib_mixture.jsonl"
    make_calibration.main(["--data", str(data), "--out", str(target)])
    found = digest(target)
    rows = read_jsonl(target)
    points = []
    for row in rows:
        rows_features = pair_features(bundle, row["text"], row["document"], MAX_MODEL_CANDIDATES)
        raw = ai_share(score_rows(rows_features, head), row["text"])
        points.append({"share": row["share"], "raw": raw, "abs_err": round(abs(raw - row["share"]), 4)})
    worst = max((point["abs_err"] for point in points), default=0.0)
    leave_one_out = 0.0
    for index, point in enumerate(points):
        rest = sorted((item["raw"], item["share"]) for position, item in enumerate(points) if position != index)
        if not rest:
            continue
        value = max(0.0, min(1.0, point["raw"]))
        if value <= rest[0][0]:
            predicted = rest[0][1]
        elif value >= rest[-1][0]:
            predicted = rest[-1][1]
        else:
            predicted = rest[-1][1]
            for position in range(1, len(rest)):
                left_raw, left_share = rest[position - 1]
                right_raw, right_share = rest[position]
                if value <= right_raw:
                    span = right_raw - left_raw
                    weight = (value - left_raw) / span if span else 0.0
                    predicted = left_share + weight * (right_share - left_share)
                    break
        leave_one_out = max(leave_one_out, abs(predicted - point["share"]))
    return {
        "points": points,
        "max_abs_err": round(worst, 4),
        "loo_max_abs_err": round(leave_one_out, 4),
        "sha256": found,
        "sha_match": found == CALIB_SHA256,
    }


def stress(bundle, document, answer):
    timings = []
    for _ in range(STRESS_RUNS):
        started = time.perf_counter()
        rows = pair_features(bundle, answer, document, MAX_MODEL_CANDIDATES)
        timings.append((time.perf_counter() - started) * 1000.0)
        candidates = len(rows)
    median_ms = int(round(statistics.median(timings)))
    return {
        "stress_chars": len(answer),
        "chars": len(answer),
        "document_chars": len(document),
        "runs": STRESS_RUNS,
        "median_elapsed_ms": median_ms,
        "ms_per_1000": round(median_ms / (len(answer) / 1000.0), 3) if answer else None,
        "candidates": candidates,
        "under_60s": median_ms <= 60000,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--extra", default="artifacts/pairs_unlabeled.jsonl")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--cache", default="artifacts/model_cache.jsonl")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--no-stress", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    data_path = Path(args.data)
    if not data_path.exists():
        fail(f"нет файла данных: {data_path}")
    gold_sha = digest(data_path)
    if gold_sha != GOLD_SHA256:
        fail(f"данные расходятся с песочницей: sha256 {gold_sha} != {GOLD_SHA256}")

    pairs = load_pairs(args.data, args.extra)
    splits = load_splits()
    by_id = {pair.get("pair_id"): pair for pair in pairs}

    test_ids = splits["test"]
    val_ids = splits["val"]
    train_ids = splits["train"]
    test_pairs = ordered_pairs(pairs, test_ids)
    val_pairs = ordered_pairs(pairs, val_ids)
    train_pairs = ordered_pairs(pairs, train_ids)
    missing = {
        "test": len(test_ids) - len(test_pairs),
        "val": len(val_ids) - len(val_pairs),
        "train": len(train_ids) - len(train_pairs),
    }
    if args.limit:
        test_pairs = test_pairs[: args.limit]
        val_pairs = val_pairs[: max(6, min(len(val_pairs), args.limit * 2))]
        train_pairs = training_subset(train_pairs, max(8, args.limit * 2))
    else:
        train_pairs = training_subset(train_pairs, TRAIN_LIMIT)

    bundle = load_model(args.weights)
    sha = bundle["sha"]
    cache = read_cache(args.cache)
    cached_before = len(cache)

    counter = {"done": 0}

    def features(pair):
        key = cache_key(pair.get("pair_id"), sha)
        if key in cache:
            return cache[key]["rows"]
        rows = pair_features(bundle, pair.get("model_output") or "", pair.get("document_text") or "", MAX_MODEL_CANDIDATES)
        cache[key] = {"key": key, "pair_id": pair.get("pair_id"), "sha": sha[:16], "rows": rows}
        counter["done"] += 1
        if counter["done"] % PROGRESS_STEP == 0:
            print(f"пар обработано: {counter['done']}", flush=True)
        return rows

    vectors = []
    labels = []
    positives = 0
    for pair in train_pairs:
        rows = features(pair)
        pair_labels = label_rows(rows, pair.get("spans") or [])
        for row, label in zip(rows, pair_labels):
            vectors.append(row["values"])
            labels.append(label)
            positives += label
    head = train_head(vectors, labels) if vectors else {"weights": [0.0] * len(HEAD_FEATURES), "mean": [0.0] * len(HEAD_FEATURES), "scale": [1.0] * len(HEAD_FEATURES)}

    def table_of(collection, label):
        table = {}
        for index, pair in enumerate(collection, start=1):
            table[pair.get("pair_id")] = score_rows(features(pair), head)
            if index % PROGRESS_STEP == 0:
                print(f"{label}: пары {index}/{len(collection)}", flush=True)
        return table

    train_table = table_of(train_pairs, "train")
    val_table = table_of(val_pairs, "val")
    test_table = table_of(test_pairs, "test")

    best, history = scan_threshold(val_table, val_pairs, THRESHOLD_GRID)
    threshold = best["threshold"]
    test_result = metrics(test_pairs, make_predictor(test_table, {pair["pair_id"]: pair.get("model_output") or "" for pair in test_pairs if pair.get("pair_id")}, threshold))
    lexical = evaluate_pairs(test_pairs)

    calibration = None
    if not args.no_calibration:
        calibration = run_calibration(bundle, head, Path(args.data))

    stress_report = None
    if not args.no_stress:
        document = stress_document(pairs)
        stress_report = stress(bundle, document, document)
        stress_report["kind"] = "документ 2M и ответ 2M"

    save_table(train_table, Path(args.out) / "model_train_table.jsonl")
    if not args.limit:
        save_head(Path(args.out) / "model_head.json", head, threshold, sha)
    write_cache(cache, args.cache)
    report = {
        "backend": "model",
        "model": {
            "name": bundle["name"],
            "directory": bundle["directory"],
            "weights_sha256": sha,
            "weights_key": sha[:16],
            "seed": MODEL_SEED,
            "threads": MODEL_THREADS,
            "torch": bundle["torch"].__version__,
        },
        "split": {
            "source": "spanver/splits.json",
            "counts": splits["counts"],
            "test_evaluated": len(test_pairs),
            "val_evaluated": len(val_pairs),
            "train_evaluated": len(train_pairs),
            "missing": missing,
        },
        "gold": {"path": str(args.data), "sha256": gold_sha, "sha_match": gold_sha == GOLD_SHA256},
        "features": {
            "token_level": list(ATTENTION_FEATURES),
            "head": list(HEAD_FEATURES),
            "max_candidates_per_pair": MAX_MODEL_CANDIDATES,
            "top_segments": TOP_SEGMENTS,
            "segment_tokens": SEGMENT_TOKENS,
        },
        "head": {
            "train_pairs": len(train_pairs),
            "rows": len(vectors),
            "positives": int(positives),
            "weights": head["weights"],
            "mean": head["mean"],
            "scale": head["scale"],
        },
        "cascade": {"combine": COMBINE, "note": "риск = модельная головка; лексический риск идёт признаком в головку"},
        "threshold": {
            "value": threshold,
            "grid": "0.50..0.99 шаг 0.01",
            "selected_on": "val",
            "val_f1": best["f1"],
            "val_fpr": best["fpr"],
        },
        "test": {
            "rule": "IoU>=0.5, no type match",
            "span_level": test_result["span_level"],
            "token_level": test_result["token_level"],
            "pair_level": test_result["pair_level"],
            "fpr": test_result["fpr"],
            "type_accuracy": test_result["type_accuracy"],
            "n_pairs": test_result["n_pairs"],
            "n_pairs_with_spans": test_result["n_pairs_with_spans"],
            "n_neg_pairs": test_result["n_neg_pairs"],
            "contract_pass": {
                "f1_ge_0.50": test_result["span_level"]["f1"] >= 0.50,
                "fpr_le_0.40": test_result["fpr"] <= 0.40,
            },
        },
        "lexical_regression": {
            "span_f1": lexical["span_level"]["f1"],
            "fpr": lexical["fpr"],
            "type_accuracy": lexical["type_accuracy"],
        },
        "calibration": calibration,
        "package": package_info(),
        "stress": stress_report,
        "cache": {"path": args.cache, "entries_before": cached_before, "entries_after": len(cache)},
        "seed": MODEL_SEED,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    target = Path(args.out) / "model_eval_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
