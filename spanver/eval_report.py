import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import evaluate_pairs, iou, load_split, read_jsonl
from .module import FEATURE_NAMES, IOU_THRESHOLD, THRESHOLD, candidate_spans, verify

CORE_FILES = ("spanver/module.py", "spanver/api.py", "spanver/cli.py")
STRESS_CHARS = 50000
RUNS = 3
SEED = 0


def core_size_kb():
    total = 0
    for name in CORE_FILES:
        total += Path(name).stat().st_size
    return round(total / 1024.0, 1)


def real_text_pool(pairs):
    chunks = []
    for pair in pairs:
        for field in ("model_output", "document_text"):
            value = pair.get(field) or ""
            if value:
                chunks.append(value)
    return " ".join(chunks)


def slice_of(pool, start, size):
    if len(pool) <= size:
        return pool
    offset = start % max(1, len(pool) - size)
    return pool[offset : offset + size]


def ceiling(pairs):
    gold_total = 0
    covered = 0
    pairs_with_gold = 0
    pairs_coverable = 0
    for pair in pairs:
        gold = pair.get("spans") or []
        if not gold:
            continue
        pairs_with_gold += 1
        candidates = candidate_spans(pair.get("model_output") or "")
        hit = False
        for span in gold:
            gold_total += 1
            best = max((iou(start, end, span["start"], span["end"]) for start, end, _ in candidates), default=0.0)
            if best >= IOU_THRESHOLD:
                covered += 1
                hit = True
        if hit:
            pairs_coverable += 1
    return {
        "gold_spans": gold_total,
        "oracle_covered": covered,
        "oracle_recall": round(covered / gold_total, 4) if gold_total else 0.0,
        "pairs_with_gold": pairs_with_gold,
        "pairs_coverable": pairs_coverable,
    }


def main_head():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def make_report(out="artifacts", data="artifacts/gold.jsonl"):
    out_path = Path(out)
    test_pairs = load_split(data, "test")
    metrics = evaluate_pairs(test_pairs)
    everything = read_jsonl(Path(data))
    pool = real_text_pool(everything)
    answer = slice_of(pool, 0, STRESS_CHARS)
    document = slice_of(pool, STRESS_CHARS // 2, STRESS_CHARS)
    timings = []
    for _ in range(RUNS):
        started = time.perf_counter()
        verify(answer, document)
        timings.append((time.perf_counter() - started) * 1000.0)
    median_ms = int(round(statistics.median(timings)))
    sample = verify(
        test_pairs[0].get("model_output") or "",
        test_pairs[0].get("document_text") or "",
    )
    report = {
        "main_head": main_head(),
        "module": {
            "core_files": list(CORE_FILES),
            "core_size_kb": core_size_kb(),
            "five_token_features": list(FEATURE_NAMES),
            "threshold_val": THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
        "metric_rule": "IoU>=0.5, no type match",
        "eval_test": {
            "n_pairs": metrics["n_pairs"],
            "n_pairs_with_spans": metrics["n_pairs_with_spans"],
            "n_neg_pairs": metrics["n_neg_pairs"],
            "span_level": metrics["span_level"],
            "token_level": metrics["token_level"],
            "pair_level": metrics["pair_level"],
            "fpr": metrics["fpr"],
            "type_accuracy": metrics["type_accuracy"],
            "contract_pass": {
                "f1_ge_0.50": metrics["span_level"]["f1"] >= 0.50,
                "fpr_le_0.40": metrics["fpr"] <= 0.40,
            },
        },
        "ceiling": ceiling(test_pairs),
        "speed": {
            "stress_chars": len(answer),
            "runs": RUNS,
            "median_elapsed_ms": median_ms,
            "under_1s": median_ms <= 1000,
        },
        "ai_share_sample": {"value": sample["ai_share"], "note": "калибровка — шаг 6"},
        "seed": SEED,
        "python": platform.python_version(),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    target = out_path / "module_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
