import hashlib
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
STRESS_LONG_CHARS = 2000000
RUNS = 3
SEED = 0
CALIB = "artifacts/calib_mixture.jsonl"


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


def calibrate_interp(raw, points):
    value = max(0.0, min(1.0, raw))
    if value <= points[0][0]:
        return points[0][1]
    for index in range(1, len(points)):
        left_raw, left_share = points[index - 1]
        right_raw, right_share = points[index]
        if value <= right_raw:
            width = right_raw - left_raw
            if width <= 0:
                return right_share
            return left_share + (value - left_raw) / width * (right_share - left_share)
    return points[-1][1]


def leave_one_out(pairs):
    worst = 0.0
    for index in range(len(pairs)):
        rest = [item for position, item in enumerate(pairs) if position != index]
        table = [(item["raw"], item["share"]) for item in rest]
        table.sort()
        predicted = calibrate_interp(pairs[index]["raw"], table)
        worst = max(worst, abs(predicted - pairs[index]["share"]))
    return round(worst, 4)


def calibration(path=CALIB):
    target = Path(path)
    if not target.exists():
        return {"points": [], "max_abs_err": None, "note": "нет файла калибровки"}
    rows = read_jsonl(target)
    points = []
    for row in rows:
        result = verify(row["text"], row["document"])
        measured = result["ai_share"]
        points.append(
            {
                "share": row["share"],
                "ai_share": measured,
                "abs_err": round(abs(measured - row["share"]), 4),
                "raw": result["meta"].get("ai_share_raw", 0.0),
                "n_human": row["n_human"],
                "n_ai": row["n_ai"],
            }
        )
    worst = max((item["abs_err"] for item in points), default=0.0)
    return {
        "points": points,
        "max_abs_err": round(worst, 4),
        "loo_max_abs_err": leave_one_out(points),
        "curve": [{"raw": item["raw"], "share": item["share"]} for item in points],
        "mixtures": len(points),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def stress(path, chars):
    lines = read_jsonl(Path(path))
    parts = []
    total = 0
    for pair in lines:
        document = pair.get("document_text") or ""
        if not document:
            continue
        parts.append(document + " ")
        total += len(document) + 1
        if total >= chars:
            break
    document = "".join(parts)[:chars]
    timings = []
    for _ in range(RUNS):
        started = time.perf_counter()
        verify(document, document)
        timings.append((time.perf_counter() - started) * 1000.0)
    median_ms = int(round(statistics.median(timings)))
    return {
        "stress_chars": len(document),
        "runs": RUNS,
        "median_elapsed_ms": median_ms,
        "ms_per_1000": round(median_ms / (len(document) / 1000.0), 3) if document else None,
        "under_60s": median_ms <= 60000,
        "under_30ms_per_1000": median_ms <= 30 * (len(document) / 1000.0),
    }


def package_info(path="artifacts/package.json"):
    target = Path(path)
    if not target.exists():
        return {"built": False}
    data = json.loads(target.read_text(encoding="utf-8"))
    return {
        "built": True,
        "package": data.get("package"),
        "bytes": data.get("bytes"),
        "size_kb": data.get("size_kb"),
        "under_limit": data.get("under_limit"),
        "sha256": data.get("sha256"),
        "python_requires": data.get("python_requires"),
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
        "ai_share_calib": calibration(),
        "package": package_info(),
        "stress": stress(data, STRESS_LONG_CHARS),
        "ai_share_sample": {"value": sample["ai_share"]},
        "seed": SEED,
        "python": platform.python_version(),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    target = out_path / "module_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
