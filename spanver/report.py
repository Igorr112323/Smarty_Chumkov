import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from .types import TYPE_KEYS

TYPE_SRC_KEYS = ("Contradiction", "Unconfirmed", "Missing", "Excess", "Partial", "Oversight")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path):
    target = Path(path)
    if not target.exists():
        return {"sha256": "", "bytes": 0, "lines": 0}
    lines = 0
    with open(target, "rb") as handle:
        for _ in handle:
            lines += 1
    return {"sha256": sha256_file(target), "bytes": target.stat().st_size, "lines": lines}


def length_stats(values):
    values = sorted(values)
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0, "min": 0}
    middle = len(values) // 2
    if len(values) % 2:
        median = float(values[middle])
    else:
        median = (values[middle - 1] + values[middle]) / 2.0
    return {
        "mean": round(sum(values) / len(values), 2),
        "median": median,
        "max": max(values),
        "min": min(values),
    }


def built_utc():
    fixed = os.environ.get("SPANVER_BUILT_UTC")
    if fixed:
        return fixed
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def candidates_block(out_path):
    stats_path = Path(out_path) / "candidates_stats.json"
    block = {
        "pool": 0,
        "count": 0,
        "per_doc_cap": 2,
        "doc_overlap_with_gold": 0,
        "candidates_sha256": "",
        "check_sha256": "",
    }
    if stats_path.exists():
        stored = json.loads(stats_path.read_text(encoding="utf-8"))
        for key in ("pool", "count", "per_doc_cap", "doc_overlap_with_gold"):
            if key in stored:
                block[key] = stored[key]
    for key, name in (
        ("candidates_sha256", "candidates_200.jsonl"),
        ("check_sha256", "candidates_200_check.jsonl"),
    ):
        target = Path(out_path) / name
        if target.exists():
            block[key] = sha256_file(target)
    return block


def build_report(dataset_path, out_path, gold, unlabeled, counters, qc, split_stats):
    out = Path(out_path)
    type_counts = {key: 0 for key in TYPE_KEYS}
    type_src_counts = {key: 0 for key in TYPE_SRC_KEYS}
    span_lengths = []
    negatives = 0
    for pair in gold:
        if pair.get("clean"):
            negatives += 1
        for span in pair.get("spans") or []:
            type_counts[span["type"]] = type_counts.get(span["type"], 0) + 1
            type_src_counts[span["type_src"]] = type_src_counts.get(span["type_src"], 0) + 1
            span_lengths.append(len(span["text"]))
    everything = list(gold) + list(unlabeled)
    document_lengths = [len(pair.get("document_text") or "") for pair in everything]
    gold_rel = sum(1 for pair in gold if pair["perturb"]["rel_position_mod"] != -1)
    gold_score = sum(1 for pair in gold if pair["perturb"]["score_mod"] != 0.0)
    train_rel = sum(1 for pair in unlabeled if pair["perturb"]["rel_position_mod"] != -1)
    train_score = sum(1 for pair in unlabeled if pair["perturb"]["score_mod"] != 0.0)
    report = {
        "pairs_total": len(gold),
        "pairs_manual": len(gold),
        "pairs_unlabeled": len(unlabeled),
        "spans_total": sum(len(pair["spans"]) for pair in gold),
        "spans_located_exact": counters.get("exact", 0),
        "spans_located_normalized": counters.get("normalized", 0),
        "spans_unresolved": counters.get("unresolved", 0),
        "negative_pairs": negatives,
        "type_counts": type_counts,
        "type_src_counts": type_src_counts,
        "split_counts": split_stats["split_counts"],
        "test_pairs_with_spans": split_stats["test_pairs_with_spans"],
        "components": split_stats["components"],
        "max_component": split_stats["max_component"],
        "doc_overlap_across_splits": split_stats["doc_overlap_across_splits"],
        "coverage": round(qc["coverage"], 4),
        "out_of_bounds": qc["out_of_bounds"],
        "span_text_mismatch": qc["span_text_mismatch"],
        "overlaps": qc["overlaps"],
        "nested": qc["nested"],
        "perturbed_rows": {
            "gold_rel_position": gold_rel,
            "gold_score": gold_score,
            "train_rel_position": train_rel,
            "train_score": train_score,
        },
        "doc_len_chars": length_stats(document_lengths),
        "span_len_chars": {
            "mean": length_stats(span_lengths)["mean"],
            "median": length_stats(span_lengths)["median"],
            "max": length_stats(span_lengths)["max"],
            "min": length_stats(span_lengths)["min"],
        },
        "unavailable_inputs": [
            {
                "path": str(Path(dataset_path) / "2_rushallu_rag_pairs" / "ruscibench-train-rag.parquet"),
                "reason": "указатель LFS, данные не получены",
            }
        ],
        "candidates": candidates_block(out),
        "files": {
            "gold.jsonl": file_info(out / "gold.jsonl"),
            "pairs_unlabeled.jsonl": file_info(out / "pairs_unlabeled.jsonl"),
        },
        "python": platform.python_version(),
        "built_utc": built_utc(),
    }
    target = out / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report, target
