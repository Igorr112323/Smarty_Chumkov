import argparse
import json
from pathlib import Path

from .answer_json import parse_answer
from .io_csv import read_gold
from .locate import METHOD_EXACT, METHOD_NORMALIZED, METHOD_UNRESOLVED, locate
from .pairs import doc_ids, ordered
from .qc import inspect
from .report import build_report
from .split import UNUSED, assign, doc_overlap
from .splits import canonical_assignment
from .types import TYPE_KEYS, classify

GOLD_FILES = (
    ("sberquad-rag.csv", "sberquad", "sber", "1_rushallu_rag_gold/sberquad-rag.csv"),
    ("ruscibench-rag.csv", "ruscibench", "rusci", "1_rushallu_rag_gold/ruscibench-rag.csv"),
)
GOLD_DIR = "1_rushallu_rag_gold"


def is_clean(raw):
    if raw is None:
        return False
    text = raw.strip()
    if not text:
        return True
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, list):
        return False
    if not data:
        return True
    return all(
        isinstance(item, dict) and item.get("span") is None and item.get("type") is None for item in data
    )


def build_gold(dataset_path, counters):
    pairs = []
    for name, domain, prefix, source in GOLD_FILES:
        path = dataset_path / GOLD_DIR / name
        for number, row in enumerate(read_gold(path)):
            pair_id = f"{prefix}{number:06d}"
            answer = row.get("answer")
            raw_spans = parse_answer(answer, pair_id)
            output = row.get("model_output") or ""
            document = row.get("joined_reference") or ""
            spans = []
            for item, found in zip(raw_spans, locate(raw_spans, output)):
                counters[found["method"]] = counters.get(found["method"], 0) + 1
                spans.append(
                    {
                        "text": item["text"],
                        "start": found["start"],
                        "end": found["end"],
                        "type_src": item["type"],
                        "type": classify(item["text"], item["type"]),
                        "evidence_ids": [],
                    }
                )
            pairs.append(
                ordered(
                    {
                        "pair_id": pair_id,
                        "source": source,
                        "domain": domain,
                        "query_id": str(row.get("query_id")),
                        "query_text": row.get("query_text") or "",
                        "document_text": document,
                        "doc_ids": doc_ids(document),
                        "model_output": output,
                        "spans": spans,
                        "clean": is_clean(answer),
                        "label_source": "manual",
                        "perturb": {
                            "rel_position_mod": int(row.get("rel_position_mod")),
                            "score_mod": float(row.get("score_mod")),
                        },
                        "split": None,
                    }
                )
            )
    return pairs


def write_jsonl(path, pairs):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)


def read_jsonl(path):
    target = Path(path)
    if not target.exists():
        return []
    pairs = []
    with open(target, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def apply_split(out_path, gold):
    unlabeled = read_jsonl(out_path / "pairs_unlabeled.jsonl")
    everything = gold + unlabeled
    assignment, stats = assign(everything)
    canonical = canonical_assignment()
    overridden = 0
    for pair in everything:
        fixed = canonical.get(pair["pair_id"])
        pair["split"] = assignment[pair["pair_id"]]
        if fixed:
            pair["split"] = fixed
            overridden += 1
    if overridden:
        counts = {key: 0 for key in (UNUSED, "val", "train", "test")}
        for pair in everything:
            counts[pair["split"]] = counts.get(pair["split"], 0) + 1
        stats["split_counts"] = counts
        stats["canonical_overridden"] = overridden
    write_jsonl(out_path / "gold.jsonl", gold)
    if unlabeled:
        write_jsonl(out_path / "pairs_unlabeled.jsonl", unlabeled)
    stats["doc_overlap_across_splits"] = doc_overlap(everything, assignment)
    stats["pairs_total"] = len(everything)
    print(f"components = {stats['components']}")
    counts = stats["split_counts"]
    print(f"pairs train/val/test/unused = {counts['train']} {counts['val']} {counts['test']} {counts['unused']}")
    print(f"test pairs = {counts['test']}")
    print(f"test manual pairs = {stats['split_manual']['test']}")
    print(f"test pairs with spans = {stats['test_pairs_with_spans']}")
    print(f"doc overlap between splits = {stats['doc_overlap_across_splits']}")
    print(f"max component size = {stats['max_component']}")
    print(f"unused unlabeled = {stats['unused_unlabeled']}")
    print(f"empty doc_ids = {stats['empty_doc_ids']}")
    return stats


def type_counts(pairs):
    counts = {key: 0 for key in TYPE_KEYS}
    for pair in pairs:
        for span in pair["spans"]:
            counts[span["type"]] += 1
    return counts


def build(dataset, out):
    dataset_path = Path(dataset)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    stats = {"files": [], "total": 0}
    for name, domain, prefix, source in GOLD_FILES:
        path = dataset_path / GOLD_DIR / name
        rows = sum(1 for _ in read_gold(path))
        stats["files"].append({"file": name, "domain": domain, "path": str(path), "rows": rows})
        stats["total"] += rows
        print(f"rows {name} = {rows}")
    print(f"total = {stats['total']}")
    counters = {METHOD_EXACT: 0, METHOD_NORMALIZED: 0, METHOD_UNRESOLVED: 0}
    pairs = build_gold(dataset_path, counters)
    lines = write_jsonl(out_path / "gold.jsonl", pairs)
    print(f"spans total = {sum(len(pair['spans']) for pair in pairs)}")
    print(f"located exact = {counters[METHOD_EXACT]}")
    print(f"located normalized = {counters[METHOD_NORMALIZED]}")
    print(f"unresolved = {counters[METHOD_UNRESOLVED]}")
    print(f"written {out_path / 'gold.jsonl'} {lines} lines")
    print("type_counts = " + json.dumps(type_counts(pairs), ensure_ascii=False, sort_keys=True))
    report = inspect(pairs)
    print(f"coverage = {report['coverage']:.4f}")
    print(f"out_of_bounds = {report['out_of_bounds']}")
    print(f"mismatch = {report['span_text_mismatch']}")
    print(f"overlaps = {report['overlaps']}")
    print(f"nested = {report['nested']}")
    print(f"qc: {'OK' if report['ok'] else 'FAIL'}")
    split_stats = apply_split(out_path, pairs)
    unlabeled = read_jsonl(out_path / "pairs_unlabeled.jsonl")
    report_doc, report_path = build_report(dataset_path, out_path, pairs, unlabeled, counters, report, split_stats)
    print(f"report written: {report_path}")
    target = out_path / "raw_stats.json"
    target.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"json written: {target}")
    ok = report["ok"] and split_stats["doc_overlap_across_splits"] == 0
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spanver")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--dataset", default="Dataset")
    build_parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)
    code = 0
    if args.command == "build":
        code = build(args.dataset, args.out)
    return code
