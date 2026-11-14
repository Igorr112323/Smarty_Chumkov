import argparse
import json
from pathlib import Path

from .answer_json import parse_answer
from .io_csv import read_gold
from .locate import METHOD_EXACT, METHOD_NORMALIZED, METHOD_UNRESOLVED, locate

GOLD_FILES = (("sberquad-rag.csv", "sberquad", "sber"), ("ruscibench-rag.csv", "ruscibench", "rusci"))
GOLD_DIR = "1_rushallu_rag_gold"


def iter_gold(dataset_path):
    for name, domain, prefix in GOLD_FILES:
        path = dataset_path / GOLD_DIR / name
        for number, row in enumerate(read_gold(path)):
            yield name, domain, f"{prefix}{number:06d}", row


def build(dataset, out):
    dataset_path = Path(dataset)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    stats = {"files": [], "total": 0}
    for name, domain, prefix in GOLD_FILES:
        path = dataset_path / GOLD_DIR / name
        rows = sum(1 for _ in read_gold(path))
        stats["files"].append({"file": name, "domain": domain, "path": str(path), "rows": rows})
        stats["total"] += rows
        print(f"rows {name} = {rows}")
    print(f"total = {stats['total']}")
    located = {METHOD_EXACT: 0, METHOD_NORMALIZED: 0, METHOD_UNRESOLVED: 0}
    steps = {}
    spans_total = 0
    for _, _, pair_id, row in iter_gold(dataset_path):
        spans = parse_answer(row.get("answer"), pair_id)
        spans_total += len(spans)
        for item, found in zip(spans, locate(spans, row.get("model_output") or "")):
            located[found["method"]] = located.get(found["method"], 0) + 1
            if found["method"] == METHOD_NORMALIZED:
                steps[found["steps"]] = steps.get(found["steps"], 0) + 1
    print(f"spans total = {spans_total}")
    print(f"located exact = {located[METHOD_EXACT]}")
    print(f"located normalized = {located[METHOD_NORMALIZED]}")
    print(f"unresolved = {located[METHOD_UNRESOLVED]}")
    if steps:
        print("normalized steps = " + json.dumps(steps, sort_keys=True))
    target = out_path / "raw_stats.json"
    target.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"json written: {target}")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spanver")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--dataset", default="Dataset")
    build_parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.dataset, args.out)
    return 0
