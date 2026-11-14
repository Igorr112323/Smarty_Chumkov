import argparse
import json
from pathlib import Path

from .io_csv import read_gold

GOLD_FILES = (("sberquad-rag.csv", "sberquad"), ("ruscibench-rag.csv", "ruscibench"))
GOLD_DIR = "1_rushallu_rag_gold"


def build(dataset, out):
    dataset_path = Path(dataset)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    stats = {"files": [], "total": 0}
    for name, domain in GOLD_FILES:
        path = dataset_path / GOLD_DIR / name
        rows = sum(1 for _ in read_gold(path))
        stats["files"].append({"file": name, "domain": domain, "path": str(path), "rows": rows})
        stats["total"] += rows
        print(f"rows {name} = {rows}")
    print(f"total = {stats['total']}")
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
