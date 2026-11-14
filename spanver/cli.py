import argparse
import json
from pathlib import Path

from .api import serve
from .build import build
from .evaluate import evaluate_pairs, load_split
from .module import verify


def value_of(raw):
    candidate = Path(raw)
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spanver")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--dataset", default="Dataset")
    build_parser.add_argument("--out", default="artifacts")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--answer", required=True)
    verify_parser.add_argument("--document", required=True)
    verify_parser.add_argument("--json", action="store_true")
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--host", default="0.0.0.0")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--data", default="artifacts/gold.jsonl")
    evaluate_parser.add_argument("--split", default="test")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--data", default="artifacts/gold.jsonl")
    report_parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)
    if args.command == "build":
        return build(args.dataset, args.out)
    if args.command == "verify":
        result = verify(value_of(args.answer), value_of(args.document))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "serve":
        serve(args.port, args.host)
        return 0
    if args.command == "evaluate":
        pairs = load_split(args.data, args.split)
        print(json.dumps(evaluate_pairs(pairs), ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        from .eval_report import make_report

        path = make_report(args.out, args.data)
        print(f"module report written: {path}")
        return 0
    return 2
