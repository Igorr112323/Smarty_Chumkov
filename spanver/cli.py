import argparse
import json
from pathlib import Path

from .api import serve
from .build import build
from .evaluate import evaluate_pairs, load_split
from .model_features import ModelUnavailable
from .model_score import HeadMissing, verify_model
from .module import verify
from .tiny_backend import DEFAULT_MAX_SPANS, TinyBundle, TinyHeadMissing, TinyUnavailable


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
    verify_parser.add_argument("--backend", choices=("lexical", "model", "tiny"), default="lexical")
    verify_parser.add_argument("--weights", default=None)
    verify_parser.add_argument("--head", default="artifacts/model_head.json")
    verify_parser.add_argument("--max-spans", type=int, default=DEFAULT_MAX_SPANS)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--backend", choices=("lexical", "model", "tiny"), default="lexical")
    serve_parser.add_argument("--weights", default=None)
    serve_parser.add_argument("--head", default="artifacts/model_head.json")
    serve_parser.add_argument("--max-spans", type=int, default=DEFAULT_MAX_SPANS)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--data", default="artifacts/gold.jsonl")
    evaluate_parser.add_argument("--split", default="test")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--data", default="artifacts/gold.jsonl")
    report_parser.add_argument("--out", default="artifacts")
    compliance_parser = subparsers.add_parser("compliance")
    compliance_parser.add_argument("--module", default="artifacts/module_report.json")
    compliance_parser.add_argument("--model", default="artifacts/model_eval_report.json")
    compliance_parser.add_argument("--out", default="artifacts/compliance.json")
    compliance_parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build":
        return build(args.dataset, args.out)
    if args.command == "verify":
        if args.backend == "tiny":
            try:
                bundle = TinyBundle(args.weights, args.head if args.head != "artifacts/model_head.json" else "artifacts/tiny_head.json")
                result = bundle.verify(value_of(args.answer), value_of(args.document), max_spans=args.max_spans)
            except (TinyUnavailable, TinyHeadMissing) as error:
                print(str(error))
                return 2
        elif args.backend == "model":
            if not args.weights:
                print("укажите каталог весов: --weights models\\ruSciBERT")
                return 2
            try:
                result = verify_model(value_of(args.answer), value_of(args.document), args.weights, args.head)
            except (ModelUnavailable, HeadMissing) as error:
                print(str(error))
                return 2
        else:
            result = verify(value_of(args.answer), value_of(args.document))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "serve":
        try:
            serve(args.port, args.host, args.backend, args.weights, args.head, args.max_spans)
        except (ModelUnavailable, HeadMissing, TinyUnavailable, TinyHeadMissing) as error:
            print(str(error))
            return 2
        return 0
    if args.command == "evaluate":
        pairs = load_split(args.data, args.split)
        print(json.dumps(evaluate_pairs(pairs), ensure_ascii=False, indent=2))
        return 0
    if args.command == "compliance":
        from .compliance import main as compliance_main

        return compliance_main(["--module", args.module, "--model", args.model, "--out", args.out] + (["--markdown"] if args.markdown else []))
    if args.command == "report":
        from .eval_report import make_report

        path = make_report(args.out, args.data)
        print(f"module report written: {path}")
        return 0
    return 2
