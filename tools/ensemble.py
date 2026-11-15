import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, module
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import load_model, pair_probabilities

UNION_NMS = 0.40
UNION_LIMIT = 10
TOP = 4
FPR_LIMIT = 0.40


def token_fpr(pairs, predictor):
    support = 0
    flagged = 0
    for pair in pairs:
        answer = pair.get("model_output") or ""
        gold = pair.get("spans") or []
        predicted = predictor(pair)
        for start, end in token_positions(answer):
            if any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in gold):
                continue
            support += 1
            if any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in predicted):
                flagged += 1
    return round(flagged / support, 4) if support else 0.0


def overlaps(first, second):
    return min(first["end"], second["end"]) - max(first["start"], second["start"]) >= 0


def union_spans(left, right):
    merged = list(left) + list(right)
    unique = {}
    for span in merged:
        unique[(span["start"], span["end"])] = span
    return module.select(list(unique.values()), UNION_LIMIT, UNION_NMS)


def intersection_spans(left, right):
    kept = {}
    for span in list(left) + list(right):
        other = right if span in left else left
        if any(overlaps(span, candidate) for candidate in other):
            kept[(span["start"], span["end"])] = span
    accepted = []
    for span in sorted(kept.values(), key=lambda item: (-item.get("confidence", 0.0), item["start"])):
        if any(overlaps(span, current) for current in accepted):
            continue
        accepted.append(span)
    accepted.sort(key=lambda item: item["start"])
    return accepted


def best_confidence(spans):
    return max((span.get("confidence", 0.0) for span in spans), default=0.0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="artifacts/frozen_config.json")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--split", default="val")
    parser.add_argument("--report", default="artifacts/ensemble.json")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    pairs = [pair for pair in ordered_pairs(gold, splits[args.split]) if pair.get("model_output")]
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}
    torch, tokenizer, model = load_model(config["checkpoint"])
    started = time.perf_counter()
    lexical, model_spans, cache = {}, {}, {}
    for pair in pairs:
        lexical[pair.get("pair_id")] = snapped(
            module.verify(answers[pair.get("pair_id")], pair.get("document_text") or "")["spans"],
            answers[pair.get("pair_id")],
            config["snap"],
        )
        cached = pair_probabilities(tokenizer, model, torch, pair, top=TOP)
        cache[pair.get("pair_id")] = cached
        top = max((value for _, _, value in cached["pieces"]), default=0.0)
        if top < config["pair_gate"]:
            model_spans[pair.get("pair_id")] = []
            continue
        flags = decode.flags_for(cached["pieces"], config["token_threshold"], config["neighbor"])
        spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, config["min_chars"])
        model_spans[pair.get("pair_id")] = snapped(spans, answers[pair.get("pair_id")], config["snap"])

    def variant_a(pair):
        key = pair.get("pair_id")
        return union_spans(lexical[key], model_spans[key])

    def variant_b(pair):
        key = pair.get("pair_id")
        return intersection_spans(lexical[key], model_spans[key])

    def variant_c(pair):
        key = pair.get("pair_id")
        left, right = lexical[key], model_spans[key]
        if left and right:
            return left if best_confidence(left) >= best_confidence(right) else right
        return left or right

    def variant_d(pair):
        key = pair.get("pair_id")
        answer = answers[key]
        cached = cache[key]
        top = max((value for _, _, value in cached["pieces"]), default=0.0)
        pieces = []
        for index, (start, end) in enumerate(token_positions(answer)):
            lexical_score = 0.0
            for span in lexical[key]:
                if min(end, span["end"]) - max(start, span["start"]) >= 0:
                    lexical_score = max(lexical_score, span.get("risk", 0.0))
            model_score = 0.0
            if top >= config["pair_gate"]:
                for piece_start, piece_end, value in cached["pieces"]:
                    if min(end + 1, piece_end) - max(start, piece_start) > 0:
                        model_score = max(model_score, value)
            pieces.append((start, end + 1, 0.5 * lexical_score + 0.5 * model_score))
        flags = decode.flags_for(pieces, config["token_threshold"], config["neighbor"])
        spans = decode.spans_from_flags(answer, pieces, flags, config["min_chars"])
        return snapped(spans, answer, config["snap"])

    table = {}
    for name, predictor in (
        ("lexical", lambda pair: lexical[pair.get("pair_id")]),
        ("model", lambda pair: model_spans[pair.get("pair_id")]),
        ("A_union_nms040", variant_a),
        ("B_intersection", variant_b),
        ("C_confidence", variant_c),
        ("D_token_average", variant_d),
    ):
        result = metrics(pairs, predictor)
        table[name] = {
            "span_precision": result["span_level"]["precision"],
            "span_recall": result["span_level"]["recall"],
            "span_f1": result["span_level"]["f1"],
            "token_precision": result["token_level"]["precision"],
            "token_recall": result["token_level"]["recall"],
            "token_f1": result["token_level"]["f1"],
            "pair_f1": result["pair_level"]["f1"],
            "fpr_pair": result["fpr"],
            "fpr_token": token_fpr(pairs, predictor),
            "type_accuracy": result["type_accuracy"],
        }
        row = table[name]
        print(
            f"{name}: span {row['span_precision']:.4f}/{row['span_recall']:.4f}/{row['span_f1']:.4f} "
            f"token {row['token_precision']:.4f}/{row['token_recall']:.4f}/{row['token_f1']:.4f} "
            f"pair {row['pair_f1']:.4f} FPR_pair {row['fpr_pair']:.4f} FPR_token {row['fpr_token']:.4f}",
            flush=True,
        )

    variants = {name: row for name, row in table.items() if name[0] in "ABCD"}
    allowed = {name: row for name, row in variants.items() if row["fpr_pair"] <= FPR_LIMIT and row["fpr_token"] <= FPR_LIMIT}
    pool = allowed or variants
    best_token = max(pool, key=lambda name: pool[name]["token_f1"])
    best_span = max(pool, key=lambda name: pool[name]["span_f1"])
    report = {
        "split": args.split,
        "pairs": len(pairs),
        "config": config,
        "table": table,
        "best_token": best_token,
        "best_span": best_span,
        "lexical_span_f1": table["lexical"]["span_f1"],
        "confirmed_on_test": False,
        "seconds": round(time.perf_counter() - started, 1),
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"лучший по token F1: {best_token} ({table[best_token]['token_f1']})", flush=True)
    print(f"лучший по span F1: {best_span} ({table[best_span]['span_f1']})", flush=True)
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
