import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.splits import load_splits, ordered_pairs
from spanver.span_detector import load_model, pair_probabilities

TOKEN_THRESHOLD = 0.20
PAIR_GATE = 0.45
MIN_CHARS = 2
LEXICAL_SPAN_F1 = 0.2835
LEXICAL_PAIR_F1 = 0.3547
TARGET = 0.50
FPR_LIMIT = 0.40


def fpr_token(pairs, predictor):
    support = 0
    flagged = 0
    for pair in pairs:
        answer = pair.get("model_output") or ""
        gold = pair.get("spans") or []
        positions = token_positions(answer)
        predicted = predictor(pair)
        for start, end in positions:
            inside = any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in gold)
            if inside:
                continue
            support += 1
            if any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in predicted):
                flagged += 1
    return round(flagged / support, 4) if support else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/negatives/epoch_1")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--report", default="artifacts/report.json")
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args(argv)

    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    pairs = [pair for pair in ordered_pairs(gold, splits["val"]) if pair.get("model_output")]
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}
    torch, tokenizer, model = load_model(args.checkpoint)
    cache = {pair.get("pair_id"): pair_probabilities(tokenizer, model, torch, pair, top=args.top) for pair in pairs}

    def predict(pair):
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            return []
        top = max((value for _, _, value in cached["pieces"]), default=0.0)
        if top < PAIR_GATE:
            return []
        flags = decode.flags_for(cached["pieces"], TOKEN_THRESHOLD, None)
        spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, MIN_CHARS)
        return snapped(spans, answers[pair.get("pair_id")], "expand")

    result = metrics(pairs, predict)
    headline = {
        "target": "token F1 >= 0,50 при FPR <= 0,40 на val150",
        "token_f1": result["token_level"]["f1"],
        "token_precision": result["token_level"]["precision"],
        "token_recall": result["token_level"]["recall"],
        "fpr_pair": result["fpr"],
        "fpr_token": fpr_token(pairs, predict),
        "fpr_limit": FPR_LIMIT,
        "span_f1_model": result["span_level"]["f1"],
        "pair_f1_model": result["pair_level"]["f1"],
        "type_accuracy": result["type_accuracy"],
        "span_f1_lexical": LEXICAL_SPAN_F1,
        "pair_f1_lexical": LEXICAL_PAIR_F1,
        "checkpoint": args.checkpoint,
        "decode": f"порог токена {TOKEN_THRESHOLD}, минимум {MIN_CHARS} знака, гейт пары {PAIR_GATE}, привязка к клаузам expand",
        "val_pairs": len(pairs),
        "gate_passed": bool(result["token_level"]["f1"] >= TARGET and result["fpr"] <= FPR_LIMIT),
    }
    target = Path(args.report)
    data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    data["headline"] = headline
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(headline, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
