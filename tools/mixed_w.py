import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, module
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.splits import load_splits, ordered_pairs

TOKEN_THRESHOLD = 0.25
PAIR_GATE = 0.45
MIN_CHARS = 2
WEIGHTS = tuple(round(0.1 * step, 1) for step in range(11))
THRESHOLDS = tuple(round(0.05 + 0.05 * step, 2) for step in range(20))
TARGET_FPR = 0.35
FPR_LIMIT = 0.40
LEXICAL_SPAN_F1_TEST = 0.2835


def coverage(span, spans):
    start, end = span["start"], span["end"]
    length = end - start + 1
    if length <= 0:
        return 0.0
    covered = 0
    for current in spans:
        left = max(start, current["start"])
        right = min(end, current["end"])
        if right >= left:
            covered += right - left + 1
    return min(1.0, covered / length)


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


def load_oof(folder):
    dumps = {}
    for path in sorted(Path(folder).glob("oof_*.json")):
        dumps.update(json.loads(path.read_text(encoding="utf-8")))
    return dumps


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="artifacts/cv/base")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--report", default="artifacts/mixed_w.json")
    args = parser.parse_args(argv)

    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    by_id = {pair.get("pair_id"): pair for pair in gold}
    ours = [by_id[pair_id] for name in ("train", "val") for pair_id in splits[name] if pair_id in by_id]
    ours = [pair for pair in ours if pair.get("model_output")]
    dumps = load_oof(args.oof)
    val_ids = set(splits["val"])
    train_ids = set(splits["train"])

    prepared = {}
    for pair in ours:
        key = pair.get("pair_id")
        dumped = dumps.get(key)
        answer = pair.get("model_output") or ""
        lexical = snapped(module.verify(answer, pair.get("document_text") or "")["spans"], answer, "expand")
        model_spans = []
        if dumped:
            top = max((value for _, _, value in dumped["pieces"]), default=0.0)
            if top >= PAIR_GATE:
                flags = decode.flags_for(dumped["pieces"], TOKEN_THRESHOLD, None)
                model_spans = decode.spans_from_flags(dumped["answer"], dumped["pieces"], flags, MIN_CHARS)
        model_spans = snapped(model_spans, answer, "expand")
        prepared[key] = {"lexical": lexical, "model": model_spans, "answer": answer}

    def candidates_for(key):
        item = prepared[key]
        merged = {}
        for span in item["lexical"] + item["model"]:
            merged.setdefault((span["start"], span["end"]), span)
        scored = []
        for span in merged.values():
            lex_cov = coverage(span, item["lexical"])
            model_cov = coverage(span, item["model"])
            scored.append(dict(span, cov_lex=lex_cov, cov_model=model_cov))
        return scored

    cache = {key: candidates_for(key) for key in prepared}

    def predict(pair, weight, threshold):
        key = pair.get("pair_id")
        chosen = []
        for span in cache.get(key, []):
            value = weight * span["cov_lex"] + (1.0 - weight) * span["cov_model"]
            if value >= threshold:
                chosen.append(dict(span, risk=round(value, 4)))
        chosen = module.select(chosen, 20, module.NMS_IOU)
        return sorted(chosen, key=lambda item: item["start"])

    def measure(subset, weight, threshold):
        predictor = lambda pair: predict(pair, weight, threshold)
        result = metrics(subset, predictor)
        return {
            "span_precision": result["span_level"]["precision"],
            "span_recall": result["span_level"]["recall"],
            "span_f1": result["span_level"]["f1"],
            "token_precision": result["token_level"]["precision"],
            "token_recall": result["token_level"]["recall"],
            "token_f1": result["token_level"]["f1"],
            "pair_f1": result["pair_level"]["f1"],
            "fpr_pair": result["fpr"],
            "fpr_token": token_fpr(subset, predictor),
            "type_accuracy": result["type_accuracy"],
        }

    full_rows = []
    for weight in WEIGHTS:
        for threshold in THRESHOLDS:
            row = measure(ours, weight, threshold)
            full_rows.append(dict(row, weight=weight, threshold=threshold))

    chosen_rows = []
    for weight in WEIGHTS:
        subset = [row for row in full_rows if row["weight"] == weight]
        nearest = min(subset, key=lambda row: abs(row["fpr_pair"] - TARGET_FPR))
        chosen_rows.append(nearest)
        print(
            f"w {weight:.1f}: порог {nearest['threshold']:.2f} span F1 {nearest['span_f1']:.4f} "
            f"token F1 {nearest['token_f1']:.4f} FPR_pair {nearest['fpr_pair']:.4f} FPR_token {nearest['fpr_token']:.4f} "
            f"pair F1 {nearest['pair_f1']:.4f}",
            flush=True,
        )

    allowed = [row for row in full_rows if row["fpr_pair"] <= FPR_LIMIT and row["fpr_token"] <= FPR_LIMIT]
    pool = allowed or full_rows
    best_span = max(pool, key=lambda row: (row["span_f1"], row["token_f1"]))
    best_token = max(pool, key=lambda row: (row["token_f1"], row["span_f1"]))
    val_pairs = [pair for pair in ours if pair.get("pair_id") in val_ids]
    train_pairs = [pair for pair in ours if pair.get("pair_id") in train_ids]
    detail = {
        "best_span_on_val": measure(val_pairs, best_span["weight"], best_span["threshold"]),
        "best_span_on_train": measure(train_pairs, best_span["weight"], best_span["threshold"]),
        "chosen_on_val": None,
        "chosen_on_train": None,
    }
    nearest_all = min(full_rows, key=lambda row: (abs(row["fpr_pair"] - TARGET_FPR), -row["span_f1"]))
    detail["chosen_on_val"] = measure(val_pairs, nearest_all["weight"], nearest_all["threshold"])
    detail["chosen_on_train"] = measure(train_pairs, nearest_all["weight"], nearest_all["threshold"])
    report = {
        "pairs": len(ours),
        "val_pairs": len(val_pairs),
        "train_pairs": len(train_pairs),
        "oof": args.oof,
        "weights": list(WEIGHTS),
        "thresholds": list(THRESHOLDS),
        "target_fpr": TARGET_FPR,
        "rows": full_rows,
        "chosen": chosen_rows,
        "best_span": best_span,
        "best_token": best_token,
        "detail": detail,
        "lexical_span_f1_reference": LEXICAL_SPAN_F1_TEST,
        "span_target_met": bool(best_span["span_f1"] > LEXICAL_SPAN_F1_TEST),
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"лучший по span F1: w {best_span['weight']} порог {best_span['threshold']} span F1 {best_span['span_f1']} "
        f"token F1 {best_span['token_f1']} FPR_pair {best_span['fpr_pair']}",
        flush=True,
    )
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
