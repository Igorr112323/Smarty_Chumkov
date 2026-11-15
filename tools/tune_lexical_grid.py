import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import module
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.splits import load_splits, ordered_pairs

NMS_GRID = (0.10, 0.25, 0.40, 0.60)
MAX_SPANS_GRID = (1, 2, 3)
MIN_LEN_GRID = (1, 2, 5)
FPR_LIMIT = 0.40


def filtered(spans, min_len):
    return [span for span in spans if span["end"] - span["start"] + 1 >= min_len]


def token_fpr(pairs, predictor):
    support = 0
    flagged = 0
    for pair in pairs:
        answer = pair.get("model_output") or ""
        gold = pair.get("spans") or []
        positions = token_positions(answer)
        predicted = filtered(predictor(pair), 1)
        for index, (start, end) in enumerate(positions):
            inside = any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in gold)
            if inside:
                continue
            support += 1
            if any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in predicted):
                flagged += 1
    return round(flagged / support, 4) if support else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--report", default="artifacts/lexical_grid.json")
    args = parser.parse_args(argv)

    splits = load_splits()
    gold = read_jsonl(Path(args.data))
    pairs = [pair for pair in ordered_pairs(gold, splits["val"]) if pair.get("model_output")]
    started = time.perf_counter()
    rows = []
    for nms in NMS_GRID:
        for max_spans in MAX_SPANS_GRID:
            clock = time.perf_counter()
            raw = {}
            for pair in pairs:
                result = module.verify(
                    pair.get("model_output") or "",
                    pair.get("document_text") or "",
                    max_spans=max_spans,
                    nms_iou=nms,
                )
                raw[pair.get("pair_id")] = result["spans"]
            seconds = time.perf_counter() - clock
            for min_len in MIN_LEN_GRID:
                predictor = lambda pair, store=raw, length=min_len: filtered(store[pair.get("pair_id")], length)
                result = metrics(pairs, predictor)
                rows.append(
                    {
                        "nms_iou": nms,
                        "max_spans": max_spans,
                        "min_len": min_len,
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
                        "seconds": round(seconds, 2),
                    }
                )
    allowed = [row for row in rows if row["fpr_pair"] <= FPR_LIMIT]
    pool = allowed or rows
    best = max(pool, key=lambda row: (row["span_f1"], row["token_f1"]))
    base = next(row for row in rows if row["nms_iou"] == 0.10 and row["max_spans"] == 1 and row["min_len"] == 1)
    best_token = max(pool, key=lambda row: (row["token_f1"], row["span_f1"]))
    report = {
        "pairs": len(pairs),
        "grid": {"nms_iou": list(NMS_GRID), "max_spans": list(MAX_SPANS_GRID), "min_len": list(MIN_LEN_GRID)},
        "fpr_limit": FPR_LIMIT,
        "rows": rows,
        "base": base,
        "best_span": best,
        "best_token": best_token,
        "seconds": round(time.perf_counter() - started, 2),
    }
    print(
        f"база NMS 0,10 / MAX_SPANS 1 / мин 1: span F1 {base['span_f1']} token F1 {base['token_f1']} "
        f"FPR_pair {base['fpr_pair']}",
        flush=True,
    )
    print(
        f"лучшая по span F1 при FPR <= {FPR_LIMIT}: NMS {best['nms_iou']} MAX_SPANS {best['max_spans']} "
        f"мин {best['min_len']} | span F1 {best['span_f1']} token F1 {best['token_f1']} FPR_pair {best['fpr_pair']} "
        f"FPR_token {best['fpr_token']}",
        flush=True,
    )
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
