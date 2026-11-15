import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, lexfeat
from spanver.evaluate import metrics, read_jsonl, token_positions
from spanver.snap import snapped
from spanver.splits import load_splits

THRESHOLDS = tuple(round(0.05 + 0.05 * step, 2) for step in range(12))
PAIR_GATE = 0.45
MIN_CHARS = 2
FPR_LIMIT = 0.40
TOP = 4


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


def measure(pairs, cache, threshold):
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}

    def predict(pair):
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            return []
        top = max((value for _, _, value in cached["pieces"]), default=0.0)
        if top < PAIR_GATE:
            return []
        flags = decode.flags_for(cached["pieces"], threshold, None)
        spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, MIN_CHARS)
        return snapped(spans, answers[pair.get("pair_id")], "expand")

    result = metrics(pairs, predict)
    return {
        "token_precision": result["token_level"]["precision"],
        "token_recall": result["token_level"]["recall"],
        "token_f1": result["token_level"]["f1"],
        "span_precision": result["span_level"]["precision"],
        "span_recall": result["span_level"]["recall"],
        "span_f1": result["span_level"]["f1"],
        "pair_f1": result["pair_level"]["f1"],
        "fpr_pair": result["fpr"],
        "fpr_token": token_fpr(pairs, predict),
        "type_accuracy": result["type_accuracy"],
    }


def aggregate(rows):
    summary = {}
    for metric in ("token_f1", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "token_precision", "token_recall", "span_precision", "span_recall", "type_accuracy"):
        values = [row[metric] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[metric] = {"mean": round(mean, 4), "std": round(variance ** 0.5, 4)}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--base-oof", default="artifacts/cv/base5")
    parser.add_argument("--feat", default="artifacts/cv/feat5")
    parser.add_argument("--report", default="artifacts/sweep_cv.json")
    args = parser.parse_args(argv)

    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    by_id = {pair.get("pair_id"): pair for pair in gold}
    ours = [by_id[pair_id] for name in ("train", "val") for pair_id in splits[name] if pair_id in by_id]
    ours = [pair for pair in ours if pair.get("model_output")]
    report = {"thresholds": list(THRESHOLDS), "fpr_limit": FPR_LIMIT, "variants": {}}

    base = {}
    for path in sorted(Path(args.base_oof).glob("oof_*.json")):
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            base[key] = {"answer": value["answer"], "pieces": [tuple(piece) for piece in value["pieces"]]}
    base_rows = {threshold: [] for threshold in THRESHOLDS}
    for path in sorted(Path(args.base_oof).glob("oof_*.json")):
        keys = set(json.loads(path.read_text(encoding="utf-8")))
        fold_pairs = [pair for pair in ours if pair.get("pair_id") in keys]
        for threshold in THRESHOLDS:
            base_rows[threshold].append(measure(fold_pairs, base, threshold))
    report["variants"]["base"] = {str(threshold): aggregate(rows) for threshold, rows in base_rows.items()}

    for threshold in THRESHOLDS:
        values = report["variants"]["base"][str(threshold)]
        print(
            f"база, порог {threshold:.2f}: token F1 {values['token_f1']['mean']}±{values['token_f1']['std']} "
            f"span F1 {values['span_f1']['mean']}±{values['span_f1']['std']} FPR_pair {values['fpr_pair']['mean']} "
            f"FPR_token {values['fpr_token']['mean']}",
            flush=True,
        )

    feat_rows = {threshold: [] for threshold in THRESHOLDS}
    for index, folder in enumerate(sorted(Path(args.feat).glob("fold_*")), 1):
        dump_path = Path(args.base_oof) / f"oof_{index}.json"
        keys = set(json.loads(dump_path.read_text(encoding="utf-8"))) if dump_path.exists() else set()
        fold_pairs = [pair for pair in ours if pair.get("pair_id") in keys]
        tokenizer, model = lexfeat.load_saved(folder, "models/rubert-tiny2")
        cache = {}
        for pair in fold_pairs:
            cached = lexfeat.pair_probabilities(tokenizer, model, pair, top=TOP)
            if cached:
                cache[pair.get("pair_id")] = cached
        for threshold in THRESHOLDS:
            feat_rows[threshold].append(measure(fold_pairs, cache, threshold))
        print(f"признаки, фолд {index}: посчитано {len(cache)} пар", flush=True)
    report["variants"]["features"] = {str(threshold): aggregate(rows) for threshold, rows in feat_rows.items()}

    for threshold in THRESHOLDS:
        values = report["variants"]["features"][str(threshold)]
        print(
            f"признаки, порог {threshold:.2f}: token F1 {values['token_f1']['mean']}±{values['token_f1']['std']} "
            f"span F1 {values['span_f1']['mean']}±{values['span_f1']['std']} FPR_pair {values['fpr_pair']['mean']} "
            f"FPR_token {values['fpr_token']['mean']}",
            flush=True,
        )

    chosen = {}
    for name, table in report["variants"].items():
        allowed = [
            (threshold, values)
            for threshold, values in table.items()
            if values["fpr_pair"]["mean"] <= FPR_LIMIT and values["fpr_token"]["mean"] <= FPR_LIMIT
        ]
        pool = allowed or list(table.items())
        best = max(pool, key=lambda item: item[1]["token_f1"]["mean"])
        chosen[name] = {"threshold": float(best[0]), "summary": best[1]}
    report["chosen"] = chosen
    report["target_span_f1"] = 0.2835
    report["span_target_met"] = bool(chosen["features"]["summary"]["span_f1"]["mean"] > 0.2835)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, item in chosen.items():
        print(f"выбрано {name}: порог {item['threshold']} token F1 {item['summary']['token_f1']['mean']} span F1 {item['summary']['span_f1']['mean']}", flush=True)
    print(f"цель span F1 0,2835 достигнута: {report['span_target_met']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
