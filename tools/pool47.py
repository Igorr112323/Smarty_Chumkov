import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens
from spanver.snap import snapped
from tools.diagnose46 import tally, without_snap
from tools.evaluate_snap42 import compute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("v4", "v5"))
    args = parser.parse_args()
    artifacts = ROOT / "artifacts"
    summary = json.loads((artifacts / f"ablation47_{args.variant}_summary.json").read_text(encoding="utf-8"))
    sources = json.loads((artifacts / "predictions40_manifest.json").read_text(encoding="utf-8"))
    if summary["status"] != "готово":
        raise ValueError("неполный вариант")
    combined = []
    ids = set()
    for index, source in enumerate(sources["outputs"]):
        payload = (ROOT / source["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != source["sha256"]:
            raise ValueError("не совпадает SHA исходного фолда")
        rows = [json.loads(line) for line in payload.splitlines() if line]
        fold = index + 1
        primary = fold % 5 + 1
        path = artifacts / f"ablation47_{args.variant}_outer{fold}_inner{primary}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        expected = summary["folds"][index]
        raw = report["outer_predictions"]
        if set(raw) != {row["pair_id"] for row in rows}:
            raise ValueError("не совпадают pair_id фолда")
        threshold = expected["threshold"]
        transformed = []
        for row in rows:
            if row["pair_id"] in ids:
                raise ValueError("повтор пары")
            ids.add(row["pair_id"])
            entry = {**row, "pieces": raw[row["pair_id"]], "gate_threshold": 0.45, "start_threshold": threshold["start"], "extend_threshold": threshold["extend"]}
            entry["predicted_spans"] = snapped(without_snap(entry), entry["answer"], "expand")
            transformed.append(entry)
        actual = compute(transformed)
        for name in ("token_precision", "token_recall", "token_f1", "pair_f1", "fpr_pair", "fpr_token"):
            if actual[name] != expected["metrics"][name]:
                raise ValueError(f"не совпадает фолд {fold}: {name}")
        combined.extend(transformed)
    if len(ids) != 699:
        raise ValueError("ожидалось 699 пар")
    counts = tally(combined)
    token_tp = token_fn = 0
    for row in combined:
        gold = covered_tokens(row["answer"], [(span["start"], span["end"]) for span in row["gold_spans"]])
        predicted = covered_tokens(row["answer"], [(span["start"], span["end"]) for span in row["predicted_spans"]])
        token_tp += len(gold & predicted)
        token_fn += len(gold - predicted)
    counts.update({"true_positive_tokens": token_tp, "false_negative_tokens": token_fn})
    metrics = compute(combined)
    denominator = 2 * token_tp + counts["false_positive_tokens"] + token_fn
    exact_f1 = 2 * token_tp / denominator if denominator else 0.0
    if round(exact_f1, 4) != metrics["token_f1"]:
        raise ValueError("сводный token F1 не совпадает")
    output = {"variant": args.variant, "scope": "объединённые сохранённые out-of-fold прогнозы 699 пар, а не независимый отложенный набор", "threshold_reselection": False, "test301": False, "external_data": False, "counts": counts, "pooled_token_f1": metrics["token_f1"], "pooled_token_f1_exact": exact_f1, "pooled_fpr_token": counts["false_positive_tokens"] / counts["negative_tokens"], "pooled_fpr_pair": counts["false_positive_pairs"] / counts["negative_pairs"]}
    (artifacts / f"ablation47_{args.variant}_pooled.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
