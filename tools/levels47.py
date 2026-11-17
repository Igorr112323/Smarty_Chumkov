import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import metrics
from spanver.snap import snapped
from tools.diagnose46 import without_snap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v2", "v4", "v5"), required=True)
    args = parser.parse_args()
    artifacts = ROOT / "artifacts"
    prefix = "ablation46" if args.variant == "v2" else "ablation47"
    summary = json.loads((artifacts / f"{prefix}_{args.variant}_summary.json").read_text(encoding="utf-8"))
    sources = json.loads((artifacts / "predictions40_manifest.json").read_text(encoding="utf-8"))
    if summary["status"] != "готово" or len(sources["outputs"]) != 5:
        raise ValueError("неполный вариант")
    folds, combined = [], []
    ids = set()
    for index, source in enumerate(sources["outputs"]):
        payload = (ROOT / source["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != source["sha256"]:
            raise ValueError("неверный SHA источника")
        rows = [json.loads(line) for line in payload.splitlines() if line]
        fold = index + 1
        inner = fold % 5 + 1
        record = json.loads((artifacts / f"{prefix}_{args.variant}_outer{fold}_inner{inner}.json").read_text(encoding="utf-8"))
        expected = summary["folds"][index]
        if set(record["outer_predictions"]) != {row["pair_id"] for row in rows}:
            raise ValueError("не совпадают ID внешнего фолда")
        prepared = []
        for row in rows:
            if row["pair_id"] in ids:
                raise ValueError("повтор пары")
            ids.add(row["pair_id"])
            entry = {**row, "pieces": record["outer_predictions"][row["pair_id"]], "gate_threshold": 0.45, "start_threshold": expected["threshold"]["start"], "extend_threshold": expected["threshold"]["extend"]}
            predicted = snapped(without_snap(entry), entry["answer"], "expand")
            prepared.append({"pair_id": row["pair_id"], "model_output": row["answer"], "spans": row["gold_spans"], "predicted_spans": predicted})
        result = metrics(prepared, lambda row: row["predicted_spans"])
        for level in ("token", "span"):
            for name in ("precision", "recall", "f1"):
                if result[f"{level}_level"][name] != expected["metrics"][f"{level}_{name}"]:
                    raise ValueError(f"не совпадает {fold} {level} {name}")
        if result["pair_level"]["f1"] != expected["metrics"]["pair_f1"] or result["fpr"] != expected["metrics"]["fpr_pair"]:
            raise ValueError("не совпадают парные метрики")
        folds.append(result)
        combined.extend(prepared)
    if len(ids) != 699:
        raise ValueError("ожидалось 699 пар")
    macro = {level: {name: round(statistics.mean(fold[f"{level}_level"][name] for fold in folds), 4) for name in ("precision", "recall", "f1")} for level in ("token", "span", "pair")}
    pooled = metrics(combined, lambda row: row["predicted_spans"])
    out = {"variant": args.variant, "scope": "699 out-of-fold train/val, не независимый тест", "macro": macro, "pooled": {level: pooled[f"{level}_level"] for level in ("token", "span", "pair")}, "test301": False, "external_data": False}
    target = artifacts / f"ablation47_{args.variant}_levels.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
