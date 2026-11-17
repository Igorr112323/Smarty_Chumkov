import json
from pathlib import Path

from tools.analysis48 import ROOT, load
from spanver.snap import snapped
from tools.diagnose46 import without_snap


def main():
    art = ROOT / "artifacts"
    summaries = {v: load(art / f"ablation47_{v}_summary.json") for v in ("v4", "v5")}
    records = {}
    epochs_equal = best_equal = 0
    for outer in range(1, 6):
        for inner in range(1, 6):
            if inner == outer:
                continue
            pair = [load(art / f"ablation47_{v}_outer{outer}_inner{inner}.json") for v in ("v4", "v5")]
            records[outer, inner] = pair
            for a, b in zip(pair[0]["seeds"], pair[1]["seeds"]):
                epochs_equal += a["epochs_ran"] == b["epochs_ran"]
                best_equal += a["best_epoch"] == b["best_epoch"]
    raw_diff = output_diff = bounds_diff = checkpoint_equal = 0
    for fold in range(1, 6):
        a, b = records[fold, fold % 5 + 1]
        checkpoint_equal += sum(x["checkpoint"]["sha256"] == y["checkpoint"]["sha256"] for x, y in zip(a["seeds"], b["seeds"]))
        for row in map(json.loads, (art / f"predictions_fold{fold}.jsonl").read_text(encoding="utf-8").splitlines()):
            predictions = []
            raw_diff += a["outer_predictions"][row["pair_id"]] != b["outer_predictions"][row["pair_id"]]
            for variant, record in (("v4", a), ("v5", b)):
                threshold = summaries[variant]["folds"][fold - 1]["threshold"]
                entry = {**row, "pieces": record["outer_predictions"][row["pair_id"]], "gate_threshold": 0.45, "start_threshold": threshold["start"], "extend_threshold": threshold["extend"]}
                predictions.append(snapped(without_snap(entry), row["answer"], "expand"))
            output_diff += predictions[0] != predictions[1]
            bounds_diff += [(s["start"], s["end"]) for s in predictions[0]] != [(s["start"], s["end"]) for s in predictions[1]]
    result = {"pairs": 699, "raw_probability_difference_pairs": raw_diff, "full_predicted_span_difference_pairs": output_diff, "boundary_difference_pairs": bounds_diff, "same_checkpoint_sha": checkpoint_equal, "checkpoint_count": 25, "same_epochs_ran": epochs_equal, "same_best_epoch": best_equal, "trained_models_per_variant": 100, "note": "отличие полного спана может относиться только к полю risk; равенство метрик не доказывает идентичность моделей"}
    (art / "comparison48.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
