import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import metrics, read_jsonl
from tools.nested38 import file_digest
from tools.preflight_13 import id_digest
from tools.train_base33 import grouped_folds
from tools.train_cv_feat import token_fpr
from tools.train_scheme13_full import save_json


def main():
    artifacts = ROOT / "artifacts"
    summary = json.loads((artifacts / "nested38_summary.json").read_text(encoding="utf-8"))
    source = artifacts / "ours33.jsonl"
    if file_digest(source) != summary["gold_train_val_sha256"]:
        raise ValueError("неверный train/val")
    held = grouped_folds(read_jsonl(source), 5)[0]
    if id_digest(held) != summary["fold_ids_sha256"][0]:
        raise ValueError("неверный внешний фолд")
    request = artifacts / "verify_final45_input.jsonl"
    output = artifacts / "verify_final45_output.jsonl"
    blind = [{key: value for key, value in row.items() if key not in ("spans", "label")} for row in held]
    request.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind), encoding="utf-8")
    subprocess.run([sys.executable, "tools/predict_frozen38.py", "--weights", "weights/final", "--input", str(request), "--output", str(output)], cwd=ROOT, check=True)
    predictions = {row["pair_id"]: row["spans"] for row in read_jsonl(output)}
    if set(predictions) != {row["pair_id"] for row in held}:
        raise ValueError("отсутствуют предсказания")
    def predictor(row):
        return predictions[row["pair_id"]]
    result = metrics(held, predictor)
    actual = {f"{level}_{name}": result[f"{level}_level"][name] for level in ("token", "span") for name in ("precision", "recall", "f1")}
    actual.update({"pair_f1": result["pair_level"]["f1"], "fpr_pair": result["fpr"], "fpr_token": token_fpr(held, predictor), "type_accuracy": result["type_accuracy"], "type_correct": result["type_correct"], "matched_spans": result["span_level"]["tp"], "predicted_pairs": sum(bool(value) for value in predictions.values()), "predicted_spans": sum(map(len, predictions.values()))})
    expected = summary["folds"][0]["metrics"]
    report = {"outer_fold": 1, "pairs": len(held), "pair_ids_sha256": id_digest(held), "metrics": actual, "expected_metrics": expected, "matches_original": actual == expected, "weights": "weights/final", "test301": False, "external_data": False}
    save_json(artifacts / "final45_restore_verification.json", report)
    request.unlink()
    output.unlink()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["matches_original"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
