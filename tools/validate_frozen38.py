import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import metrics, read_jsonl
from tools.preflight_13 import id_digest
from tools.train_base33 import grouped_folds
from tools.train_cv_feat import token_fpr


def main():
    artifacts = ROOT / "artifacts"
    all_pairs = read_jsonl(artifacts / "ours33.jsonl")
    held = grouped_folds(all_pairs, 5)[0]
    summary = json.loads((artifacts / "nested38_summary.json").read_text(encoding="utf-8"))
    primary = json.loads((artifacts / "nested38_outer_1_inner_2.json").read_text(encoding="utf-8"))
    if id_digest(held) != primary["outer_id_sha256"]:
        raise ValueError("не совпал внешний фолд 1")
    request = artifacts / "frozen38_validation_input.jsonl"
    output = artifacts / "frozen38_validation_output.jsonl"
    request.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in held), encoding="utf-8")
    subprocess.run([sys.executable, "tools/predict_frozen38.py", "--input", str(request), "--output", str(output)], cwd=ROOT, check=True)
    predictions = {row["pair_id"]: row["spans"] for row in read_jsonl(output)}
    if set(predictions) != {row["pair_id"] for row in held}:
        raise ValueError("отсутствуют предсказания")
    def predictor(pair):
        return predictions[pair["pair_id"]]
    result = metrics(held, predictor)
    actual = {f"{level}_{name}": result[f"{level}_level"][name] for level in ("token", "span") for name in ("precision", "recall", "f1")}
    actual.update({"pair_f1": result["pair_level"]["f1"], "fpr_pair": result["fpr"], "fpr_token": token_fpr(held, predictor), "type_accuracy": result["type_accuracy"], "type_correct": result["type_correct"], "matched_spans": result["span_level"]["tp"], "predicted_pairs": sum(bool(value) for value in predictions.values()), "predicted_spans": sum(map(len, predictions.values()))})
    expected = summary["folds"][0]["metrics"]
    if actual != expected:
        raise ValueError(f"восстановленная модель не воспроизвела фолд: {actual} != {expected}")
    for name in ("artifacts/frozen_final.json", "weights/frozen/manifest.json"):
        path = ROOT / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "готово" if name.startswith("weights/") else "готово: сохранён ансамбль первого внешнего фолда"
        value["restore_verified"] = True
        value["restore_verification"] = "степень совпадения: все метрики внешнего фолда 1 после восстановления пяти state_dict"
        if "deployable" in value:
            value["deployable"] = True
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check = {"status": "готово", "outer_fold": 1, "pairs": len(held), "pair_ids_sha256": id_digest(held), "metrics": actual, "test301": False}
    (artifacts / "frozen38_verification.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    request.unlink()
    output.unlink()
    print("Восстановление пяти весов и метрики фолда 1 совпали.")
    return 0


if __name__ == "__main__":
    main()
