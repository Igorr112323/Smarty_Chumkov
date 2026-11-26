"""V9: честный 3-fold OOF CV для v3 local head.

В v8 CV-fold обучался на 55 mFAVA-парах, тогда как финальная конфигурация
видит 148. Здесь каждый фолд обучается на 2/3 (≈99 пар) и оценивается только
на своей доле; OOF-предсказания агрегируются. Held-out не читается.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import ranker
from spanver.evaluate import load_split, metrics


def cell(result):
    return {
        "span": result["span_level"]["f1"],
        "token": result["token_level"]["f1"],
        "pair": result["pair_level"]["f1"],
        "fpr": result["fpr"],
    }


def main():
    folds = {}
    for key in range(3):
        path = ROOT / f"artifacts/mfava_train_v3_fold{key}.jsonl"
        folds[key] = [json.loads(line) for line in path.open(encoding="utf-8")]
    val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")
    oof = []
    report = {"protocol": "3-fold OOF; each fold trained on 2/3 mFAVA + RusHallu train; evaluated on held-out fold; val не используется в обучении весов, только в порогах", "folds": {}}

    for held_key in range(3):
        train_keys = [key for key in range(3) if key != held_key]
        train_path = ROOT / f"artifacts/mfava_cv3_train_fold{held_key}.jsonl"
        with train_path.open("w", encoding="utf-8") as handle:
            for key in train_keys:
                for pair in folds[key]:
                    handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        weights = ROOT / f"artifacts/ranker_weights_cv3_fold{held_key}.json"
        commands = [
            [sys.executable, str(ROOT / "tools/train_ranker.py"),
             "--gold", "artifacts/gold_v3.jsonl",
             "--extra", str(train_path.relative_to(ROOT)),
             "--weights", str(weights.relative_to(ROOT)),
             "--report", f"artifacts/ranker_report_cv3_fold{held_key}.json"],
            [sys.executable, str(ROOT / "tools/train_local_ranker.py"),
             "--gold", "artifacts/gold_v3.jsonl",
             "--extra", str(train_path.relative_to(ROOT)),
             "--weights", str(weights.relative_to(ROOT)),
             "--report", f"artifacts/local_ranker_report_cv3_fold{held_key}.json"],
        ]
        for command in commands:
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        ranker._WEIGHTS = json.loads(weights.read_text(encoding="utf-8"))

        def predictor(pair):
            return ranker.select_spans(pair.get("model_output") or "", pair.get("document_text") or "")

        fold = folds[held_key]
        fold_metric = metrics(fold, predictor)
        report["folds"][str(held_key)] = {
            "train_pairs": sum(len(folds[key]) for key in train_keys),
            "eval_pairs": len(fold),
            "metrics": cell(fold_metric),
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        }
        for pair in fold:
            oof.append({
                "pair_id": pair["pair_id"],
                "spans": predictor(pair),
                "gold": pair.get("spans") or [],
                "model_output": pair.get("model_output") or "",
            })
        print("fold", held_key, report["folds"][str(held_key)]["metrics"], flush=True)

    combined = metrics(folds[0] + folds[1] + folds[2],
                       lambda pair: next(row["spans"] for row in oof if row["pair_id"] == pair["pair_id"]))
    report["oof_combined"] = cell(combined)
    report["full_metrics"] = combined
    report["guard_pass"] = (
        report["oof_combined"]["span"] >= 0.5
        and report["oof_combined"]["token"] >= 0.5
        and report["oof_combined"]["pair"] >= 0.5
        and report["oof_combined"]["fpr"] <= 0.4
    )
    target = ROOT / "artifacts/cv3_oof_v9.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "artifacts/cv3_oof_predictions_v9.json").write_text(
        json.dumps(oof, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["oof_combined"], ensure_ascii=False), "guard", report["guard_pass"], flush=True)


if __name__ == "__main__":
    main()
