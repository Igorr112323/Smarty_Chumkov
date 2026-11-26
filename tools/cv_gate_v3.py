"""Честный CV-gate для v3: обучение на dоле 0, пороги на доле 1, оценка на val + доле 2.

Held-out (test/extra из gold1200_v3) не читается. Разделение mFAVA по SHA-256
документа, поэтому один документ не попадает в разные доли.
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


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / "artifacts/mfava_train_v3.jsonl"
    pairs = [json.loads(line) for line in source.open(encoding="utf-8")]
    folds = {0: [], 1: [], 2: []}
    for pair in pairs:
        bucket = int(hashlib.sha256(pair["document_text"].encode("utf-8")).hexdigest(), 16) % 3
        folds[bucket].append(pair)
    paths = {}
    for key, rows in folds.items():
        paths[key] = ROOT / f"artifacts/mfava_train_v3_fold{key}.jsonl"
        dump(paths[key], rows)

    weights = ROOT / "artifacts/ranker_weights_v3_fold0.json"
    commands = [
        [sys.executable, str(ROOT / "tools/train_ranker.py"),
         "--gold", "artifacts/gold_v3.jsonl",
         "--extra", str(paths[0].relative_to(ROOT)),
         "--weights", str(weights.relative_to(ROOT)),
         "--report", "artifacts/ranker_report_v3_fold0.json"],
        [sys.executable, str(ROOT / "tools/train_local_ranker.py"),
         "--gold", "artifacts/gold_v3.jsonl",
         "--extra", str(paths[1].relative_to(ROOT)),
         "--weights", str(weights.relative_to(ROOT)),
         "--report", "artifacts/local_ranker_report_v3_fold0.json"],
    ]
    for command in commands:
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    ranker._WEIGHTS = json.loads(weights.read_text(encoding="utf-8"))
    val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")

    def predictor(pair):
        return ranker.select_spans(pair["model_output"] or "", pair["document_text"] or "")

    def cell(result):
        return {
            "span": result["span_level"]["f1"],
            "token": result["token_level"]["f1"],
            "pair": result["pair_level"]["f1"],
            "fpr": result["fpr"],
            "iou": result.get("iou_threshold"),
        }

    combined = metrics(val + folds[2], predictor)
    report = {
        "protocol": "train=RusHallu train + mFAVA fold0; thresholds=val RusHallu + mFAVA fold1; report=val RusHallu + mFAVA fold2; held-out не читается",
        "fold_sizes": {key: len(rows) for key, rows in folds.items()},
        "fold_sha256": {str(key): hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "config": {
            "gate": ranker._WEIGHTS.get("config", {}).get("gate"),
            "threshold": ranker._WEIGHTS.get("config", {}).get("threshold"),
            "max_spans": ranker._WEIGHTS.get("config", {}).get("max_spans"),
            "nms_iou": ranker._WEIGHTS.get("config", {}).get("nms_iou"),
            "mode_cut": ranker._WEIGHTS.get("mode_cut"),
            "local_config": ranker._WEIGHTS.get("local_config"),
        },
        "dev_combined": cell(combined),
        "dev_rushallu_val": cell(metrics(val, predictor)),
        "dev_mfava_fold2": cell(metrics(folds[2], predictor)),
        "full_dev_metrics": combined,
    }
    report["guard_pass"] = (
        report["dev_combined"]["span"] >= 0.5
        and report["dev_combined"]["token"] >= 0.5
        and report["dev_combined"]["pair"] >= 0.5
        and report["dev_combined"]["fpr"] <= 0.4
    )
    target = ROOT / "artifacts/cv_gate_v3.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
