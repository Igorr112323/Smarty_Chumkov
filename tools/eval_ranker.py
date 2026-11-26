"""Финальная оценка ранкера на удержанных данных без подбора порогов.

Пороги и веса берутся из spanver/ranker_weights.json (обучено на train + 48
дополнительных пар mFAVA). Сплиты test (301) и extra (200) не участвуют в
обучении и подборе гиперпараметров.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from spanver import ranker
from spanver.evaluate import load_split, metrics, read_jsonl
from spanver.module import verify


def evaluator(use_ranker):
    if use_ranker:
        def predict(pair):
            return ranker.select_spans(
                pair.get("model_output") or "", pair.get("document_text") or ""
            )
        return predict
    def predict(pair):
        return verify(
            pair.get("model_output") or "",
            pair.get("document_text") or "",
            threshold=0.98,
            use_ranker=False,
        )["spans"]
    return predict


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/gold1200.jsonl")
    parser.add_argument("--report", default="artifacts/ranker_eval.json")
    parser.add_argument("--source-weights", default="artifacts/ranker_report.json")
    args = parser.parse_args(argv)

    gold = Path(args.gold)
    pairs = read_jsonl(gold)
    test = [pair for pair in pairs if pair.get("split") == "test"]
    extra = [pair for pair in pairs if pair.get("split") == "extra"]
    slices = {
        f"test{len(test)}": test,
        f"extra{len(extra)}": extra,
        f"held{len(test) + len(extra)}": test + extra,
    }
    results = {}
    for name, subset in slices.items():
        results[name] = {
            "ranker": metrics(subset, evaluator(True)),
            "lexical_baseline": metrics(subset, evaluator(False)),
        }
    weights_path = ROOT / "spanver" / "ranker_weights.json"
    report = {
        "dataset": str(gold),
        "dataset_sha256": hashlib.sha256(gold.read_bytes()).hexdigest(),
        "config": ranker._model().get("config"),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "split_sizes": {key: len(value) for key, value in slices.items()},
        "results": results,
        "notes": "Пороги зафиксированы на train/val до оценки; test301 и extra200 — только финальная проверка.",
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
