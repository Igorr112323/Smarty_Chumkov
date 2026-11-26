"""Обучение двухкаскадного лексического ранкера.

Training-only зависимости: numpy, scikit-learn. В рантайме нужен только stdlib:
веса сохраняются в spanver/ranker_weights.json и читаются spanver/ranker.py.

Подбор порогов выполняется ТОЛЬКО на val (150 размеченных пар); test и 200
удержанных mFAVA-пар в обучении/подборе не участвуют.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl


def build_rows(pairs):
    from spanver.module import candidate_spans, document_context

    rows = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        gold = pair.get("spans") or []
        context = document_context(document)
        for start, end, fragment in candidate_spans(answer):
            x = ranker.features(fragment, start, end, answer, document, context[0], context[2])
            label = int(any(iou(start, end, item["start"], item["end"]) >= 0.5 for item in gold))
            rows.append((pair.get("pair_id"), x, label, bool(gold), len(answer)))
    return rows


def params_of(scaler, classifier):
    return {
        "w": [round(float(value), 8) for value in classifier.coef_[0]],
        "b": round(float(classifier.intercept_[0]), 8),
        "mu": [round(float(value), 8) for value in scaler.mean_],
        "sd": [round(float(value), 8) for value in scaler.scale_],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--extra", default="artifacts/mfava_train_extra.jsonl")
    parser.add_argument("--weights", default="spanver/ranker_weights.json")
    parser.add_argument("--report", default="artifacts/ranker_report.json")
    args = parser.parse_args(argv)

    train = load_split(args.gold, "train")
    extra = read_jsonl(Path(args.extra)) if Path(args.extra).exists() else []
    val = load_split(args.gold, "val")
    rows = build_rows(train + extra)
    X = np.array([row[1] for row in rows], dtype=float)
    y = np.array([row[2] for row in rows], dtype=int)

    scale = StandardScaler().fit(X)
    classifier = LogisticRegression(
        C=1.0, max_iter=2000, class_weight={0: 1, 1: 4}, random_state=0, solver="lbfgs"
    ).fit(scale.transform(X), y)
    scores = classifier.predict_proba(scale.transform(X))[:, 1]

    pair_rows = {}
    for index, (pid, x, label, has_gold, answer_len) in enumerate(rows):
        entry = pair_rows.setdefault(pid, {"features": [], "scores": [], "label": has_gold, "len": answer_len})
        entry["features"].append(x)
        entry["scores"].append(float(scores[index]))

    Xp = np.array(
        [ranker.pair_features(entry["features"], entry["scores"], "x" * entry["len"]) for entry in pair_rows.values()],
        dtype=float,
    )
    yp = np.array([int(entry["label"]) for entry in pair_rows.values()], dtype=int)
    pair_scale = StandardScaler().fit(Xp)
    pair_classifier = LogisticRegression(
        C=1.0, max_iter=2000, class_weight={0: 1, 1: 2}, random_state=0, solver="lbfgs"
    ).fit(pair_scale.transform(Xp), yp)

    # Предварительно пишем веса, чтобы val оценивался через рантайм-код.
    candidate_params = params_of(scale, classifier)
    pair_params = params_of(pair_scale, pair_classifier)
    initial = {
        "cand": candidate_params,
        "pair": pair_params,
        "config": {"gate": 0.50, "threshold": 0.60, "max_spans": 1, "nms_iou": 0.10},
    }
    Path(args.weights).write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ranker._WEIGHTS = None

    best = None
    grid = []
    for gate in (0.40, 0.50, 0.60):
        for threshold in (0.50, 0.60, 0.70):
            for max_spans in (1, 2):
                def predictor(pair, gate=gate, threshold=threshold, max_spans=max_spans):
                    return ranker.select_spans(
                        pair.get("model_output") or "",
                        pair.get("document_text") or "",
                        gate=gate,
                        threshold=threshold,
                        max_spans=max_spans,
                        nms_iou=0.10,
                    )

                result = metrics(val, predictor)
                row = {
                    "gate": gate,
                    "threshold": threshold,
                    "max_spans": max_spans,
                    "span_f1": result["span_level"]["f1"],
                    "token_f1": result["token_level"]["f1"],
                    "pair_f1": result["pair_level"]["f1"],
                    "fpr": result["fpr"],
                }
                grid.append(row)
                if result["fpr"] <= 0.40:
                    key = (row["span_f1"], row["token_f1"])
                    if best is None or key > best[0]:
                        best = (key, row)
    if best is None:
        raise SystemExit("на val нет конфигурации с FPR <= 0,40")
    config = {
        "gate": best[1]["gate"],
        "threshold": best[1]["threshold"],
        "max_spans": best[1]["max_spans"],
        "nms_iou": 0.10,
    }
    final = {"cand": candidate_params, "pair": pair_params, "config": config}
    Path(args.weights).write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ranker._WEIGHTS = None

    def best_predictor(pair):
        return ranker.select_spans(
            pair.get("model_output") or "",
            pair.get("document_text") or "",
            gate=config["gate"],
            threshold=config["threshold"],
            max_spans=config["max_spans"],
            nms_iou=config["nms_iou"],
        )

    val_result = metrics(val, best_predictor)
    report = {
        "seed": 0,
        "train_pairs": len(train),
        "train_extra_pairs": len(extra),
        "val_pairs": len(val),
        "candidate_rows": len(rows),
        "positive_candidates": int(y.sum()),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "model": {
            "candidate": "LogisticRegression(C=1, class_weight={0:1,1:4})",
            "pair_gate": "LogisticRegression(C=1, class_weight={0:1,1:2})",
        },
        "config": config,
        "val": val_result,
        "grid": grid,
        "weights": args.weights,
        "weights_sha256": __import__("hashlib").sha256(Path(args.weights).read_bytes()).hexdigest(),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": config, "val": val_result, "weights_sha256": report["weights_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
