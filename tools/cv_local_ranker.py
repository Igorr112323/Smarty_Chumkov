"""GroupKFold CV для выбора гиперпараметров локального каскада.

Только train RusHallu + 150 mFAVA из artifacts/mfava_train_v2.jsonl.
Held-out пар в CV нет. Результат — artifacts/local_cv_report.json.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans, document_context

CONFIGS = (
    (4, 31, 150, 0.1, 0.0),
    (3, 15, 150, 0.1, 0.0),
    (4, 31, 150, 0.05, 1.0),
    (2, 7, 100, 0.1, 0.0),
)


def rows(pairs):
    result = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        context = document_context(document)
        for start, end, fragment in candidate_spans(answer):
            if start == 0 and end == len(answer) - 1:
                continue
            features = ranker.features(fragment, start, end, answer, document, context[0], context[2])
            label = int(any(iou(start, end, span["start"], span["end"]) >= 0.5 for span in pair["spans"] or []))
            result.append((pair["pair_id"], features, label, start, end))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--extra", default="artifacts/mfava_train_v2.jsonl")
    parser.add_argument("--report", default="artifacts/local_cv_report.json")
    args = parser.parse_args(argv)

    train = load_split(args.gold, "train")
    extra = read_jsonl(Path(args.extra))
    local_pairs = [
        pair
        for pair in train + extra
        if pair in extra
        or (
            len(pair.get("spans") or []) > 1
            and not any(
                span["start"] == 0 and span["end"] == len(pair.get("model_output") or "") - 1
                for span in pair["spans"] or []
            )
        )
    ]
    data = rows(local_pairs)
    X = np.array([item[1] for item in data], dtype=float)
    y = np.array([item[2] for item in data], dtype=int)
    groups = np.array([item[0] for item in data])

    report = {
        "seed": 0,
        "folds": 4,
        "pairs": len(local_pairs),
        "rows": len(data),
        "positives": int(y.sum()),
        "python": platform.python_version(),
        "versions": {"numpy": np.__version__},
        "grid": [],
    }
    for depth, leaves, iterations, learning_rate, l2 in CONFIGS:
        oof = np.zeros(len(data), dtype=float)
        for train_index, test_index in GroupKFold(n_splits=4).split(X, y, groups):
            model = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_depth=depth,
                max_leaf_nodes=leaves,
                max_iter=iterations,
                l2_regularization=l2,
                random_state=0,
            ).fit(X[train_index], y[train_index])
            oof[test_index] = model.predict_proba(X[test_index])[:, 1]
        cache = {}
        for index, (pid, _, _, start, end) in enumerate(data):
            cache.setdefault(pid, []).append((start, end, float(oof[index])))

        best = (0.0, None)
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
            for max_spans in (2, 3, 4, 5):
                def predictor(pair, threshold=threshold, max_spans=max_spans):
                    chosen = []
                    for start, end, score in sorted(cache[pair["pair_id"]], key=lambda item: -item[2]):
                        if score < threshold or len(chosen) >= max_spans:
                            break
                        if any(iou(start, end, other[0], other[1]) > 0.1 for other in chosen):
                            continue
                        chosen.append((start, end, score))
                    return [
                        {"start": start, "end": end, "risk": score, "type": "unsupported_addition", "confidence": 1.0}
                        for start, end, score in chosen
                    ]

                result = metrics(local_pairs, predictor)
                if result["span_level"]["f1"] > best[0]:
                    best = (
                        result["span_level"]["f1"],
                        {
                            "threshold": threshold,
                            "max_spans": max_spans,
                            "span_f1": result["span_level"]["f1"],
                            "token_f1": result["token_level"]["f1"],
                            "pair_f1": result["pair_level"]["f1"],
                            "fpr": result["fpr"],
                        },
                    )
        report["grid"].append(
            {
                "max_depth": depth,
                "max_leaf_nodes": leaves,
                "max_iter": iterations,
                "learning_rate": learning_rate,
                "l2_regularization": l2,
                "best": best[1],
            }
        )
        print(json.dumps(report["grid"][-1], ensure_ascii=False), flush=True)

    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
