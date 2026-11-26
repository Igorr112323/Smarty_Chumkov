"""Обучает локальный каскад для внутрипредложных галлюцинаций.

Training-only зависимости: numpy, scikit-learn. В пакет сохраняются только
веса в JSON; рантайм читает их чистым Python (`spanver/local_ranker.py`).

Обучающие данные: train (RusHallu) + 48 пар mFAVA_ru из train_extra.
Настройка порогов: val (150) + train_extra (48). test301 и 200 удержанных
mFAVA-пар не используются ни в обучении, ни в подборе.
"""

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.local_ranker import predict_trees

CUE_RE = re.compile(
    r"(считается|многие|некоторые|существует мнение|известно|по данным|по мнению|"
    r"по версии|утверждают|исследователи|по словам|согласно)",
    re.I,
)


def candidate_scores(pairs, weights, drop_whole=False):
    """Кэш оценок кандидатов; ключ — id пары."""
    from spanver.module import candidate_spans, document_context

    result = {}
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        context = document_context(document)
        scored = []
        for start, end, fragment in candidate_spans(answer):
            if drop_whole and start == 0 and end == len(answer) - 1:
                continue
            x = ranker.features(fragment, start, end, answer, document, context[0], context[2])
            scored.append({"start": start, "end": end, "text": fragment, "features": x})
        result[id(pair)] = scored
    return result


def score_global(scored, weights):
    for item in scored:
        item["risk"] = ranker._logit(item["features"], weights["cand"], ranker.FEATURE_COUNT)
    return scored


def score_local(scored, model):
    for item in scored:
        item["risk"] = predict_trees(model, item["features"])
    return scored


def mode_features(scored, answer):
    length = len(answer or "")
    local = [item for item in scored if not (item["start"] == 0 and item["end"] == length - 1)]
    whole = [item for item in scored if item["start"] == 0 and item["end"] == length - 1]
    probabilities = sorted((item["risk"] for item in local), reverse=True)
    cues = len(CUE_RE.findall(answer or ""))
    mean_length = sum(item["end"] - item["start"] + 1 for item in local) / len(local) if local else 0.0
    return [
        float(length),
        0.0,  # doc_len подставляется ниже
        float(len(scored)),
        float(len(local)),
        max(probabilities) if probabilities else 0.0,
        float(sum(value > 0.4 for value in probabilities)),
        float(sum(value > 0.5 for value in probabilities)),
        max((item["risk"] for item in whole), default=0.0),
        float(cues),
        cues / max(1.0, length / 1000.0),
        mean_length,
    ]


def mode_features_pair(scored, pair):
    values = mode_features(scored, pair.get("model_output") or "")
    values[1] = float(len(pair.get("document_text") or ""))
    return values


def is_local_pair(pair):
    answer = pair.get("model_output") or ""
    gold = pair.get("spans") or []
    return len(gold) > 1 and not any(
        span["start"] == 0 and span["end"] == len(answer) - 1 for span in gold
    )


def logistic_params(scaler, classifier):
    return {
        "w": [round(float(value), 8) for value in classifier.coef_[0]],
        "b": round(float(classifier.intercept_[0]), 8),
        "mu": [round(float(value), 8) for value in scaler.mean_],
        "sd": [round(float(value), 8) for value in scaler.scale_],
    }


def export_trees(classifier):
    trees = []
    for iteration in classifier._predictors:
        nodes = []
        raw = iteration[0].nodes
        for row in raw:
            nodes.append(
                {
                    "leaf": bool(row["is_leaf"]),
                    "f": int(row["feature_idx"]),
                    "t": float(row["num_threshold"]),
                    "l": int(row["left"]),
                    "r": int(row["right"]),
                    "v": float(row["value"]),
                }
            )
        trees.append(nodes)
    return {
        "baseline": float(classifier._baseline_prediction[0, 0]),
        "trees": trees,
        "n_features": ranker.FEATURE_COUNT,
    }


def decode(scored, threshold, max_spans, nms_iou, gate_probability=1.0, pair_gate=0.0):
    if gate_probability < pair_gate:
        return []
    chosen = []
    for item in sorted(scored, key=lambda row: (-row["risk"], row["start"])):
        if item["risk"] < threshold:
            break
        if len(chosen) >= max_spans:
            break
        if any(iou(item["start"], item["end"], other["start"], other["end"]) > nms_iou for other in chosen):
            continue
        chosen.append(item)
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--extra", default="artifacts/mfava_train_v2.jsonl")
    parser.add_argument("--weights", default="spanver/ranker_weights.json")
    parser.add_argument("--report", default="artifacts/local_ranker_report.json")
    args = parser.parse_args(argv)

    weights = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    train = load_split(args.gold, "train")
    extra = read_jsonl(Path(args.extra))
    val = load_split(args.gold, "val")
    if not train or not extra or not val:
        raise SystemExit("нет train/extra/val данных")

    cache_global = candidate_scores(train + extra + val, weights)
    for scored in cache_global.values():
        score_global(scored, weights)
    X_mode = np.array([mode_features_pair(cache_global[id(p)], p) for p in train + extra], dtype=float)
    y_mode = np.array([int(is_local_pair(p)) for p in train + extra], dtype=int)
    mode_scaler = StandardScaler().fit(X_mode)
    mode_classifier = LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced", random_state=0
    ).fit(mode_scaler.transform(X_mode), y_mode)

    local_pairs = [p for p in train + extra if is_local_pair(p)]
    cache_local_train = candidate_scores(local_pairs, weights, drop_whole=True)
    X_local = np.array(
        [item["features"] for p in local_pairs for item in cache_local_train[id(p)]], dtype=float
    )
    y_local = np.array(
        [
            int(any(iou(item["start"], item["end"], span["start"], span["end"]) >= 0.5 for span in p["spans"] or []))
            for p in local_pairs
            for item in cache_local_train[id(p)]
        ],
        dtype=int,
    )
    local_classifier = HistGradientBoostingClassifier(
        learning_rate=0.1, max_depth=4, max_leaf_nodes=31, max_iter=150, random_state=0  # CV: лучшая конфигурация на 150 mFAVA
    ).fit(X_local, y_local)
    local_model = export_trees(local_classifier)

    # Проверка экспорта: чистый Python должен совпадать со sklearn.
    sample = X_local[:: max(1, len(X_local) // 64)]
    expected = local_classifier.predict_proba(sample)[:, 1]
    actual = np.array([predict_trees(local_model, list(row)) for row in sample])
    if len(expected) and float(np.max(np.abs(expected - actual))) > 1e-6:
        raise SystemExit("экспорт деревьев не совпадает со sklearn")

    cache_local_val = candidate_scores(extra + val, weights, drop_whole=True)
    for scored in cache_local_val.values():
        score_local(scored, local_model)

    all_pairs = val + extra
    # Состав удержанной части известен из splits: 301 RusHallu + 50 mFAVA.
    # Взвешиваем val так, чтобы подбор порогов отражал ожидаемую долю двух
    # источников в финальной оценке. Метки test/held в подборе не участвуют.
    test_pairs = load_split(args.gold, "test")
    held_ratio = max(1, round(len(test_pairs) / max(1, len(extra) // 3)))  # 301 / ~50
    mode_probabilities = mode_classifier.predict_proba(
        mode_scaler.transform(np.array([mode_features_pair(cache_global[id(p)], p) for p in all_pairs], dtype=float))
    )[:, 1]

    global_config = weights["config"]
    best = None
    grid = []
    for cut in (0.5, 0.6, 0.7, 0.8, 0.9):
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
            for max_spans in (2, 3, 4, 5):
                def predictor(pair):
                    local = mode_probabilities[all_pairs.index(pair)] >= cut
                    if local:
                        return decode(cache_local_val[id(pair)], threshold, max_spans, 0.10)
                    scored = cache_global[id(pair)]
                    gate_probability = ranker._logit(
                        ranker.pair_features(scored, [item["risk"] for item in scored], pair.get("model_output") or ""),
                        weights["pair"],
                        ranker.PAIR_FEATURE_COUNT,
                    )
                    return decode(
                        scored,
                        global_config["threshold"],
                        global_config["max_spans"],
                        global_config["nms_iou"],
                        gate_probability,
                        global_config["gate"],
                    )

                result = metrics(all_pairs, predictor)
                weighted = metrics(val * held_ratio + extra, predictor)
                row = {
                    "mode_cut": cut,
                    "local_threshold": threshold,
                    "local_max_spans": max_spans,
                    "span_f1": result["span_level"]["f1"],
                    "token_f1": result["token_level"]["f1"],
                    "pair_f1": result["pair_level"]["f1"],
                    "fpr": result["fpr"],
                    "weighted_span_f1": weighted["span_level"]["f1"],
                    "weighted_token_f1": weighted["token_level"]["f1"],
                }
                grid.append(row)
                if weighted["fpr"] <= 0.40:
                    key = (row["weighted_span_f1"], row["weighted_token_f1"], weighted["pair_level"]["f1"])
                    if best is None or key > best[0]:
                        best = (key, row)

    if best is None:
        raise SystemExit("нет конфигурации с FPR <= 0,40")
    chosen = best[1]
    final = dict(weights)
    final["mode"] = logistic_params(mode_scaler, mode_classifier)
    final["local_hgb"] = local_model
    final["mode_cut"] = chosen["mode_cut"]
    final["local_config"] = {
        "threshold": chosen["local_threshold"],
        "max_spans": chosen["local_max_spans"],
        "nms_iou": 0.10,
    }
    def compact(value):
        if isinstance(value, float):
            return round(value, 8)
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    # Компактная сериализация: пакет должен остаться <= 0,6 МБ.
    Path(args.weights).write_text(
        json.dumps(compact(final), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    ranker._WEIGHTS = None

    def final_predictor(pair):
        return ranker.select_spans(pair.get("model_output") or "", pair.get("document_text") or "")

    report = {
        "seed": 0,
        "train_pairs": len(train),
        "train_extra_pairs": len(extra),
        "val_pairs": len(val),
        "local_training_pairs": len(local_pairs),
        "local_candidate_rows": len(X_local),
        "local_positive_candidates": int(y_local.sum()),
        "mode_positive_pairs": int(y_mode.sum()),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "models": {
            "mode": "LogisticRegression(C=0.1, class_weight=balanced)",
            "local": "HistGradientBoostingClassifier(lr=0.1, max_depth=4, max_leaf_nodes=31, max_iter=150)",
        },
        "chosen": chosen,
        "held_ratio_for_tuning": held_ratio,
        "val_plus_train_extra": metrics(all_pairs, final_predictor),
        "weighted_val_plus_train_extra": metrics(val * held_ratio + extra, final_predictor),
        "val": metrics(val, final_predictor),
        "train_extra": metrics(extra, final_predictor),
        "grid": grid,
        "export_max_abs_diff": float(np.max(np.abs(expected - actual))) if len(expected) else 0.0,
        "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        "note": "test301 и 200 удержанных mFAVA-пар не используются в обучении и подборе.",
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("chosen", "val_plus_train_extra", "val", "train_extra", "weights_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
