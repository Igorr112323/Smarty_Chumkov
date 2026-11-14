import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spanver.evaluate import evaluate_pairs, load_split
from spanver.module import MODEL_FEATURES, candidate_spans, document_context, feature_vector, predict_type

EPOCHS = 400
LEARNING_RATE = 1.2
L2 = 1e-4


def build_rows(pairs):
    rows = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        gold = pair.get("spans") or []
        document = pair.get("document_text") or ""
        doc_tokens, doc_stems, doc_ngrams = document_context(document)
        for start, end, text in candidate_spans(answer):
            whole = start == 0 and end == len(answer) - 1
            values, support = feature_vector(text, len(answer), doc_tokens, whole, doc_stems, doc_ngrams, start, len(document))
            label = 0
            kind = predict_type(text, support)
            for item in gold:
                intersection = min(end, item["end"]) - max(start, item["start"]) + 1
                if intersection <= 0:
                    continue
                union = max(end, item["end"]) - min(start, item["start"]) + 1
                if intersection / union >= 0.5 and item["type"] == kind:
                    label = 1
                    break
            rows.append((values, label))
    return rows


def fit(rows):
    positives = sum(label for _, label in rows)
    negatives = len(rows) - positives
    weight_positive = negatives / positives if positives else 1.0
    vector = [0.0] * len(MODEL_FEATURES)
    for _ in range(EPOCHS):
        gradient = [0.0] * len(vector)
        for values, label in rows:
            total = sum(weight * value for weight, value in zip(vector, values))
            prediction = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))
            error = prediction - label
            if label == 1:
                error *= weight_positive
            for index, value in enumerate(values):
                gradient[index] += error * value
        norm = float(len(rows))
        for index in range(len(vector)):
            vector[index] -= LEARNING_RATE * (gradient[index] / norm + L2 * vector[index])
    return vector


def main():
    train = load_split("artifacts/gold.jsonl", "train")
    val = load_split("artifacts/gold.jsonl", "val")
    rows = build_rows(train)
    positives = sum(label for _, label in rows)
    print(f"train pairs = {len(train)}, candidates = {len(rows)}, positives = {positives}")
    vector = fit(rows)
    print("weights = " + json.dumps([round(value, 4) for value in vector]))
    best = None
    for threshold in [round(0.20 + step * 0.01, 2) for step in range(80)]:
        metrics = evaluate_pairs(val, threshold=threshold, weights=vector)
        span = metrics["span_level"]
        print(f"threshold {threshold:.2f} F1 {span['f1']:.4f} P {span['precision']:.4f} R {span['recall']:.4f} FPR {metrics['fpr']:.4f} tp {span['tp']} fp {span['fp']} fn {span['fn']}")
        if metrics["fpr"] <= 0.40 and (best is None or span["f1"] > best[0]):
            best = (span["f1"], threshold, metrics)
    if best:
        print(f"best F1 {best[0]:.4f} at threshold {best[1]:.2f}")
        Path("artifacts/model_weights.json").write_text(
            json.dumps({"weights": [round(value, 6) for value in vector], "threshold": best[1], "f1_val": best[0]}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
