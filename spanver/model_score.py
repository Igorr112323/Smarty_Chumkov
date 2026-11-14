import json
import math
from pathlib import Path

from .evaluate import metrics
from .model_features import HEAD_FEATURES
from .module import MAX_SPANS, confidence, predict_type

HEAD_NAMES = HEAD_FEATURES

THRESHOLD_GRID = tuple(round(0.50 + 0.01 * step, 2) for step in range(50))
MAX_FPR = 0.40
TRAIN_LIMIT = 400
TRAIN_ITERATIONS = 300
TRAIN_RATE = 0.5
TRAIN_L2 = 0.001
TAIL = 6


def sigmoid(value):
    if value < -30:
        return 0.0
    if value > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def column_stats(vectors):
    width = len(vectors[0]) if vectors else 0
    mean = []
    scale = []
    for index in range(width):
        values = [row[index] for row in vectors]
        average = sum(values) / len(values)
        variance = sum((value - average) ** 2 for value in values) / len(values)
        mean.append(average)
        scale.append(math.sqrt(variance) or 1.0)
    return mean, scale


def normalize_row(row, mean, scale):
    return [1.0] + [(row[index] - mean[index]) / scale[index] if scale[index] else 0.0 for index in range(1, len(row))]


def train_head(vectors, labels):
    mean, scale = column_stats(vectors)
    rows = [normalize_row(row, mean, scale) for row in vectors]
    width = len(rows[0])
    weights = [0.0] * width
    count = len(rows)
    for _ in range(TRAIN_ITERATIONS):
        gradient = [0.0] * width
        for row, label in zip(rows, labels):
            total = sum(weight * value for weight, value in zip(weights, row))
            error = sigmoid(total) - label
            for index in range(width):
                gradient[index] += error * row[index]
        for index in range(width):
            penalty = TRAIN_L2 * weights[index] if index else 0.0
            weights[index] -= TRAIN_RATE * (gradient[index] / count + penalty)
    return {"weights": [round(value, 6) for value in weights], "mean": [round(value, 6) for value in mean], "scale": [round(value, 6) for value in scale]}


def head_risk(values, head):
    row = normalize_row(values, head["mean"], head["scale"])
    total = sum(weight * value for weight, value in zip(head["weights"], row))
    return round(sigmoid(total), 4)


def score_rows(rows, head):
    scored = []
    for row in rows:
        scored.append(
            {
                "start": row["start"],
                "end": row["end"],
                "support": row.get("support", 0.0),
                "risk": head_risk(row["values"], head),
                "values": row["values"],
            }
        )
    return scored


def select_spans(scored, threshold, max_spans=MAX_SPANS):
    chosen = [item for item in scored if item["risk"] >= threshold]
    chosen.sort(key=lambda item: (-item["risk"], item["start"]))
    return chosen[:max_spans]


def spans_payload(chosen, answer):
    payload = []
    for item in chosen:
        fragment = answer[item["start"] : item["end"] + 1]
        payload.append(
            {
                "start": item["start"],
                "end": item["end"],
                "risk": item["risk"],
                "type": predict_type(fragment, item["support"]),
                "confidence": confidence(item["risk"]),
            }
        )
    return payload


def make_predictor(table, answers, threshold, max_spans=MAX_SPANS):
    def predictor(pair):
        pair_id = pair.get("pair_id")
        scored = table.get(pair_id) or []
        chosen = select_spans(scored, threshold, max_spans)
        return spans_payload(chosen, answers.get(pair_id, ""))

    return predictor


def evaluate_with(table, pairs, threshold, max_spans=MAX_SPANS):
    answers = {pair.get("pair_id"): pair.get("model_output") or "" for pair in pairs}
    return metrics(pairs, make_predictor(table, answers, threshold, max_spans))


def scan_threshold(table, pairs, grid=THRESHOLD_GRID, max_fpr=MAX_FPR):
    best = None
    history = []
    for threshold in grid:
        result = evaluate_with(table, pairs, threshold)
        f1 = result["span_level"]["f1"]
        fpr = result["fpr"]
        history.append({"threshold": threshold, "f1": f1, "fpr": fpr})
        if fpr > max_fpr:
            continue
        if best is None or (f1, threshold) > (best["f1"], best["threshold"]):
            best = {"threshold": threshold, "f1": f1, "fpr": fpr}
    if best is None:
        best = {"threshold": float(grid[-1]), "f1": 0.0, "fpr": 1.0}
    return best, history


def label_rows(rows, gold):
    labelled = []
    for row in rows:
        label = 0
        for span in gold:
            left = max(row["start"], span["start"])
            right = min(row["end"], span["end"])
            if right < left:
                continue
            union = max(row["end"], span["end"]) - min(row["start"], span["start"]) + 1
            if union and (right - left + 1) / union >= 0.5:
                label = 1
                break
        labelled.append(label)
    return labelled


def training_subset(pairs, limit=TRAIN_LIMIT):
    positives = []
    negatives = []
    for pair in sorted(pairs, key=lambda item: item.get("pair_id") or ""):
        if pair.get("spans"):
            positives.append(pair)
        else:
            negatives.append(pair)
    chosen = positives[:limit]
    room = limit - len(chosen)
    if room > 0:
        chosen = chosen + negatives[:room]
    return chosen


def load_table(path):
    target = Path(path)
    if not target.exists():
        return {}
    table = {}
    with open(target, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            table[row["pair_id"]] = row["rows"]
    return table


def save_table(table, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        for pair_id in sorted(table):
            handle.write(json.dumps({"pair_id": pair_id, "rows": table[pair_id]}, ensure_ascii=False) + "\n")
    return target


def ai_share(scored, answer):
    from .module import sentence_spans

    text = answer or ""
    by_span = {(item["start"], item["end"]): item["risk"] for item in scored}
    total = 0.0
    covered = 0.0
    for start, end in sentence_spans(text):
        length = end - start + 1
        total += length
        covered += length * by_span.get((start, end), 0.0)
    return round(covered / total, 4) if total else 0.0


HEAD_FILE = "artifacts/model_head.json"


def save_head(path, head, threshold, sha, extra=None):
    payload = {
        "features": list(HEAD_NAMES),
        "weights": head["weights"],
        "mean": head["mean"],
        "scale": head["scale"],
        "threshold": threshold,
        "weights_sha256": sha,
    }
    if extra:
        payload.update(extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_head(path=HEAD_FILE):
    target = Path(path)
    if not target.exists():
        raise HeadMissing(f"веса головки не найдены: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    return {
        "head": {"weights": data["weights"], "mean": data["mean"], "scale": data["scale"]},
        "threshold": data.get("threshold"),
        "features": data.get("features"),
        "weights_sha256": data.get("weights_sha256"),
    }


class HeadMissing(RuntimeError):
    pass


def verify_model(answer, document, weights, head_path=HEAD_FILE, threshold=None, max_spans=MAX_SPANS):
    from .model_features import MAX_MODEL_CANDIDATES, load_model, pair_features, weights_files

    weights_files(weights)
    pack = load_head(head_path)
    bundle = load_model(weights)
    text = answer or ""
    rows = pair_features(bundle, text, document or "", MAX_MODEL_CANDIDATES)
    scored = score_rows(rows, pack["head"])
    limit = pack["threshold"] if threshold is None else threshold
    if limit is None:
        limit = 0.5
    chosen = select_spans(scored, limit, max_spans)
    return {
        "spans": spans_payload(chosen, text),
        "ai_share": ai_share(scored, text),
        "meta": {
            "answer_len": len(text),
            "doc_len": len(document or ""),
            "n_spans": len(chosen),
            "backend": "model",
            "candidates": len(rows),
            "weights_key": bundle["sha"][:16],
        },
    }
