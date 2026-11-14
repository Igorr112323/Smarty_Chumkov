import json
import re
from pathlib import Path

from .module import IOU_THRESHOLD, verify

TOKEN_RE = re.compile(r"\S+")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iou(first_start, first_end, second_start, second_end):
    intersection = min(first_end, second_end) - max(first_start, second_start) + 1
    if intersection <= 0:
        return 0.0
    union = max(first_end, second_end) - min(first_start, second_start) + 1
    return intersection / union if union else 0.0


def match_spans(predicted, gold, iou_threshold=IOU_THRESHOLD):
    used = set()
    matched = 0
    type_correct = 0
    for span in predicted:
        for index, truth in enumerate(gold):
            if index in used:
                continue
            if iou(span["start"], span["end"], truth["start"], truth["end"]) >= iou_threshold:
                used.add(index)
                matched += 1
                if truth.get("type") == span.get("type"):
                    type_correct += 1
                break
    return matched, used, type_correct


def token_positions(text):
    found = []
    for match in TOKEN_RE.finditer(text):
        found.append((match.start(), match.end() - 1))
    return found


def covered_tokens(text, intervals):
    positions = token_positions(text)
    covered = set()
    for index, (start, end) in enumerate(positions):
        for left, right in intervals:
            if left <= start <= right or left <= end <= right:
                covered.add(index)
                break
    return covered


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_pairs(pairs, threshold=None, iou_threshold=IOU_THRESHOLD, max_spans=None, nms_iou=None, weights=None):
    tp = fp = fn = 0
    pair_tp = pair_fp = pair_fn = 0
    negatives = 0
    negatives_flagged = 0
    token_tp = token_fp = token_fn = 0
    type_correct = 0
    with_spans = 0
    for pair in pairs:
        answer = pair.get("model_output") or ""
        gold = pair.get("spans") or []
        if gold:
            with_spans += 1
        predicted = verify(answer, pair.get("document_text") or "", threshold=threshold, weights=weights, max_spans=max_spans, nms_iou=nms_iou)["spans"]
        matched, used, correct = match_spans(predicted, gold, iou_threshold)
        type_correct += correct
        tp += matched
        fp += len(predicted) - matched
        fn += len(gold) - matched
        if gold:
            if matched:
                pair_tp += 1
            else:
                pair_fn += 1
        else:
            negatives += 1
            if predicted:
                pair_fp += 1
                negatives_flagged += 1
        gold_intervals = [(span["start"], span["end"]) for span in gold]
        predicted_intervals = [(span["start"], span["end"]) for span in predicted]
        gold_tokens = covered_tokens(answer, gold_intervals)
        predicted_tokens = covered_tokens(answer, predicted_intervals)
        token_tp += len(gold_tokens & predicted_tokens)
        token_fp += len(predicted_tokens - gold_tokens)
        token_fn += len(gold_tokens - predicted_tokens)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    token_precision = ratio(token_tp, token_tp + token_fp)
    token_recall = ratio(token_tp, token_tp + token_fn)
    pair_precision = ratio(pair_tp, pair_tp + pair_fp)
    pair_recall = ratio(pair_tp, pair_tp + pair_fn)
    return {
        "n_pairs": len(pairs),
        "n_pairs_with_spans": with_spans,
        "n_neg_pairs": negatives,
        "span_level": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1(precision, recall), 4),
        },
        "token_level": {
            "precision": round(token_precision, 4),
            "recall": round(token_recall, 4),
            "f1": round(f1(token_precision, token_recall), 4),
        },
        "pair_level": {
            "precision": round(pair_precision, 4),
            "recall": round(pair_recall, 4),
            "f1": round(f1(pair_precision, pair_recall), 4),
        },
        "fpr": round(ratio(negatives_flagged, negatives), 4),
        "type_accuracy": round(ratio(type_correct, tp), 4),
        "type_correct": type_correct,
        "iou_threshold": iou_threshold,
    }


def load_split(path, split):
    return [pair for pair in read_jsonl(Path(path)) if pair.get("split") == split]


def run(path="artifacts/gold.jsonl", split="test", threshold=None):
    pairs = load_split(path, split)
    return evaluate_pairs(pairs, threshold=threshold)
