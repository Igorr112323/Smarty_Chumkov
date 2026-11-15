from .evaluate import metrics
from .module import confidence, predict_type

NEIGHBOR = 0.8
MIN_CHARS = 2
GRID = tuple(round(0.30 + 0.05 * step, 2) for step in range(13))


def flags_for(pieces, threshold, neighbor=NEIGHBOR):
    scores = [score for _, _, score in pieces]
    flags = []
    for index, score in enumerate(scores):
        if score < threshold:
            flags.append(False)
            continue
        limit = threshold * neighbor
        left = index > 0 and scores[index - 1] >= limit
        right = index + 1 < len(scores) and scores[index + 1] >= limit
        flags.append(bool(left or right))
    return flags


def spans_from_flags(answer, pieces, flags, min_chars=MIN_CHARS):
    spans = []
    current = None
    for (start, end, score), flag in zip(pieces, flags):
        if flag:
            if current is None:
                current = {"start": start, "end": end - 1, "risk": score}
            else:
                current["end"] = end - 1
                current["risk"] = max(current["risk"], score)
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    result = []
    for span in spans:
        text = answer[span["start"] : span["end"] + 1]
        if len(text) < min_chars:
            continue
        result.append(
            {
                "start": span["start"],
                "end": span["end"],
                "risk": round(span["risk"], 4),
                "type": predict_type(text, 0.0),
                "confidence": confidence(span["risk"]),
                "text": text,
            }
        )
    return result


def predictor_from_cache(cache, threshold, neighbor=NEIGHBOR, min_chars=MIN_CHARS):
    def predict(pair):
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            return []
        flags = flags_for(cached["pieces"], threshold, neighbor)
        return spans_from_flags(cached["answer"], cached["pieces"], flags, min_chars)

    return predict


def token_counts(cache, pairs, threshold, neighbor=NEIGHBOR, min_chars=MIN_CHARS):
    tp = fp = fn = 0
    negatives = 0
    flagged = 0
    for pair in pairs:
        cached = cache.get(pair.get("pair_id"))
        if not cached:
            continue
        pieces = cached["pieces"]
        flags = flags_for(pieces, threshold, neighbor)
        kept = [(span["start"], span["end"]) for span in spans_from_flags(cached["answer"], pieces, flags, min_chars)]
        gold = pair.get("spans") or []
        hit = False
        for (start, end, score), flag in zip(pieces, flags):
            inside = False
            for span in gold:
                if min(end, span["end"] + 1) - max(start, span["start"]) > 0:
                    inside = True
                    break
            predicted = flag and any(left <= start and end - 1 <= right for left, right in kept)
            if predicted:
                hit = True
            if inside and predicted:
                tp += 1
            elif inside:
                fn += 1
            elif predicted:
                fp += 1
        if not gold:
            negatives += 1
            if hit:
                flagged += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(flagged / negatives, 4) if negatives else 0.0,
    }


def sweep(cache, pairs, grid=GRID, neighbor=NEIGHBOR, min_chars=MIN_CHARS):
    rows = []
    for threshold in grid:
        tokens = token_counts(cache, pairs, threshold, neighbor, min_chars)
        spans = metrics(pairs, predictor_from_cache(cache, threshold, neighbor, min_chars))
        row = dict(tokens)
        row["threshold"] = threshold
        row["span_precision"] = spans["span_level"]["precision"]
        row["span_recall"] = spans["span_level"]["recall"]
        row["span_f1"] = spans["span_level"]["f1"]
        row["pair_f1"] = spans["pair_level"]["f1"]
        row["pair_fpr"] = spans["fpr"]
        row["type_accuracy"] = spans["type_accuracy"]
        rows.append(row)
    return rows
