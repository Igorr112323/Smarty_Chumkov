import unicodedata

from .locate import STEPS, _apply


def is_control(char):
    return unicodedata.category(char) == "Cc" and char not in "\n\r\t"


def has_control(text):
    return any(is_control(char) for char in text or "")


def match_steps(text, model_output, start, end):
    window = model_output[start : end + 1]
    if window == text:
        return 0
    for count in range(1, len(STEPS) + 1):
        left, _ = _apply(window, count)
        right, _ = _apply(text, count)
        if left == right:
            return count
    return None


def inspect(pairs):
    spans_total = 0
    located = 0
    out_of_bounds = 0
    mismatch = 0
    overlaps = 0
    nested = 0
    control_pairs = 0
    for pair in pairs:
        document = pair.get("document_text") or ""
        if has_control(document):
            control_pairs += 1
        output = pair.get("model_output") or ""
        intervals = []
        for span in pair.get("spans") or []:
            spans_total += 1
            start = span.get("start")
            end = span.get("end")
            if start is None or end is None:
                continue
            if not (0 <= start <= end < len(output)):
                out_of_bounds += 1
                continue
            if match_steps(span.get("text") or "", output, start, end) is None:
                mismatch += 1
                continue
            located += 1
            intervals.append((start, end))
        for left_index in range(len(intervals)):
            for right_index in range(left_index + 1, len(intervals)):
                first = intervals[left_index]
                second = intervals[right_index]
                inside_first = first[0] <= second[0] and second[1] <= first[1]
                inside_second = second[0] <= first[0] and first[1] <= second[1]
                if inside_first or inside_second:
                    nested += 1
                elif not (first[1] < second[0] or first[0] > second[1]):
                    overlaps += 1
    coverage = located / spans_total if spans_total else 1.0
    ok = out_of_bounds == 0 and mismatch == 0 and coverage >= 1.0
    return {
        "spans_total": spans_total,
        "spans_located": located,
        "coverage": coverage,
        "out_of_bounds": out_of_bounds,
        "span_text_mismatch": mismatch,
        "overlaps": overlaps,
        "nested": nested,
        "control_pairs": control_pairs,
        "ok": ok,
    }
