import re

BOUNDARY_RE = re.compile(
    r"[,;:!?—]|\.(?=\s|$)|(?<=[ \u00a0])(?:что|чтобы|который|которая|которое|которые|где|когда|если|потому|поскольку|хотя)(?=[ \u00a0])",
    re.IGNORECASE,
)


def boundaries(text):
    edges = [0]
    for match in BOUNDARY_RE.finditer(text or ""):
        edges.append(match.start())
        edges.append(match.end())
    edges.append(len(text or ""))
    return sorted({edge for edge in edges if 0 <= edge <= len(text or "")})


def expand(span, edges, text_length):
    start = span["start"]
    end = span["end"]
    left = max([edge for edge in edges if edge <= start], default=0)
    right = min([edge for edge in edges if edge >= end + 1], default=text_length)
    return left, max(left, right - 1)


def shrink(span, edges, text_length):
    start = span["start"]
    end = span["end"]
    left = min([edge for edge in edges if edge > start], default=start)
    right = max([edge for edge in edges if edge < end + 1], default=end + 1)
    if right - left < 2:
        return start, end
    return left, right - 1


def nearest(span, edges, text_length):
    start = span["start"]
    end = span["end"]
    left_out = [edge for edge in edges if edge <= start]
    left_in = [edge for edge in edges if edge > start]
    right_out = [edge for edge in edges if edge >= end + 1]
    right_in = [edge for edge in edges if edge < end + 1]
    new_start = start
    if left_out and left_in:
        new_start = left_out[-1] if start - left_out[-1] <= left_in[0] - start else left_in[0]
    elif left_out:
        new_start = left_out[-1]
    new_end = end
    if right_out and right_in:
        new_end = right_out[0] - 1 if right_out[0] - (end + 1) <= end - right_in[-1] + 1 else right_in[-1] - 1
    elif right_out:
        new_end = right_out[0] - 1
    if new_end < new_start:
        return start, end
    return new_start, new_end


def snapped(spans, answer, mode):
    edges = boundaries(answer)
    length = len(answer or "")
    result = []
    for span in spans:
        if mode == "expand":
            start, end = expand(span, edges, length)
        elif mode == "shrink":
            start, end = shrink(span, edges, length)
        else:
            start, end = nearest(span, edges, length)
        item = dict(span)
        item["start"] = max(0, start)
        item["end"] = min(length - 1, end)
        result.append(item)
    return result
