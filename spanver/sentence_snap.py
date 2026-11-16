import re

from .module import confidence, predict_type

SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\n+")


def sentence_intervals(text):
    result = []
    start = 0
    for match in SENTENCE_BREAK.finditer(text):
        end = match.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            result.append((start, end - 1))
        start = match.end()
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        result.append((start, end - 1))
    return result


def snap_sentences(spans, text):
    sentences = sentence_intervals(text)
    result = []
    for span in spans:
        left = span["start"]
        right = span["end"]
        if not 0 <= left <= right < len(text):
            raise ValueError("спан вне границ ответа")
        for start, end in sentences:
            overlap = max(0, min(right, end) - max(left, start) + 1)
            if overlap * 10 >= 3 * (end - start + 1):
                left = min(left, start)
                right = max(right, end)
        item = dict(span)
        item["start"] = left
        item["end"] = right
        result.append(item)
    result.sort(key=lambda span: (span["start"], span["end"]))
    merged = []
    for item in result:
        if merged and item["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            if "risk" in item and "risk" in merged[-1]:
                merged[-1]["risk"] = max(item["risk"], merged[-1]["risk"])
        else:
            merged.append(item)
    for item in merged:
        item["text"] = text[item["start"]:item["end"] + 1]
        item["type"] = predict_type(item["text"], 0.0)
        if "risk" in item and "confidence" in item:
            item["confidence"] = confidence(item["risk"])
    return merged
