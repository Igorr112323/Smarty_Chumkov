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


def snap_sentences(spans, text, mode="B1"):
    if mode not in {"B1", "B2", "B3", "B4"}:
        raise ValueError("неизвестный режим снапа")
    sentences = sentence_intervals(text)
    result = []
    for span in spans:
        original_left = left = span["start"]
        original_right = right = span["end"]
        if not 0 <= left <= right < len(text):
            raise ValueError("спан вне границ ответа")
        numerator = 15 if mode == "B2" else 30
        selected = []
        for index, (start, end) in enumerate(sentences):
            overlap = max(0, min(original_right, end) - max(original_left, start) + 1)
            if overlap * 100 >= numerator * (end - start + 1):
                selected.append(index)
        included = set(selected)
        if mode == "B3":
            for index in selected:
                first = last = index
                while first > 0 and sentences[first][0] - sentences[first - 1][1] - 1 < 20:
                    first -= 1
                while last + 1 < len(sentences) and sentences[last + 1][0] - sentences[last][1] - 1 < 20:
                    last += 1
                included.update(range(first, last + 1))
        elif mode == "B4":
            for index in selected:
                if index > 0:
                    included.add(index - 1)
                if index + 1 < len(sentences):
                    included.add(index + 1)
        for index in included:
            start, end = sentences[index]
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
