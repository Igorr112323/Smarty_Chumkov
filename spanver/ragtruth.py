import hashlib
import json


def spans_from_labels(labels):
    prepared = []
    for label in labels or []:
        start = int(label.get("start", 0))
        end = int(label.get("end", 0)) - 1
        if end < start:
            continue
        prepared.append({"start": start, "end": end, "label_type": label.get("label_type") or ""})
    prepared.sort(key=lambda item: (item["start"], item["end"]))
    merged = []
    for span in prepared:
        if merged and span["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
            types = merged[-1]["label_type"].split("|")
            if span["label_type"] and span["label_type"] not in types:
                merged[-1]["label_type"] = "|".join(filter(None, types + [span["label_type"]]))
            continue
        merged.append(dict(span))
    return merged


def chunks_from(text, spans):
    text = text or ""
    length = len(text)
    chunks = []
    cursor = 0
    for index, span in enumerate(spans):
        start = max(0, min(span["start"], length))
        end = max(start, min(span["end"], length - 1)) if length else 0
        if start > cursor:
            chunks.append({"kind": "text", "text": text[cursor:start], "span": None})
        chunks.append({"kind": "span", "text": text[start : end + 1], "span": dict(span, start=start, end=end)})
        cursor = end + 1
    if cursor < length:
        chunks.append({"kind": "text", "text": text[cursor:], "span": None})
    return [chunk for chunk in chunks if chunk["text"] or chunk["kind"] == "span"]


def rebuild(chunks, translated):
    pieces = []
    spans = []
    cursor = 0
    for chunk, value in zip(chunks, translated):
        piece = value or ""
        if chunk["kind"] == "span":
            start = cursor
            end = cursor + len(piece) - 1
            if end >= start:
                spans.append({"start": start, "end": end, "span_text": piece, "label_type": (chunk["span"] or {}).get("label_type", "")})
        pieces.append(piece)
        cursor += len(piece)
    text = "".join(pieces)
    return text, spans


def context_text(source):
    value = (source or {}).get("source_info")
    if value is None:
        value = (source or {}).get("prompt")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value or ""


def doc_id(context):
    return hashlib.sha256((context or "").encode("utf-8")).hexdigest()[:16]


def src_sha1(text):
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def select_pairs(rows, limit=6000, negatives=1500, seed=0):
    import random

    primary = ("Evident Conflict", "Evident Baseless Info", "Subtle Baseless Info")
    labelled = []
    for row in rows:
        spans = spans_from_labels(row.get("labels") or [])
        if spans:
            labelled.append((row, spans))
    rng = random.Random(seed)
    rng.shuffle(labelled)
    buckets = {name: [] for name in primary}
    rest = []
    for row, spans in labelled:
        types = {span["label_type"] for span in spans}
        placed = False
        if len(spans) <= 2:
            for name in primary:
                if name in types:
                    buckets[name].append((row, spans))
                    placed = True
                    break
        if not placed:
            rest.append((row, spans))
    per_type = limit // len(primary)
    chosen = []
    for name in primary:
        chosen.extend(buckets[name][:per_type])
    rest.sort(key=lambda item: len(item[1]))
    chosen.extend(rest[: max(0, limit - len(chosen))])
    seen = set()
    unique = []
    for row, spans in chosen:
        if row.get("id") in seen:
            continue
        seen.add(row.get("id"))
        unique.append((row, spans))
    without = [row for row in rows if not (row.get("labels") or [])]
    rng.shuffle(without)
    return unique[:limit], without[:negatives]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
