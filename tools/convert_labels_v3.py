"""Детерминированная гармонизация span-меток v3 (правило C1).

Правило C1 (общее для RusHallu и mFAVA):
  каждый исходный gold-span заменяется на минимальный фрагмент из
  spanver.module.candidate_spans, который его содержит; границы trimming
  whitespace; дубликаты (start,end) удаляются; тип сохраняется.
Обоснование: и RusHallu, и mFAVA содержат clause-уровневые annotation; это
наименьший общий знаменатель. whole-answer остаётся whole, если нет меньшего
содержащего фрагмента. Held-out не используется при выборе правила.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.module import candidate_spans


def convert_pair(pair):
    answer = pair.get("model_output") or ""
    fragments = candidate_spans(answer)
    converted = []
    seen = set()
    for gold in pair.get("spans") or []:
        start, end = gold["start"], gold["end"]
        containing = [f for f in fragments if f[0] <= start and f[1] >= end]
        if containing:
            chosen = min(containing, key=lambda f: (f[1] - f[0], f[0]))
            start, end = chosen[0], chosen[1]
        while start <= end and start < len(answer) and answer[start].isspace():
            start += 1
        while end >= start and answer[end].isspace():
            end -= 1
        if start > end:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        item = dict(gold)
        item["start"] = start
        item["end"] = end
        item["original_start"] = gold["start"]
        item["original_end"] = gold["end"]
        item["conversion"] = "C1_min_containing_clause"
        converted.append(item)
    result = dict(pair)
    result["spans"] = converted
    return result


def convert_all(pairs):
    return [convert_pair(p) for p in pairs]
