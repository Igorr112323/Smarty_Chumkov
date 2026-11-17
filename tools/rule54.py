"""ПРОМПТ 54: фиксация правила разметки и сводка задачи 3.

Три вещи, каждая проверяема по сохранённым данным, без обучения:

  1. эмпирическое правило наших 300 спанов (что помечается и как выглядит);
  2. сравнение с правилом других датасетов на их собственных файлах:
     сколько спанов в RAGTruth-ru / RuFacts / mFAVA дословно есть в документе;
  3. чистота прогонов: какие данные реально попадали в обучение схем 44–52.

Скрипт не строит выводов, которых нет в данных: он печатает доли и оставляет
вердикт отчёту.
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.marking54 import doc_text_clean, normalize, SENTENCE_END


def sentences_of(text):
    parts, current = [], []
    for char in text:
        current.append(char)
        if char in SENTENCE_END:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return parts


def whole_sentence_share(answer, start, end):
    cursor = 0
    for sentence in sentences_of(answer):
        left = answer.find(sentence, cursor)
        if left < 0:
            left = cursor
        right = left + len(sentence)
        cursor = right
        if left <= start and end + 1 <= right:
            return abs(start - left) <= 2 and abs(end + 1 - right) <= 2
    return False


def literal_share(pairs):
    """Доля спанов, дословно встречающихся в документе (нормализация без фаззи)."""
    inside = total = 0
    for pair in pairs:
        corpus = " " + normalize(doc_text_clean(pair.get("document_text"))) + " "
        answer = pair.get("model_output") or ""
        for span in pair.get("spans") or []:
            text = normalize(answer[span["start"]: span["end"] + 1])
            if not text:
                continue
            total += 1
            if (" " + text + " ") in corpus:
                inside += 1
    return inside, total, (inside / total if total else None)


def load_other(name):
    path = ROOT / "Dataset" / name
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cleanliness():
    """Какие ДАННЫЕ читают обучающие скрипты схем 44–52.

    Различаем данные и импорт метрик: `from tools.sweep_ragtruth30 import ...`
    это метрика, а не датасет, и в счёт источников попадать не должна.
    """
    scripts = sorted((ROOT / "tools").glob("train*.py"))
    hits = {}
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        data, metrics_only = [], []
        for token, label in (
            ("ours33", "наши 699 пар"),
            ("RAGTruth_ru", "RAGTruth-ru"),
            ("hard_negatives", "hard negatives"),
            ("RuFacts", "RuFacts"),
            ("mFAVA", "mFAVA"),
            ("ragtruth", "RAGTruth"),
        ):
            if token not in text:
                continue
            lines = [line for line in text.splitlines() if token in line]
            reads = [line for line in lines
                     if ("read_jsonl" in line or "json.load" in line or "open(" in line
                         or "Dataset/" in line or "Path(" in line)
                     and "import" not in line]
            if reads:
                data.append(label)
            else:
                metrics_only.append(label)
        hits[script.name] = {"data": sorted(set(data)), "metrics_only": sorted(set(metrics_only))}
    return hits


def main():
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    spans = [(pair, span) for pair in pairs for span in pair["spans"]]
    total = len(spans)
    human = Counter(span.get("type_src") for _, span in spans)
    ours = Counter(span["type"] for _, span in spans)
    lengths = [span["end"] - span["start"] + 1 for _, span in spans]
    whole = sum(1 for pair, span in spans if whole_sentence_share(pair["model_output"], span["start"], span["end"]))
    inside, seen, literal = literal_share(pairs)

    others = {}
    for name in ("RAGTruth_ru/ragtruth_ru.jsonl", "mFAVA_ru/test.json"):
        if name.endswith(".jsonl"):
            rows = load_other(name)
        else:
            path = ROOT / "Dataset" / name
            rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            rows = [{"document_text": row.get("references") or "", "model_output": row.get("generated_text") or "",
                     "spans": [s for s in (row.get("gold_annotations") or []) if isinstance(s, dict)]} for row in rows]
        if not rows:
            continue
        inside_other, seen_other, share_other = literal_share(rows)
        in_answer = 0
        for row in rows:
            answer = normalize(row.get("model_output") or "")
            for span in row.get("spans") or []:
                text = normalize((row.get("model_output") or "")[span["start"]: span["end"] + 1])
                if text and text in answer:
                    in_answer += 1
        others[name] = {
            "pairs": len(rows),
            "spans_seen": seen_other,
            "literal_in_document": inside_other,
            "literal_share": share_other,
            "literal_in_answer_share": (in_answer / seen_other) if seen_other else None,
        }

    report = {
        "status": "готово",
        "ours": {
            "pairs": len(pairs),
            "spans": total,
            "pairs_with_spans": sum(1 for pair in pairs if pair["spans"]),
            "length_median": statistics.median(lengths),
            "length_mean": sum(lengths) / total,
            "whole_sentence": whole,
            "whole_sentence_share": whole / total,
            "part_of_sentence_share": (total - whole) / total,
            "literal_in_document": inside,
            "literal_share": literal,
            "human_types": dict(human),
            "human_type_shares": {key: value / total for key, value in human.items()},
            "our_types": dict(ours),
            "our_type_shares": {key: value / total for key, value in ours.items()},
            "non_addition_share": (total - human.get("Excess", 0)) / total,
            "median_length_by_human_type": {
                key: statistics.median([s["end"] - s["start"] + 1 for _, s in spans if s.get("type_src") == key])
                for key in human
            },
            "question_share": None,
            "question_note": "вопроса нет в парах: он остался в колонке query_text исходного CSV, "
                             "поэтому признак «отвечает на вопрос» машинно не считается",
        },
        "others": others,
        "training_sources": cleanliness(),
        "test301": False,
    }
    (ROOT / "artifacts/rule54.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report["ours"][key] for key in (
        "spans", "length_median", "whole_sentence_share", "literal_share", "non_addition_share")},
        ensure_ascii=False, indent=1))
    print("человеческие типы:", json.dumps(report["ours"]["human_type_shares"], ensure_ascii=False))
    for name, value in others.items():
        print(f"другой датасет {name}: спанов {value['spans_seen']}, дословно в документе "
              f"{value['literal_share']}")
    print("источники обучения:")
    for script, sources in report["training_sources"].items():
        print(f"  {script:22s} данные={sources['data']} метрики={sources['metrics_only']}")


if __name__ == "__main__":
    main()
