"""ПРОМПТ 54, задача 1: эмпирическое правило нашей разметки.

Классифицирует все 300 золотых спанов train/val по трём признакам:

  1. дословно присутствует в документе или нет (нормализация: нижний регистр,
     схлопывание пробелов, снятие вики-разметки документа);
  2. совпадает ли спан с полным предложением ответа или это его часть;
  3. исходный человеческий тип (type_src: Partial / Unconfirmed / Missing /
     Oversight / Excess / Contradiction) и наш семиклассовый тип.

Признак «отвечает ли на вопрос» автоматически не выставляется: вопроса нет в
парах (он остался в CSV), а ручная оценка 300 спанов — работа куратора. Вместо
прямого ответа считается доля лексического пересечения спана с документом,
которая для «ответ на вопрос» даёт верхнюю границу.

Ничего не обучает, ничего не меняет.
"""
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
SENTENCE_END = ".!?;:…"


def normalize(text):
    text = re.sub(r"\[\d+\]", " ", text or "")
    text = text.lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def words(text):
    return set(WORD_RE.findall(normalize(text)))


def doc_text_clean(document):
    """Документ в парах склеен из фрагментов вида '## id\\n...', убираем маркеры."""
    return re.sub(r"##\s*\d+", " ", document or "")


def split_sentences(text):
    parts, current = [], []
    for char in text:
        current.append(char)
        if char in SENTENCE_END:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return parts


def main():
    rows = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    records = []
    for row in rows:
        answer = row["model_output"]
        document = doc_text_clean(row["document_text"])
        corpus = " " + normalize(document) + " "
        sentences = split_sentences(answer)
        offsets, cursor = [], 0
        for sentence in sentences:
            start = answer.find(sentence, cursor)
            if start < 0:
                start = cursor
            offsets.append((start, start + len(sentence)))
            cursor = start + len(sentence)
        for span in row["spans"]:
            text = answer[span["start"]: span["end"] + 1]
            normalized = normalize(text)
            exact = bool(normalized) and (" " + normalized + " ") in corpus
            # запасной вариант: каждое слово спана встречается в документе подряд
            order_ok = False
            if not exact and normalized:
                flat = " ".join(WORD_RE.findall(normalized))
                doc_words = " " + " ".join(WORD_RE.findall(corpus)) + " "
                order_ok = bool(flat) and (doc_words.find(" " + flat + " ") >= 0)
            answer_words = words(text)
            document_words = words(document)
            overlap = len(answer_words & document_words) / len(answer_words) if answer_words else 0.0
            whole_sentence = False
            sentence_length = None
            for start, end in offsets:
                if start <= span["start"] and span["end"] + 1 <= end:
                    whole_sentence = (abs(span["start"] - start) <= 2 and abs(span["end"] + 1 - end) <= 2)
                    sentence_length = end - start
                    break
            records.append({
                "pair_id": row["pair_id"],
                "domain": row["domain"],
                "type": span["type"],
                "type_src": span.get("type_src"),
                "length": span["end"] - span["start"] + 1,
                "in_document_exact": exact,
                "in_document_contiguous": bool(exact or order_ok),
                "word_overlap_with_document": overlap,
                "whole_sentence": whole_sentence,
                "sentence_length": sentence_length,
            })

    total = len(records)
    exact = sum(1 for r in records if r["in_document_exact"])
    contiguous = sum(1 for r in records if r["in_document_contiguous"])
    whole = sum(1 for r in records if r["whole_sentence"])
    overlaps = [r["word_overlap_with_document"] for r in records]

    by_type = defaultdict(lambda: {"count": 0, "exact": 0, "contiguous": 0, "whole": 0, "lengths": []})
    for r in records:
        entry = by_type[r["type"]]
        entry["count"] += 1
        entry["exact"] += r["in_document_exact"]
        entry["contiguous"] += r["in_document_contiguous"]
        entry["whole"] += r["whole_sentence"]
        entry["lengths"].append(r["length"])

    by_src = defaultdict(lambda: {"count": 0, "exact": 0, "contiguous": 0, "whole": 0})
    for r in records:
        entry = by_src[r["type_src"]]
        entry["count"] += 1
        entry["exact"] += r["in_document_exact"]
        entry["contiguous"] += r["in_document_contiguous"]
        entry["whole"] += r["whole_sentence"]

    def shape(entry):
        count = entry["count"] or 1
        return {
            "count": entry["count"],
            "exact_share": entry["exact"] / count,
            "contiguous_share": entry["contiguous"] / count,
            "whole_sentence_share": entry["whole"] / count,
        }

    report = {
        "status": "готово",
        "scope": "все 300 золотых спанов train/val из artifacts/ours33.jsonl",
        "total_spans": total,
        "in_document_exact": exact,
        "in_document_exact_share": exact / total,
        "in_document_contiguous": contiguous,
        "in_document_contiguous_share": contiguous / total,
        "not_in_document_exact": total - exact,
        "not_in_document_exact_share": (total - exact) / total,
        "whole_sentence": whole,
        "whole_sentence_share": whole / total,
        "part_of_sentence": total - whole,
        "part_of_sentence_share": (total - whole) / total,
        "length_median": statistics.median([r["length"] for r in records]),
        "length_mean": sum(r["length"] for r in records) / total,
        "word_overlap_median": statistics.median(overlaps),
        "word_overlap_mean": sum(overlaps) / total,
        "questione_answering_note": (
            "признак «отвечает на вопрос» машинно не выставлялся: вопроса нет в парах, "
            "а ручная оценка 300 спанов — работа куратора. Косвенная оценка — доля "
            "пересечения слов с документом"
        ),
        "by_type": {key: shape(value) for key, value in sorted(by_type.items())},
        "by_source_type": {key: shape(value) for key, value in sorted(by_src.items())},
        "length_by_type": {key: statistics.median(value["lengths"]) for key, value in sorted(by_type.items())},
        "spans": records,
    }
    (ROOT / "artifacts/marking54_rule.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "total_spans", "in_document_exact_share", "in_document_contiguous_share",
        "not_in_document_exact_share", "whole_sentence_share", "part_of_sentence_share",
        "length_median", "word_overlap_median")}, ensure_ascii=False, indent=1))
    print("по типам:")
    for key, value in report["by_type"].items():
        print(f"  {key:22s} n={value['count']:3d} exact {value['exact_share']:.4f} "
              f"contiguous {value['contiguous_share']:.4f} whole {value['whole_sentence_share']:.4f}")
    print("по человеческому типу (type_src):")
    for key, value in report["by_source_type"].items():
        print(f"  {str(key):15s} n={value['count']:3d} exact {value['exact_share']:.4f} "
              f"contiguous {value['contiguous_share']:.4f} whole {value['whole_sentence_share']:.4f}")


if __name__ == "__main__":
    main()
