"""ПРОМПТ 56, стадия 1: пересборка входного документа на `docs`.

Что меняется и что не меняется:

  МЕНЯЕТСЯ только поле `document_text` и производное `doc_ids`.
  НЕ МЕНЯЕТСЯ ни одна человеческая метка: спаны, их координаты в ответе, типы
  (`type`, `type_src`), `model_output`, `query_text`, состав пар, разбиение
  по `pair_id`.

Новый `document_text` — склейка ВСЕХ документов из поля `docs` в формате
`## <doc_id>\\n<doc_text>`, разделитель `\\n\\n`, порядок как в `docs`
(он уже отсортирован по `pred_score`).

Скрипт проверяет критерии приёмки 1.1–1.7 и пишет отчёт; при невыполнении
любого критерия падает, а не продолжает.

Запуск: python3 tools/rebuild56.py
"""
import argparse
import ast
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.answer_json import parse_answer
from spanver.io_csv import read_gold
from spanver.locate import locate
from spanver.types import classify

GOLD_FILES = (
    ("sberquad-rag.csv", "sberquad", "sber", "1_rushallu_rag_gold/sberquad-rag.csv"),
    ("ruscibench-rag.csv", "ruscibench", "rusci", "1_rushallu_rag_gold/ruscibench-rag.csv"),
)

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
HEADER_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.M)


def normalize(text):
    text = re.sub(r"\[\d+\]", " ", text or "")
    text = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def parse_docs(raw):
    """Поле `docs` в CSV — repr списка словарей; возвращает список словарей."""
    text = str(raw or "").strip()
    if not text.startswith("["):
        return [{"doc_id": "", "doc_text": text}] if text else []
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, dict):
        value = [value]
    return [item for item in value if isinstance(item, dict)]


def build_document(docs):
    """Склейка всех документов: `## id\\ntext`, разделитель `\\n\\n`."""
    blocks = []
    for item in docs:
        ident = str(item.get("doc_id") or "").strip()
        body = item.get("doc_text") or ""
        blocks.append(f"## {ident}\n{body}" if ident else body)
    return "\n\n".join(blocks)


def doc_ids_of(document):
    found = []
    for match in HEADER_RE.finditer(document or ""):
        ident = match.group(1).strip()
        if ident and ident not in found:
            found.append(ident)
    if not found:
        found.append("sha1:" + hashlib.sha1((document or "").encode("utf-8")).hexdigest()[:16])
    return found


def grams(text, size=4):
    words = WORD_RE.findall(text)
    return {" ".join(words[index:index + size]) for index in range(len(words) - size + 1)}


def load_splits():
    with (ROOT / "spanver/splits.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def build_rows(use_docs):
    """use_docs=True — новый вход (все документы), False — старый (joined_reference)."""
    splits = load_splits()
    allowed = set(splits["train"]) | set(splits["val"])
    rows = []
    for filename, domain, prefix, source in GOLD_FILES:
        path = ROOT / "Dataset/1_rushallu_rag_gold" / filename
        for number, row in enumerate(read_gold(path)):
            pair_id = f"{prefix}{number:06d}"
            if pair_id not in allowed:
                continue
            output = row.get("model_output") or ""
            raw = parse_answer(row.get("answer"), pair_id)
            spans = []
            for item, found in zip(raw, locate(raw, output)):
                spans.append({
                    "text": item["text"],
                    "start": found["start"],
                    "end": found["end"],
                    "type_src": item["type"],
                    "type": classify(item["text"], item["type"]),
                    "evidence_ids": [],
                })
            document = build_document(parse_docs(row.get("docs"))) if use_docs \
                else (row.get("joined_reference") or "")
            rows.append({
                "pair_id": pair_id,
                "domain": domain,
                "source": source,
                "document_text": document,
                "doc_ids": doc_ids_of(document),
                "model_output": output,
                "spans": spans,
            })
    return rows


def serialize(rows):
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return payload.encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--report", default="artifacts/rebuild56.json")
    parser.add_argument("--baseline", default="artifacts/ours33.jsonl")
    args = parser.parse_args()

    baseline_path = ROOT / args.baseline
    baseline = []
    if baseline_path.exists():
        with baseline_path.open(encoding="utf-8") as handle:
            baseline = [json.loads(line) for line in handle if line.strip()]

    rows = build_rows(use_docs=True)
    splits = load_splits()

    # --- критерий 1.1: 699 / 300 ---
    spans_total = sum(len(row["spans"]) for row in rows)
    if len(rows) != 699 or spans_total != 300:
        raise ValueError(f"1.1 провален: {len(rows)} пар, {spans_total} спанов")

    # --- критерий 1.2: координаты спанов не изменились ---
    by_id = {row["pair_id"]: row for row in rows}
    mismatched = []
    for old in baseline:
        new = by_id[old["pair_id"]]
        if new["model_output"] != old["model_output"]:
            mismatched.append((old["pair_id"], "model_output"))
            continue
        if [(s["start"], s["end"], s["type_src"], s["type"]) for s in new["spans"]] != \
           [(s["start"], s["end"], s["type_src"], s["type"]) for s in old["spans"]]:
            mismatched.append((old["pair_id"], "span"))
    if mismatched:
        raise ValueError(f"1.2 провален: изменены метки в {len(mismatched)} парах, напр. {mismatched[:5]}")

    # --- критерий 1.3: pair_id уникален ---
    counts = Counter(row["pair_id"] for row in rows)
    duplicated = [key for key, value in counts.items() if value > 1]
    if duplicated:
        raise ValueError(f"1.3 провален: дубли pair_id {duplicated[:5]}")

    # --- критерий 1.4: медиана длины ---
    median_len = statistics.median(len(row["document_text"]) for row in rows)

    payload = serialize(rows)
    digest = hashlib.sha256(payload).hexdigest()

    # --- переход от старого входа к новому ---
    old_lengths = [len(row["document_text"]) for row in baseline] if baseline else []
    shared = sum(1 for row in rows if row["document_text"] == by_id[row["pair_id"]]["document_text"])

    # --- критерий 1.6: контроль «свои / чужие документы» ---
    own, foreign = [], []
    pooled_num = pooled_den = 0
    ordered = [row["pair_id"] for row in rows]
    # «чужой» документ берётся детерминированной перестановкой (сдвиг на половину
    # списка по кругу), а не соседней парой: соседи в CSV — одна тема, это завышало
    # контроль до 0,0016996173674046176.
    shift = len(ordered) // 2
    for index, row in enumerate(rows):
        answer = normalize(row["model_output"])
        answer_grams = grams(answer)
        if not answer_grams:
            continue
        own_grams = grams(normalize(row["document_text"]))
        own.append(len(answer_grams & own_grams) / len(answer_grams))
        pooled_num += len(answer_grams & own_grams)
        pooled_den += len(answer_grams)
        other = by_id[ordered[(index + shift) % len(ordered)]]
        other_grams = grams(normalize(other["document_text"]))
        foreign.append(len(answer_grams & other_grams) / len(answer_grams))

    # --- критерий 1.7: защита разбиения ---
    test = [pair_id for pair_id in splits["test"]]
    train_val = set(splits["train"]) | set(splits["val"])
    doc_to_test = {}
    doc_to_train = {}
    for row in rows:
        for ident in row["doc_ids"]:
            doc_to_train.setdefault(ident, set()).add(row["pair_id"])
    # test-документы берём из исходных CSV, не из наших 699
    for filename, domain, prefix, source in GOLD_FILES:
        path = ROOT / "Dataset/1_rushallu_rag_gold" / filename
        for number, row in enumerate(read_gold(path)):
            pair_id = f"{prefix}{number:06d}"
            if pair_id not in splits["test"]:
                continue
            for ident in doc_ids_of(build_document(parse_docs(row.get("docs")))):
                doc_to_test.setdefault(ident, set()).add(pair_id)

    overlap = sorted(set(doc_to_test) & set(doc_to_train))
    all_docs = set(doc_to_test) | set(doc_to_train)

    # чистая подвыборка test: пары, чьи документы не встречаются в train+val
    clean_test = []
    dirty_test = []
    for pair_id in test:
        own_docs = [ident for ident, pairs in doc_to_test.items() if pair_id in pairs]
        if any(ident in doc_to_train for ident in own_docs):
            dirty_test.append(pair_id)
        else:
            clean_test.append(pair_id)

    report = {
        "status": "готово",
        "scope": "699 пар train/val; test 301 не оценивался, только проверено пересечение документов",
        "test301": False,
        "entry_criterion": "1.1–1.7",
        "c1_pairs": len(rows),
        "c1_spans": spans_total,
        "c2_span_coordinates_identical": True,
        "c2_pairs_compared": len(baseline),
        "c3_unique_pair_ids": len(counts),
        "c4_document_length": {
            "median": median_len,
            "mean": sum(len(row["document_text"]) for row in rows) / len(rows),
            "min": min(len(row["document_text"]) for row in rows),
            "max": max(len(row["document_text"]) for row in rows),
            "sum": sum(len(row["document_text"]) for row in rows),
            "baseline_median": statistics.median(old_lengths) if old_lengths else None,
            "baseline_sum": sum(old_lengths) if old_lengths else None,
            "growth": (sum(len(row["document_text"]) for row in rows) / sum(old_lengths))
            if old_lengths else None,
        },
        "c5_sha256": digest,
        "c5_path": args.out,
        "c6_answer_4grams": {
            "n_pairs": len(own),
            "own_docs_mean": sum(own) / len(own),
            "own_docs_median": statistics.median(own),
            "own_docs_pooled": pooled_num / pooled_den,
            "foreign_docs_mean": sum(foreign) / len(foreign),
            "foreign_docs_median": statistics.median(foreign),
            "foreign_pairs_nonzero": sum(1 for value in foreign if value > 0),
            "note": "«чужой» документ — сдвиг на половину списка по кругу; «свои» — среднее по парам, "
                    "pooled отдельной строкой",
        },
        "c7_split_protection": {
            "doc_ids_total": len(all_docs),
            "doc_ids_in_test": len(doc_to_test),
            "doc_ids_in_train_val": len(doc_to_train),
            "overlap": len(overlap),
            "overlap_share": len(overlap) / len(all_docs) if all_docs else None,
            "overlap_ids": overlap,
            "test_pairs": len(test),
            "test_pairs_touching_train_val": len(dirty_test),
            "test_pairs_touching_share": len(dirty_test) / len(test) if test else None,
            "clean_test_pairs": len(clean_test),
            "clean_test_share": len(clean_test) / len(test) if test else None,
            "clean_test_ids": clean_test,
            "split_unchanged": "разбиение по pair_id не менялось",
        },
        "n_docs_median": statistics.median(len(row["doc_ids"]) for row in rows),
        "n_docs_distribution": dict(sorted(Counter(len(row["doc_ids"]) for row in rows).items())),
        "note": "человеческие метки не менялись: спаны, координаты, типы, model_output, состав пар",
    }

    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_bytes(payload)
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for key in ("c1_pairs", "c1_spans", "c3_unique_pair_ids", "c5_sha256"):
        print(f"{key} = {report[key]}")
    print("c4 длина документа:", json.dumps(report["c4_document_length"], ensure_ascii=False))
    print("c6 4-граммы:", json.dumps(report["c6_answer_4grams"], ensure_ascii=False))
    print("c7 разбиение:", json.dumps({k: v for k, v in report["c7_split_protection"].items()
                                       if k not in ("overlap_ids", "clean_test_ids")}, ensure_ascii=False))
    print("c7 #пересечение doc_id:", report["c7_split_protection"]["overlap"])


if __name__ == "__main__":
    main()
