"""Отчёт 55: перепроверка методики отчёта 54.

Четыре проверки, каждая обязана воспроизвести или опровергнуть заявленную цифру:

  1. autojunk в difflib: медиана покрытия спана документом при autojunk=True и False;
  2. полный охват 300 спанов (префиксы sber + rusci), а не подвыборка;
  3. четыре контрольных спана: целиком в контексте или нет;
  4. лексическая достижимость золотых токенов: joined_reference против всех docs.

Отдельно: поле `docs` в CSV — это `repr` списка словарей, а не текст. Первый
пересчёт 55 считал нормализацию по repr и получал мусор в строке; здесь docs
парсится и склеивается только `doc_text`.
"""
import argparse
import ast
import difflib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl, token_positions
from spanver.io_csv import read_gold

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def normalize(text):
    text = re.sub(r"\[\d+\]", " ", text or "")
    text = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def parse_docs(raw):
    """CSV хранит docs как repr списка словарей. Возвращает список doc_text."""
    if not raw:
        return []
    text = str(raw).strip()
    if text.startswith("[") or text.startswith("{"):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            return [item.get("doc_text", "") for item in value if isinstance(item, dict)]
    return [text]


def coverage(text, corpus):
    if not text or not corpus:
        return 0.0
    match = difflib.SequenceMatcher(None, text, corpus, autojunk=False).find_longest_match(
        0, len(text), 0, len(corpus))
    return match.size / len(text)


def build_gold():
    table = {}
    for prefix, name in (("sber", "sberquad-rag.csv"), ("rusci", "ruscibench-rag.csv")):
        rows = list(read_gold(ROOT / "Dataset/1_rushallu_rag_gold" / name))
        for index, row in enumerate(rows):
            table[f"{prefix}{index:06d}"] = row
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/audit55.json")
    args = parser.parse_args()
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    gold = build_gold()
    missing = [pair["pair_id"] for pair in pairs if pair["pair_id"] not in gold]
    if missing:
        raise ValueError(f"пары без исходной записи: {missing[:5]}")

    spans = [(pair, span) for pair in pairs for span in pair["spans"]]
    rows = []
    for pair, span in spans:
        row = gold[pair["pair_id"]]
        answer = pair["model_output"]
        text = normalize(answer[span["start"]:span["end"] + 1])
        docs = parse_docs(row.get("docs"))
        corpus_docs = normalize(" ".join(docs))
        corpus_joined = normalize(str(row.get("joined_reference") or ""))
        repo_document = normalize(pair["document_text"])
        rows.append({
            "pair_id": pair["pair_id"],
            "type": span["type"],
            "type_src": span.get("type_src"),
            "length": span["end"] - span["start"] + 1,
            "text": text,
            "coverage_docs": coverage(text, corpus_docs),
            "coverage_joined": coverage(text, corpus_joined),
            "coverage_repo": coverage(text, repo_document),
            "in_docs_exact": bool(text) and text in corpus_docs,
            "in_joined_exact": bool(text) and text in corpus_joined,
            "n_docs": len(docs),
        })

    # --- проверка 1: autojunk ---
    autojunk = {}
    for flag in (True, False):
        values = []
        for row in rows:
            raw_docs = normalize(" ".join(parse_docs(gold[row["pair_id"]].get("docs"))))
            match = difflib.SequenceMatcher(None, row["text"], raw_docs, autojunk=flag).find_longest_match(
                0, len(row["text"]), 0, len(raw_docs))
            values.append(match.size / len(row["text"]) if row["text"] else 0.0)
        autojunk[str(flag)] = {
            "median": statistics.median(values),
            "mean": sum(values) / len(values),
            "share_ge_0_9": sum(1 for value in values if value >= 0.9) / len(values),
            "share_ge_0_5": sum(1 for value in values if value >= 0.5) / len(values),
        }

    # --- проверка 2: полный охват ---
    by_src = defaultdict(lambda: {"count": 0, "coverage": [], "exact": 0})
    for row in rows:
        entry = by_src[row["type_src"]]
        entry["count"] += 1
        entry["coverage"].append(row["coverage_docs"])
        entry["exact"] += row["in_docs_exact"]

    # --- проверка 4: лексическая достижимость золотых токенов ---
    def reachable(pairs_subset, field):
        hit = total = 0
        for pair in pairs_subset:
            row = gold[pair["pair_id"]]
            answer = pair["model_output"]
            positions = token_positions(answer)
            if field == "joined":
                corpus = normalize(str(row.get("joined_reference") or ""))
            else:
                corpus = normalize(" ".join(parse_docs(row.get("docs"))))
            words = set(WORD_RE.findall(corpus))
            gold_indices = set()
            for span in pair["spans"]:
                for index, (start, end) in enumerate(positions):
                    if min(end, span["end"]) - max(start, span["start"]) >= 0:
                        gold_indices.add(index)
            total += len(gold_indices)
            hit += sum(1 for index in gold_indices
                       if WORD_RE.findall(answer[positions[index][0]:positions[index][1] + 1])
                       and WORD_RE.findall(answer[positions[index][0]:positions[index][1] + 1])[0].lower().replace("ё", "е") in words)
        return hit, total, (hit / total if total else None)

    positive = [pair for pair in pairs if pair["spans"]]
    reach_joined = reachable(positive, "joined")
    reach_docs = reachable(positive, "docs")

    report = {
        "status": "готово",
        "scope": "все 699 пар train/val, 300 спанов; test 301 не оценивался",
        "coverage_csv": len(pairs),
        "check1_autojunk": autojunk,
        "check2_full_coverage": {
            "spans": len(rows),
            "coverage_docs_median": statistics.median([r["coverage_docs"] for r in rows]),
            "coverage_joined_median": statistics.median([r["coverage_joined"] for r in rows]),
            "coverage_repo_median": statistics.median([r["coverage_repo"] for r in rows]),
            "exact_in_docs": sum(1 for r in rows if r["in_docs_exact"]),
            "exact_in_joined": sum(1 for r in rows if r["in_joined_exact"]),
            "share_ge_0_9_docs": sum(1 for r in rows if r["coverage_docs"] >= 0.9) / len(rows),
            "share_ge_0_5_docs": sum(1 for r in rows if r["coverage_docs"] >= 0.5) / len(rows),
            "length_median": statistics.median([r["length"] for r in rows]),
            "by_human_type": {
                key: {"count": value["count"],
                      "coverage_median": statistics.median(value["coverage"]),
                      "exact": value["exact"]}
                for key, value in sorted(by_src.items())},
        },
        "check3_control_spans": [
            {"pair_id": row["pair_id"], "text": row["text"], "in_docs_exact": row["in_docs_exact"],
             "coverage_docs": row["coverage_docs"], "coverage_joined": row["coverage_joined"]}
            for row in rows if row["pair_id"] in
            {"sber000397", "sber000652", "sber000522", "sber000524"}],
        "check4_reachability": {
            "joined": {"hit": reach_joined[0], "total": reach_joined[1], "share": reach_joined[2]},
            "docs": {"hit": reach_docs[0], "total": reach_docs[1], "share": reach_docs[2]},
            "delta": (reach_docs[2] - reach_joined[2]) if reach_docs[2] and reach_joined[2] else None,
        },
        "n_docs_median": statistics.median([r["n_docs"] for r in rows]),
        "test301": False,
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "check1_autojunk", "check2_full_coverage", "check4_reachability", "n_docs_median")},
        ensure_ascii=False, indent=1)[:3000])
    print("контрольные спаны:")
    for item in report["check3_control_spans"]:
        print(f"  {item['pair_id']} в документе целиком: {item['in_docs_exact']} "
              f"покрытие {item['coverage_docs']:.4f} | {item['text'][:50]}")


if __name__ == "__main__":
    main()
