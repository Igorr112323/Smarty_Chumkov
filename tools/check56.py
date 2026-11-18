"""Ответ на разбор 55: проверка снятых чисел, правило достижимости, вопрос корпуса.

Что считается здесь (и чего нет):

  1. медиана reference_docs — на 699 и на 239 парах, чтобы снять спор о 1 против 2;
  2. тождество joined_reference и reference_docs — прямое сравнение по всем 1000 строкам;
  3. достижимость золотых токенов на reference_docs, а не «не измерено»;
  4. пересечение doc_id между test и не-test на reference_docs;
  5. правило подсчёта, дающее 167 / 50 / 22 и 0,6454;
  6. видит ли модель reference_docs: сверка поля prompt в parquet.

Пункт 6 — находка, которая меняет постановку стадии A: если reference_docs нет
в prompt, то пересборка на reference_docs даёт вход, которого модель при
разметке не видела.
"""
import argparse
import ast
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
from spanver.pairs import doc_ids
from spanver.splits import canonical_assignment

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def normalize(text):
    text = re.sub(r"\[\d+\]", " ", text or "")
    text = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def parse_docs(raw):
    text = str(raw or "").strip()
    if not text.startswith("["):
        return [{"doc_text": text}] if text else []
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, dict):
        value = [value]
    return [item for item in value if isinstance(item, dict)]


def words(text):
    return WORD_RE.findall(normalize(text))


def corpus_text(docs):
    return normalize(" ".join(item.get("doc_text", "") or "" for item in docs))


def load_gold():
    gold = {}
    for prefix, name in (("sber", "sberquad-rag.csv"), ("rusci", "ruscibench-rag.csv")):
        for index, row in enumerate(read_gold(ROOT / "Dataset/1_rushallu_rag_gold" / name)):
            gold[f"{prefix}{index:06d}"] = row
    return gold


def reachability(pairs, gold, field):
    """Доля золотых токенов, чья первая словоформа есть в корпусе поля."""
    hit = total = 0
    for pair in pairs:
        row = gold[pair["pair_id"]]
        if field == "joined_reference":
            corpus = normalize(str(row.get("joined_reference") or ""))
        else:
            corpus = corpus_text(parse_docs(row[field]))
        known = set(words(corpus))
        answer = pair["model_output"]
        positions = token_positions(answer)
        indices = set()
        for span in pair["spans"]:
            for index, (start, end) in enumerate(positions):
                if min(end, span["end"]) - max(start, span["start"]) >= 0:
                    indices.add(index)
        for index in indices:
            start, end = positions[index]
            found = WORD_RE.findall(answer[start:end + 1])
            total += 1
            if found and found[0].lower().replace("ё", "е") in known:
                hit += 1
    return hit, total, hit / total if total else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours33.jsonl")
    parser.add_argument("--out", default="artifacts/check56.json")
    args = parser.parse_args()

    pairs = read_jsonl(ROOT / args.pairs)
    gold = load_gold()
    with_spans = [pair for pair in pairs if pair["spans"]]

    report = {
        "status": "готово",
        "scope": f"{len(pairs)} пар train/val, {sum(len(p['spans']) for p in pairs)} спанов; test 301 не оценивался",
        "test301": False,
    }

    # 1. медиана reference_docs
    report["reference_docs_count"] = {}
    for label, subset in (("699 пар", pairs), ("239 пар со спанами", with_spans)):
        counts = [len(parse_docs(gold[p["pair_id"]]["reference_docs"])) for p in subset]
        report["reference_docs_count"][label] = {
            "median": statistics.median(counts),
            "mean": sum(counts) / len(counts),
            "distribution": dict(sorted(Counter(counts).items())),
        }

    # 2. joined_reference == склейка reference_docs
    identical = 0
    for pair_id, row in gold.items():
        rebuilt = "\n\n".join(
            f"## {item.get('doc_id')}\n{item.get('doc_text', '')}"
            for item in parse_docs(row["reference_docs"]))
        identical += rebuilt == row["joined_reference"]
    report["joined_equals_reference_docs"] = {
        "identical": identical, "rows": len(gold),
        "share": identical / len(gold),
        "note": "склейка через двойной перевод строки; поэтому joined_reference и reference_docs — один корпус",
    }

    # 3. достижимость на трёх корпусах
    report["reachability"] = {}
    for field in ("joined_reference", "reference_docs", "docs"):
        hit, total, share = reachability(pairs, gold, field)
        report["reachability"][field] = {"hit": hit, "total": total, "share": share}

    # 4. пересечение doc_id при разбиении по canonical_assignment
    assignment = canonical_assignment()
    report["docid_overlap_test_vs_rest"] = {}
    for field in ("joined_reference", "reference_docs", "docs"):
        seen = defaultdict(lambda: [0, 0])
        for pair_id, row in gold.items():
            if pair_id not in assignment:
                continue
            ids = doc_ids(row["joined_reference"]) if field == "joined_reference" else \
                [str(item.get("doc_id")) for item in parse_docs(row[field])]
            for ident in ids:
                seen[ident][0 if assignment[pair_id] == "test" else 1] += 1
        overlap = [ident for ident, (in_test, out_test) in seen.items() if in_test and out_test]
        report["docid_overlap_test_vs_rest"][field] = {
            "total": len(seen),
            "in_test": sum(1 for value in seen.values() if value[0]),
            "out_test": sum(1 for value in seen.values() if value[1]),
            "overlap": len(overlap),
            "share": len(overlap) / len(seen) if seen else None,
        }

    # 5. правило «число найденных токенов» → 167 / 50 / 22
    better = same = worse = 0
    for pair in with_spans:
        row = gold[pair["pair_id"]]
        joined = {word.lower().replace("ё", "е") for word in words(row["joined_reference"])}
        docs = {word.lower().replace("ё", "е") for word in words(corpus_text(parse_docs(row["docs"])))}
        answer = pair["model_output"]
        positions = token_positions(answer)
        indices = set()
        for span in pair["spans"]:
            for index, (start, end) in enumerate(positions):
                if min(end, span["end"]) - max(start, span["start"]) >= 0:
                    indices.add(index)
        hits_joined = hits_docs = 0
        for index in indices:
            start, end = positions[index]
            found = WORD_RE.findall(answer[start:end + 1])
            if not found:
                continue
            word = found[0].lower().replace("ё", "е")
            hits_joined += word in joined
            hits_docs += word in docs
        if hits_docs > hits_joined:
            better += 1
        elif hits_docs < hits_joined:
            worse += 1
        else:
            same += 1
    report["pair_rule_token_count"] = {"better": better, "same": same, "worse": worse,
                                       "total": better + same + worse}

    # 6. видит ли модель reference_docs: сверка prompt в parquet
    try:
        import pyarrow.parquet as pq
        path = ROOT / "Dataset/2_rushallu_rag_pairs/sberquad-train-rag.parquet"
        table = pq.ParquetFile(path)
        total = prompt_eq_docs = ref_subset = ref_missing = 0
        for group in range(table.num_row_groups):
            chunk = table.read_row_group(group)
            for index in range(chunk.num_rows):
                prompt = chunk.column("prompt")[index].as_py()
                mark = prompt.rfind("## Input Data")
                shown = set(re.findall(r'"([\d]{2,6})"\s*:', prompt[mark:]))
                docs = {str(item["doc_id"]) for item in chunk.column("docs")[index].as_py()}
                ref = {str(item["doc_id"]) for item in chunk.column("reference_docs")[index].as_py()}
                total += 1
                prompt_eq_docs += shown == docs
                ref_subset += ref <= shown
                ref_missing += not (ref & shown)
        report["prompt_vs_reference_docs"] = {
            "rows": total,
            "prompt_ids_equal_docs": prompt_eq_docs,
            "reference_subset_of_prompt": ref_subset,
            "reference_disjoint_from_prompt": ref_missing,
            "note": "parquet 2_rushallu_rag_pairs — другой срез, query_id с gold не пересекается; "
                    "проверка показывает устройство поля prompt, а не наши 699 пар",
        }
    except ImportError:
        report["prompt_vs_reference_docs"] = {"status": "пропущено: нет pyarrow"}

    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
