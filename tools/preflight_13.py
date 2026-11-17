import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.answer_json import parse_answer
from spanver.build import GOLD_FILES
from spanver.io_csv import read_gold
from spanver.locate import locate
from spanver.pairs import doc_ids
from spanver.types import classify

SEED = 0


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def components(rows, keys):
    parents = list(range(len(rows)))
    seen = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(rows):
        for key in keys(row):
            if key in seen:
                left, right = root(index), root(seen[key])
                if left != right:
                    parents[left] = right
            else:
                seen[key] = index
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(root(index), []).append(row)
    return sorted(groups.values(), key=lambda group: min(row["pair_id"] for row in group))


def split_groups(groups, count, seed=SEED):
    groups = list(groups)
    random.Random(seed).shuffle(groups)
    return [[row for group in groups[index::count] for row in group] for index in range(count)]


def our_rows():
    with (ROOT / "spanver/splits.json").open(encoding="utf-8") as handle:
        splits = json.load(handle)
    allowed = set(splits["train"]) | set(splits["val"])
    result = []
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
                spans.append({"text": item["text"], "start": found["start"], "end": found["end"], "type_src": item["type"], "type": classify(item["text"], item["type"]), "evidence_ids": []})
            document = row.get("joined_reference") or ""
            result.append({"pair_id": pair_id, "domain": domain, "source": source, "document_text": document, "doc_ids": doc_ids(document), "model_output": output, "spans": spans})
    return result


def our_keys(row):
    return [("doc", value) for value in row["doc_ids"]]


def rag_keys(row):
    text_hash = hashlib.sha256(row["document_text"].encode("utf-8")).hexdigest()
    return [("id", row["doc_id"]), ("text", text_hash)]


def id_digest(rows):
    data = "\n".join(sorted(row["pair_id"] for row in rows)) + "\n"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def summarize(rows):
    lengths = [span["end"] - span["start"] + 1 for row in rows for span in row.get("spans", [])]
    invalid = sum(not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int) or not 0 <= span["start"] <= span["end"] < len(row.get("model_output") or "") for row in rows for span in row.get("spans", []))
    return {"pairs": len(rows), "spans": len(lengths), "median_span_chars": statistics.median(lengths) if lengths else None, "invalid_spans": invalid, "empty_documents": sum(not row.get("document_text") for row in rows), "id_sha256": id_digest(rows)}


def main():
    paths = [ROOT / "Dataset/RAGTruth_ru/ragtruth_ru.jsonl", ROOT / "Dataset/RAGTruth_ru/ragtruth_ru_negative.jsonl"]
    data = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            data.append([json.loads(line) for line in handle if line.strip()])
    rag = data[0] + data[1]
    ours = our_rows()
    rag_groups = components(rag, rag_keys)
    rag_folds = split_groups(rag_groups, 5)
    ours_groups = components(ours, our_keys)
    our_folds = split_groups(ours_groups, 5)
    rag_train = [row for fold in rag_folds[1:] for row in fold]
    rag_holdout = rag_folds[0]
    if {key for row in rag_train for key in rag_keys(row)} & {key for row in rag_holdout for key in rag_keys(row)}:
        raise ValueError("Документы RAGTruth пересекают границу holdout")
    for index, fold in enumerate(our_folds):
        other = [row for j, other_fold in enumerate(our_folds) if j != index for row in other_fold]
        if {key for row in fold for key in our_keys(row)} & {key for row in other for key in our_keys(row)}:
            raise ValueError("Документы наших данных пересекают границу фолда")
    text_ids = {}
    text_hashes = {}
    for row in rag:
        ident = row["doc_id"]
        hashed = hashlib.sha256(row["document_text"].encode("utf-8")).hexdigest()
        text_ids.setdefault(ident, set()).add(hashed)
        text_hashes.setdefault(hashed, set()).add(ident)
    report = {
        "status": "подготовка и аудит; обучение и метрики не выполнены",
        "seed": SEED,
        "inputs": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "rag_positive": summarize(data[0]),
        "rag_negative": summarize(data[1]),
        "rag_types": dict(sorted(Counter(span.get("type", "") for row in data[0] for span in row.get("spans", [])).items())),
        "rag_doc_ids": len(text_ids),
        "rag_document_texts": len(text_hashes),
        "rag_ids_with_multiple_texts": sum(len(values) > 1 for values in text_ids.values()),
        "rag_texts_with_multiple_ids": sum(len(values) > 1 for values in text_hashes.values()),
        "rag_ids_not_sha256_of_translated_document": sum(row["doc_id"] != hashlib.sha256(row["document_text"].encode("utf-8")).hexdigest()[:16] for row in rag),
        "rag_holdout_method": "связные компоненты по doc_id и точному тексту; seed 0; первая из пяти групп",
        "rag_train": summarize(rag_train),
        "rag_holdout": summarize(rag_holdout),
        "rag_train_doc_ids": len({row["doc_id"] for row in rag_train}),
        "rag_holdout_doc_ids": len({row["doc_id"] for row in rag_holdout}),
        "rag_doc_overlap": 0,
        "ours_train_val_only": summarize(ours),
        "ours_components": len(ours_groups),
        "ours_fold_sizes": [len(fold) for fold in our_folds],
        "ours_fold_id_sha256": [id_digest(fold) for fold in our_folds],
        "notes": ["Разметка test не обрабатывалась и оценка test не проводилась.", "Для честной оценки RAGTruth ступень A должна использовать только 80% документов.", "Типы RAGTruth не совпадают с типами наших данных; прямое сравнение точности типа некорректно."],
    }
    if len({row["pair_id"] for row in rag}) != len(rag) or any(not row.get("doc_id") or not row.get("document_text") for row in rag) or any(not row.get("spans") for row in data[0]) or any(row.get("spans") for row in data[1]) or report["rag_positive"]["invalid_spans"] or report["rag_negative"]["invalid_spans"]:
        raise ValueError("Некорректные данные RAGTruth")
    target = ROOT / "artifacts/preflight_13.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
