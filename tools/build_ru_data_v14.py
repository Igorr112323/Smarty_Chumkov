"""V14 шаг 1–2: поиск и сборка русских/переносимых данных для span-классификатора.

Что проверено (только программно, без ручной разметки):

  1. feudor2/RusHallu-RAG (Apache-2.0) — ИСХОДНИК нашего gold. Все 959 уникальных
     документов из results/annotation/human/merged/*.csv уже есть в наших
     gold_v3/gold1200_v3/mfava_train_v3 → doc_overlap = 959/959 = 100 %,
     пригодных для обучения новых пар с doc_overlap=0 — 0.
  2. kaengreg/rusBEIR (лицензия не указана) — вики-корпус лежит в Git LFS
     (`wikifacts-datasets/wiki-corpus.jsonl` = указатель на 440 МБ), LFS-хост в
     песочнице недоступен → 0 пригодных документов.
  3. ParticleMedia/RAGTruth (MIT) — 17 790 ответов с чар-спанами, английский,
     данные лежат в репозитории (`dataset/response.jsonl` + `source_info.jsonl`).
     Используется как проекция стиля (структурные признаки), doc_overlap = 0.
  4. HF-only наборы (hallucination-detection-benchmark ru-часть, RU-RAG/RU-RAGTruth,
     полный mFAVA, RAGTruth-RU) — хаб huggingface.co в песочнице недоступен
     (разрешены github/codeload/api.github/npm/pypi) → зафиксировано как блокер,
     SHA получить нельзя.

Проекция RAGTruth в наш формат: answer↔document, чар-спаны, правило C1
(min containing clause, tools/convert_labels_v3.py) — то же, что применялось к
RusHallu/mFAVA. Пишет artifacts/ragtruth_proj_v14.jsonl и artifacts/ru_dataset_v14.json.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from convert_labels_v3 import convert_pair

EXTERNAL = ROOT / "tools" / "external"
TYPE_MAP = {
    "Evident Conflict": "contradictory",
    "Subtle Conflict": "contradictory",
    "Evident Baseless Info": "unverifiable",
    "Subtle Baseless Info": "unverifiable",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def doc_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def style_of(answer, spans):
    """Стиль пары: whole / clause / short (правило одинаково для всех источников)."""
    if not spans:
        return "clean"
    length = max(1, len(answer or ""))
    best = max(spans, key=lambda span: span["end"] - span["start"])
    span_len = best["end"] - best["start"]
    if span_len / length >= 0.8:
        return "whole"
    if span_len <= 60:
        return "short"
    return "clause"


def our_documents():
    hashes = set()
    for name in ("gold_v3.jsonl", "mfava_train_v3.jsonl", "gold1200_v3.jsonl"):
        path = ROOT / "artifacts" / name
        for line in path.open(encoding="utf-8"):
            hashes.add(doc_hash(json.loads(line).get("document_text")))
    return hashes


def audit_rushallu(ours):
    report = {}
    for name in ("sberquad-rag.csv", "ruscibench-rag.csv"):
        path = EXTERNAL / "RusHallu-RAG" / "results" / "annotation" / "human" / "merged" / name
        if not path.exists():
            report[name] = {"error": "нет файла"}
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        documents = [row.get("joined_reference") or "" for row in rows]
        unique = set(documents)
        overlap = sum(1 for text in unique if doc_hash(text) in ours)
        report[name] = {
            "rows": len(rows),
            "unique_documents": len(unique),
            "documents_overlapping_ours": overlap,
            "usable_new_pairs": 0,
            "reason": "все документы уже в наших сплитах (doc_overlap=100 %), обучение на них "
                      "нарушило бы doc_overlap=0",
            "sha256": sha256_file(path),
        }
    return report


def audit_rusbeir():
    corpus = EXTERNAL / "rusBEIR" / "wikifacts-datasets" / "wiki-corpus.jsonl"
    if not corpus.exists():
        return {"status": "нет файла"}
    head = corpus.read_text(encoding="utf-8", errors="ignore")[:200]
    lfs = head.startswith("version https://git-lfs.github.com/spec")
    size = None
    for line in head.splitlines():
        if line.startswith("size "):
            size = int(line.split()[1])
    return {"status": "git-lfs pointer" if lfs else "обычный файл",
            "lfs_pointer_bytes": size,
            "usable_documents": 0,
            "reason": "LFS-хост недоступен в песочнице; 440 МБ в пакет не тащим"}


def build_projection(limit, max_doc_chars, max_answer_chars):
    responses = EXTERNAL / "RAGTruth" / "dataset" / "response.jsonl"
    sources = EXTERNAL / "RAGTruth" / "dataset" / "source_info.jsonl"
    if not responses.exists() or not sources.exists():
        return [], {"error": "нет RAGTruth в tools/external"}

    info = {}
    with sources.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            info[row["source_id"]] = row

    pairs = []
    skipped = Counter()
    with responses.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "train":
                skipped["split"] += 1
                continue
            if not row.get("labels"):
                skipped["no_labels"] += 1
                continue
            source = info.get(row.get("source_id"))
            if not source:
                skipped["no_source"] += 1
                continue
            answer = row.get("response") or ""
            document = source.get("source_info") or ""
            if isinstance(document, (dict, list)):
                document = json.dumps(document, ensure_ascii=False, sort_keys=True)
            if not answer or not document:
                skipped["empty"] += 1
                continue
            if len(document) > max_doc_chars or len(answer) > max_answer_chars:
                skipped["too_long"] += 1
                continue
            spans = []
            for label in row["labels"]:
                start = int(label.get("start", -1))
                end = int(label.get("end", -1))
                if start < 0 or end <= start or end > len(answer):
                    skipped["bad_offset"] += 1
                    continue
                # C1 в convert_labels_v3 ожидает end < len(answer) (в наших данных так и было)
                end = min(end, len(answer) - 1)
                if end <= start:
                    skipped["bad_offset"] += 1
                    continue
                spans.append({
                    "start": start, "end": end, "text": answer[start:end],
                    "type": "unsupported_addition",
                    "type_src": TYPE_MAP.get(label.get("label_type"), "unverifiable"),
                })
            if not spans:
                skipped["empty_spans"] += 1
                continue
            pairs.append({
                "pair_id": f"ragtruth{len(pairs) + 1:06d}",
                "source": "ragtruth_train_projection",
                "domain": source.get("task_type", "unknown").lower(),
                "query_text": (source.get("prompt") or "")[:400],
                "query_id": str(row.get("source_id")),
                "document_text": document,
                "doc_ids": [str(row.get("source_id"))],
                "model_output": answer,
                "model": row.get("model"),
                "spans": spans,
                "clean": False,
                "label_source": "ragtruth_manual",
                "split": "extra",
            })
            if len(pairs) >= limit:
                break

    # дедуп по документу: один документ — одна пара (как в наших сплитах)
    seen = set()
    unique = []
    for pair in pairs:
        key = doc_hash(pair["document_text"])
        if key in seen:
            skipped["dup_document"] += 1
            continue
        seen.add(key)
        unique.append(pair)
    converted = [convert_pair(pair) for pair in unique]
    for pair in converted:
        pair["style"] = style_of(pair["model_output"], pair["spans"])
    return converted, dict(skipped)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ru_dataset_v14.json")
    parser.add_argument("--output", default="artifacts/ragtruth_proj_v14.jsonl")
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--max-doc-chars", type=int, default=6000)
    parser.add_argument("--max-answer-chars", type=int, default=2000)
    args = parser.parse_args(argv)

    ours = our_documents()
    audit = audit_rushallu(ours)
    beir = audit_rusbeir()
    pairs, skipped = build_projection(args.limit, args.max_doc_chars, args.max_answer_chars)

    output = ROOT / args.output
    write_jsonl(output, pairs)
    overlap = sum(1 for pair in pairs if doc_hash(pair["document_text"]) in ours)

    def style_counts(rows):
        return dict(Counter(row.get("style") for row in rows))

    def span_counts(rows):
        return {
            "pairs": len(rows),
            "spans": sum(len(row.get("spans") or []) for row in rows),
            "spans_per_pair": round(sum(len(row.get("spans") or []) for row in rows)
                                    / max(1, len(rows)), 2),
            "documents": len({doc_hash(row["document_text"]) for row in rows}),
            "type_src": dict(Counter(span.get("type_src") for row in rows
                                     for span in row.get("spans") or [])),
        }

    val = [json.loads(line) for line in (ROOT / "artifacts" / "gold_v3.jsonl").open(encoding="utf-8")
           if json.loads(line).get("split") == "val"]
    for row in val:
        row["style"] = style_of(row.get("model_output") or "", row.get("spans") or [])

    report = {
        "rule": "только программные источники; ручная разметка не использовалась; "
                "doc_overlap с нашими train/val/held-out обязателен = 0",
        "sources": {
            "RusHallu-RAG": {
                "repo": "https://github.com/feudor2/RusHallu-RAG",
                "license": "Apache-2.0 (badge в README; файла LICENSE в дереве нет)",
                "files": audit,
                "verdict": "reject как новые данные: doc_overlap = 100 %",
            },
            "rusBEIR": {
                "repo": "https://github.com/kaengreg/rusBEIR",
                "license": "не указана",
                "corpus": beir,
                "verdict": "reject: корпус в Git LFS, хост недоступен",
            },
            "RAGTruth": {
                "repo": "https://github.com/ParticleMedia/RAGTruth",
                "license": "MIT",
                "language": "en",
                "files": {
                    "response.jsonl": sha256_file(EXTERNAL / "RAGTruth/dataset/response.jsonl"),
                    "source_info.jsonl": sha256_file(EXTERNAL / "RAGTruth/dataset/source_info.jsonl"),
                },
                "verdict": "accept как проекция стиля (структурные признаки), doc_overlap = 0",
            },
            "hf_only": {
                "list": ["hallucination-detection-benchmark (ru)", "RU-QA/RU-RAG", "RU-RAGTruth",
                         "полный mFAVA", "RAGTruth-RU"],
                "blocker": "huggingface.co недоступен в песочнице (разрешены github/codeload/"
                           "api.github/npm/pypi); SHA и статистика получить нельзя",
            },
        },
        "dataset": {
            "file": args.output,
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            **span_counts(pairs),
            "style": style_counts(pairs),
            "doc_overlap_with_ours": overlap,
            "skipped": skipped,
            "normalization": "C1_min_containing_clause (tools/convert_labels_v3.py)",
        },
        "our_val_style": {"pairs": len(val), "style": style_counts(val)},
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report["dataset"], ensure_ascii=False), flush=True)
    print("val style", report["our_val_style"], flush=True)


if __name__ == "__main__":
    main()
