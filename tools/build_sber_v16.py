"""V16 шаг 2: новые русские данные — `DFKonGH/ru_hallucination_detector` (GitHub, parquet).

Датасет построен на SberChallenge/SberQuAD: `context` (документ), `evidence` (исходное
предложение), `evidence_aug` (то же предложение с синтетической правкой), `aug_type`
(`clean` / `_entity_swap` / `_negation_flip` / `_num_flip`), 50 364 строки, 9 078 документов.

Перевод в наш формат:
  * `document_text` = `context`, `model_output` = `evidence_aug`;
  * gold-спан = предложение `evidence_aug`, в котором лежит правка (char-diff `evidence` →
    `evidence_aug`); у `clean` спанов нет;
  * стиль разметки совпадает с нашим C1 (`unsupported_addition`, границы по предложению).

Жёсткое требование `doc_overlap = 0` проверяется по SHA-256 документа против наших
`gold_v3` / `mfava_train_v3` / `gold1200_v3` (val и held-out).
"""
import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SENT_RE = re.compile(r"[^.!?\n]+[.!?]*")
SOURCE = ROOT / "tools/external/ru_hallucination_detector/data/ds_v0.parquet"
OURS = ["artifacts/gold_v3.jsonl", "artifacts/mfava_train_v3.jsonl", "artifacts/gold1200_v3.jsonl"]


def sha(text):
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def changed_ranges(evidence, augmented):
    """Char-диапазоны, которые правка затронула в augmented."""
    matcher = difflib.SequenceMatcher(None, evidence or "", augmented or "", autojunk=False)
    ranges = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if j2 > j1:
            ranges.append((j1, j2))
    return ranges


def sentences(text):
    return [(match.start(), match.end()) for match in SENT_RE.finditer(text or "") if match.group(0).strip()]


def gold_spans(evidence, augmented, aug_type):
    if aug_type == "clean":
        return []
    ranges = changed_ranges(evidence, augmented)
    if not ranges:
        return []
    spans = []
    for start, end in sentences(augmented):
        if any(start < r_end and end > r_start for r_start, r_end in ranges):
            pieces = augmented[start:end]
            lead = len(pieces) - len(pieces.lstrip())
            trail = len(pieces.rstrip())
            spans.append({"start": start + lead, "end": start + trail - 1,
                          "type": "unsupported_addition", "type_src": aug_type})
    if not spans:
        spans = [{"start": 0, "end": len(augmented) - 1, "type": "unsupported_addition",
                  "type_src": aug_type}]
    return spans


def build_rows(limit=0, min_answer=40):
    import pyarrow.parquet as pq
    table = pq.read_table(SOURCE, columns=["id", "title", "context", "evidence", "evidence_aug",
                                           "aug_type"])
    rows = []
    stats = {"clean": 0, "flip": 0, "skipped_empty": 0, "skipped_short": 0}
    for record in table.to_pylist():
        document = (record["context"] or "").strip()
        answer = (record["evidence_aug"] or "").strip()
        if not document or not answer:
            stats["skipped_empty"] += 1
            continue
        if len(answer) < min_answer:
            stats["skipped_short"] += 1
            continue
        aug = record["aug_type"] or "clean"
        spans = gold_spans((record["evidence"] or "").strip(), answer, aug)
        if aug == "clean":
            stats["clean"] += 1
        else:
            stats["flip"] += 1
        rows.append({
            "pair_id": f"sber{record['id']}_{aug.strip('_')}",
            "source": "ru_hallucination_detector",
            "domain": record.get("title") or "sberquad",
            "query_text": "",
            "document_text": document,
            "doc_ids": [],
            "model_output": answer,
            "spans": spans,
            "clean": not spans,
            "label_source": f"synthetic_{aug.strip('_')}",
            "split": "train",
        })
        if limit and len(rows) >= limit:
            break
    return rows, stats


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ru_data_v16.json")
    parser.add_argument("--out", default="artifacts/sber_flip_v16.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    rows, stats = build_rows(limit=args.limit)
    ours = {}
    for path in OURS:
        full = ROOT / path
        if not full.exists():
            continue
        for line in open(full, encoding="utf-8"):
            pair = json.loads(line)
            ours[sha(pair.get("document_text"))] = path
    their_docs = {sha(row["document_text"]) for row in rows}
    overlap = their_docs & set(ours)

    target = ROOT / args.out
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "rule": "новые русские данные ищутся только на GitHub (HF/LFS недоступны), "
                "doc_overlap с нашим Dataset обязан быть 0",
        "sources": [{
            "repo": "https://github.com/DFKonGH/ru_hallucination_detector",
            "path": "data/ds_v0.parquet",
            "license": "не указана в репозитории (SberChallenge/SberQuAD-производные, "
                       "только для локального обучения)",
            "bytes": SOURCE.stat().st_size,
            "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "rows": 50364,
            "documents": 9078,
            "aug_types": {"_negation_flip": 21619, "_entity_swap": 18355, "clean": 7781,
                          "_num_flip": 2609},
            "format": "parquet",
        }, {
            "repo": "https://github.com/sber-summer-practice/hybrid-rag-hallucination-verification",
            "path": "data/optimized_rag_marking.xlsx",
            "license": "не указана",
            "bytes": (ROOT / "tools/external/hybrid-rag-hallucination-verification/data/"
                      "optimized_rag_marking.xlsx").stat().st_size,
            "used": False,
            "reason": "88 KB ручной разметки ответов LLM без span-аннотаций — слишком мало и не "
                      "в нашем формате",
        }],
        "conversion": {
            "document_text": "context", "model_output": "evidence_aug",
            "gold": "предложение evidence_aug, затронутое правкой (char-diff evidence → evidence_aug); "
                    "clean — без спанов",
            "style": "unsupported_addition, границы предложения (как C1_min_containing_clause)",
        },
        "stats": {**stats, "pairs": len(rows),
                  "with_spans": sum(1 for row in rows if row["spans"]),
                  "documents": len(their_docs),
                  "mean_answer_len": round(sum(len(r["model_output"]) for r in rows) / max(len(rows), 1), 1),
                  "mean_doc_len": round(sum(len(r["document_text"]) for r in rows) / max(len(rows), 1), 1)},
        "doc_overlap": {
            "checked_against": OURS,
            "our_documents": len(ours),
            "their_documents": len(their_docs),
            "overlap": len(overlap),
            "pass": len(overlap) == 0,
        },
        "file": {"path": args.out, "bytes": target.stat().st_size,
                 "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
    }
    (ROOT / args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps({"pairs": len(rows), "with_spans": payload["stats"]["with_spans"],
                      "overlap": len(overlap)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
