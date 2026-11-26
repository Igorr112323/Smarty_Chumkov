"""Формирует набор из 1200 пар: 1000 RusHallu-RAG + 200 удержанных mFAVA_ru.

mFAVA_ru/test.json содержит многоязычный набор; берём только `language == "russian"`.
Из разметки gold_annotations извлекаются теги недоверенных фрагментов
(contradictory/unverifiable/invented и опечатки этих названий); субъективные,
(entity/relation) и прочие метки в спаны не превращаются. Первые 48 пар уходят
в дополнительный train, следующие 200 — в удержанную часть набора (split=extra).
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAD_TAGS = {
    "contradictory",
    "unverifiable",
    "invented",
    "unverified",
    "conradictory",
    "unvented",
    "unverifiabe",
}
TRAIN_EXTRA = 48
EXTRA_HELD = 200


def norm(value):
    return (value or "").replace("\xa0", " ")


def mfava_pairs(source):
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    result = []
    for index, row in enumerate(rows):
        if row.get("language") != "russian":
            continue
        answer = norm(row.get("generated_text"))
        document = norm(row.get("references"))
        annotation = norm(row.get("gold_annotations"))
        if not answer or not document:
            continue
        spans = []
        taken = []
        for match in re.finditer(r"<([a-z_]+)>(.*?)</\1>", annotation, re.S):
            tag = match.group(1)
            text = re.sub(r"</?[a-z_]+>", "", match.group(2)).strip()
            if tag not in BAD_TAGS or len(text) < 3:
                continue
            start = answer.find(text)
            if start < 0:
                continue
            end = start + len(text) - 1
            if any(start < stop and begin < end for begin, stop in taken):
                continue
            taken.append((start, end))
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "text": answer[start : end + 1],
                    "type": "unsupported_addition",
                    "type_src": tag,
                }
            )
        if not spans:
            continue
        result.append(
            {
                "pair_id": f"mfava{index:06d}",
                "source": "Dataset/mFAVA_ru/test.json",
                "domain": "mfava",
                "document_text": document,
                "doc_ids": [f"mfava-doc-{index}"],
                "model_output": answer,
                "spans": spans,
                "split": "extra",
                "query_text": "",
                "clean": False,
            }
        )
    return result


def write_jsonl(path, pairs):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--mfava", default="Dataset/mFAVA_ru/test.json")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gold = [
        json.loads(line)
        for line in Path(args.gold).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = mfava_pairs(args.mfava)
    if len(pairs) < TRAIN_EXTRA + EXTRA_HELD:
        raise SystemExit(f"mFAVA_ru: получено {len(pairs)} пар, нужно {TRAIN_EXTRA + EXTRA_HELD}")
    train_extra = pairs[:TRAIN_EXTRA]
    held = pairs[TRAIN_EXTRA : TRAIN_EXTRA + EXTRA_HELD]
    for pair in train_extra:
        pair["split"] = "train_extra"
    combined = gold + held
    gold_sha = write_jsonl(out / "gold1200.jsonl", combined)
    train_sha = write_jsonl(out / "mfava_train_extra.jsonl", train_extra)

    gold_docs = {doc for pair in gold for doc in pair.get("doc_ids") or []}
    held_docs = {doc for pair in held for doc in pair.get("doc_ids") or []}
    overlap = gold_docs & held_docs
    report = {
        "gold_pairs": len(gold),
        "mfava_russian_pairs": len(pairs),
        "mfava_train_extra": len(train_extra),
        "mfava_held_extra": len(held),
        "total_1200": len(combined),
        "held_total": sum(1 for pair in combined if pair["split"] == "test") + len(held),
        "doc_overlap": len(overlap),
        "mfava_bad_spans": sum(len(pair["spans"]) for pair in pairs),
        "gold1200_sha256": gold_sha,
        "train_extra_sha256": train_sha,
    }
    (out / "gold1200_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not overlap else 2


if __name__ == "__main__":
    raise SystemExit(main())
