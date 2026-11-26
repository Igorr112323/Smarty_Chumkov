"""Собирает договорной набор 1 200 пар (версия 2) из доступных разметок.

Отличия от v1:
* mFAVA_ru конвертируется точным проходом по тегам с отслеживанием позиций,
  а не поиском текста через `str.find`; это убирает случайные смещения и
  потерю спанов при повторяющихся подстроках;
* в наборе 200 русских пар mFAVA_ru: первые 150 по порядку источника идут в
  train, последние 50 — в document-disjoint held-out. В v1 было 48/200.
  Причина: 48 пар недостаточно для обучения локального каскада; деление
  детерминированное, по порядку исходного файла, без оглядки на метрики.

Состав: 1 000 RusHallu-RAG из artifacts/gold.jsonl + 200 mFAVA_ru = 1 200.
held-out из 301 RusHallu `test` + 50 mFAVA. Старый набор и отчёты сохраняются.
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
TAG_RE = re.compile(r"</?([a-z_]+)>")
MFAVA_HELD = 50
MFAVA_IN_SET = 200


def norm(value):
    return (value or "").replace("\xa0", " ").replace("\u200b", "")


def convert_mfava(row, index):
    """Возвращает пару с точными координатами тегированных фрагментов."""
    annotation = norm(row.get("gold_annotations"))
    answer = norm(row.get("generated_text"))
    document = norm(row.get("references"))
    if not answer or not document:
        return None
    clean = []
    spans = []
    stack = []
    position = 0
    for match in TAG_RE.finditer(annotation):
        piece = annotation[position : match.start()]
        for char in piece:
            clean.append(char)
        tag = match.group(1)
        if match.group(0).startswith("</"):
            if stack:
                opened_tag, opened_at = stack.pop()
                text = "".join(clean[opened_at:]).strip()
                if opened_tag in BAD_TAGS and len(text) >= 3:
                    start = answer.find(text)
                    if start >= 0:
                        spans.append((start, start + len(text) - 1, opened_tag))
        else:
            stack.append((tag, len(clean)))
        position = match.end()
    for char in annotation[position:]:
        clean.append(char)
    # Закрытые теги обработаны выше; незакрытые теги в конце разметки
    # игнорируются, чтобы не разметить хвост ответа как галлюцинацию.
    if not spans:
        return None
    gold = []
    for start, end, tag in spans:
        gold.append(
            {
                "start": start,
                "end": end,
                "text": answer[start : end + 1],
                "type": "unsupported_addition",
                "type_src": tag,
            }
        )
    return {
        "pair_id": f"mfava{index:06d}",
        "source": "Dataset/mFAVA_ru/test.json",
        "domain": "mfava",
        "document_text": document,
        "doc_ids": [f"mfava-doc-{index}"],
        "model_output": answer,
        "spans": gold,
        "split": "extra",
        "query_text": "",
        "clean": False,
    }


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
    raw = json.loads(Path(args.mfava).read_text(encoding="utf-8"))
    converted = []
    for index, row in enumerate(raw):
        if row.get("language") != "russian":
            continue
        pair = convert_mfava(row, index)
        if pair:
            converted.append(pair)
    if len(converted) < MFAVA_IN_SET:
        raise SystemExit(f"mFAVA_ru: {len(converted)} пар, нужно {MFAVA_IN_SET}")
    selected = converted[:MFAVA_IN_SET]
    train_extra = selected[: MFAVA_IN_SET - MFAVA_HELD]
    held_extra = selected[MFAVA_IN_SET - MFAVA_HELD :]
    for pair in held_extra:
        pair["split"] = "extra"
    for pair in train_extra:
        pair["split"] = "train_extra"
    combined = gold + train_extra + held_extra
    docs_gold = {doc for pair in gold for doc in pair.get("doc_ids") or []}
    docs_mfava = {doc for pair in train_extra + held_extra for doc in pair.get("doc_ids") or []}
    overlap = (docs_gold & docs_mfava) | (docs_mfava & {doc for pair in train_extra for doc in pair.get("doc_ids") or []} & {doc for pair in held_extra for doc in pair.get("doc_ids") or []})
    train_sha = write_jsonl(out / "mfava_train_v2.jsonl", train_extra)
    gold_sha = write_jsonl(out / "gold1200_v2.jsonl", combined)
    report = {
        "version": 2,
        "gold_pairs": len(gold),
        "mfava_converted_pairs": len(converted),
        "mfava_in_set": len(selected),
        "mfava_train": len(train_extra),
        "mfava_held": len(held_extra),
        "total_1200": len(combined),
        "held_total": sum(1 for pair in combined if pair.get("split") == "test") + len(held_extra),
        "doc_overlap": len(overlap),
        "mfava_bad_spans_total": sum(len(pair["spans"]) for pair in converted),
        "mfava_train_spans": sum(len(pair["spans"]) for pair in train_extra),
        "mfava_held_spans": sum(len(pair["spans"]) for pair in held_extra),
        "gold1200_v2_sha256": gold_sha,
        "mfava_train_v2_sha256": train_sha,
        "conversion": "positional tag walk; NBSP/ZWSP normalization; first-occurrence fallback",
        "split_rule": "first 150 mFAVA pairs -> train, next 50 -> held; deterministic source order",
    }
    (out / "gold1200_v2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not overlap else 2


if __name__ == "__main__":
    raise SystemExit(main())
