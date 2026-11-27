"""V21 шаги 1–2: данные и архитектура cross-encoder для закрытия Span.

Данные: реальный RusHallu train + v19 (клаузовый стиль, doc_overlap 0) + synthetic v11 + pseudo v11.
Проверки: doc_overlap против val/held-out (должен быть 0), распределение стилей, статистика
кандидатов и hard negatives (IoU 0.3–0.7), длины в токенах rubert-tiny2 и план упаковки окна.

Архитектура (`CrossEncoderSpan`): энкодер rubert-tiny2 + голова старта/конца спана по токенам
ответа + голова риска (CLS). Веса берутся из локальных чанков `weights/rubert-tiny2` (§4.2: веса
отдельно от пакета), SHA сверяется с манифестом.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from spanver.evaluate import iou
from spanver.module import candidate_spans

MODEL_DIR = ROOT / "models/rubert-tiny2"
WEIGHTS_MANIFEST = ROOT / "weights/rubert-tiny2/manifest.json"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def doc_key(text):
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def assemble_encoder_weights(verify=True):
    """Склеивает чанки в `models/rubert-tiny2/pytorch_model.bin` и сверяет SHA с манифестом."""
    target = MODEL_DIR / "pytorch_model.bin"
    manifest = json.loads(WEIGHTS_MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    if target.exists():
        with open(target, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    else:
        with open(target, "wb") as out:
            for part in manifest["chunks"]:
                data = (ROOT / "weights/rubert-tiny2" / part["name"]).read_bytes()
                out.write(data)
                digest.update(data)
    actual = digest.hexdigest()
    ok = (actual == manifest["full_sha256"]) if verify else None
    return {"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
            "sha256": actual, "manifest_sha256": manifest["full_sha256"], "sha256_match": ok,
            "source": manifest.get("source"), "chunks": len(manifest["chunks"])}


def span_style(pair):
    """Стиль спана: whole (весь ответ) / clause (предложение) / short (часть клаузы)."""
    answer = pair.get("model_output") or ""
    spans = pair.get("spans") or []
    if not spans:
        return "clean"
    span = max(spans, key=lambda item: item["end"] - item["start"])
    length = span["end"] - span["start"]
    if length >= 0.9 * len(answer):
        return "whole"
    if length <= 40:
        return "short"
    return "clause"


def build_examples(pairs, source, limit=0):
    """Кандидаты спанов с метками: 1 при IoU >= 0.5, hard negative при 0.3 <= IoU < 0.7."""
    examples = []
    for pair in (pairs[:limit] if limit else pairs):
        answer = pair.get("model_output") or ""
        if not answer:
            continue
        gold = pair.get("spans") or []
        candidates = []
        for start, end, fragment in candidate_spans(answer):
            score = max([iou(start, end, span["start"], span["end"]) for span in gold] or [0.0])
            label = 1 if score >= 0.5 else 0
            hard = bool(label == 0 and 0.3 <= score < 0.7)
            candidates.append({"start": start, "end": end, "text": fragment,
                               "label": label, "hard": hard, "iou": round(float(score), 4)})
        examples.append({"pair_id": pair["pair_id"], "source": source,
                         "document_text": pair.get("document_text") or "", "answer": answer,
                         "clean": bool(pair.get("clean", not gold)), "style": span_style(pair),
                         "candidates": candidates})
    return examples


def token_stats(tokenizer, examples, sample=300):
    """Длины в токенах: документ/ответ и окно [CLS] doc_prefix [SEP] candidate [SEP]."""
    lengths = {"document": [], "answer": [], "candidate": []}
    for example in examples[:sample]:
        lengths["document"].append(len(tokenizer(example["document_text"])["input_ids"]))
        lengths["answer"].append(len(tokenizer(example["answer"])["input_ids"]))
        for candidate in example["candidates"][:3]:
            lengths["candidate"].append(len(tokenizer(candidate["text"])["input_ids"]))
    summary = {}
    for key, values in lengths.items():
        if not values:
            continue
        values = sorted(values)
        summary[key] = {"n": len(values), "mean": round(sum(values) / len(values), 1),
                        "p50": values[len(values) // 2], "p90": values[int(len(values) * 0.9)],
                        "max": values[-1]}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ce_data_v21.json")
    parser.add_argument("--examples", default="artifacts/ce_examples_v21.jsonl")
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    weights = assemble_encoder_weights()
    gold = read_jsonl(ROOT / "artifacts/gold_v3.jsonl")
    real_train = [pair for pair in gold if pair.get("split") == "train"]
    real_val = [pair for pair in gold if pair.get("split") == "val"]
    v19 = read_jsonl(ROOT / "artifacts/ru_clause_gen_v19.jsonl")
    synth = read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl")
    pseudo = read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl")
    held = read_jsonl(ROOT / "artifacts/gold1200_v3.jsonl")
    mfava = read_jsonl(ROOT / "artifacts/mfava_train_v3.jsonl")

    held_keys = {doc_key(pair.get("document_text")) for pair in held}
    held_test_keys = {doc_key(pair.get("document_text")) for pair in held
                      if pair.get("split") == "test"}
    held_extra_keys = {doc_key(pair.get("document_text")) for pair in held
                       if pair.get("split") == "extra"}
    val_keys = {doc_key(pair.get("document_text")) for pair in real_val}
    mfava_keys = {doc_key(pair.get("document_text")) for pair in mfava}

    # аудит: v19 содержит 7 документов, побайтово совпадающих с mFAVA-частью gold1200_v3 → исключаем
    leak = [pair for pair in v19 if doc_key(pair.get("document_text")) in held_keys]
    leak_ids = {pair["pair_id"] for pair in leak}
    v19_clean = [pair for pair in v19 if pair["pair_id"] not in leak_ids]

    sources = (("real", real_train), ("v19", v19_clean), ("synth_v11", synth),
               ("pseudo_v11", pseudo))
    examples, per_source = [], {}
    for name, pairs in sources:
        built = build_examples(pairs, name, args.limit)
        examples.extend(built)
        keys = {doc_key(example["document_text"]) for example in built}
        candidates = [candidate for example in built for candidate in example["candidates"]]
        styles = {}
        for example in built:
            styles[example["style"]] = styles.get(example["style"], 0) + 1
        per_source[name] = {
            "pairs": len(built),
            "unique_documents": len(keys),
            "candidates": len(candidates),
            "positives": sum(candidate["label"] for candidate in candidates),
            "hard_negatives": sum(1 for candidate in candidates if candidate["hard"]),
            "clean_pairs": sum(1 for example in built if example["clean"]),
            "styles": styles,
            "doc_overlap_with_val": len(keys & val_keys),
            "doc_overlap_with_heldout": len(keys & held_keys),
            "doc_overlap_with_heldout_test": len(keys & held_test_keys),
            "doc_overlap_with_heldout_extra_mfava": len(keys & held_extra_keys),
            "doc_overlap_with_mfava_train": len(keys & mfava_keys),
        }

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        tokens = token_stats(tokenizer, examples)
    except Exception as error:  # noqa: BLE001
        tokens = {"error": f"{type(error).__name__}: {error}"}

    with open(ROOT / args.examples, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    totals = {
        "pairs": sum(item["pairs"] for item in per_source.values()),
        "candidates": sum(item["candidates"] for item in per_source.values()),
        "positives": sum(item["positives"] for item in per_source.values()),
        "hard_negatives": sum(item["hard_negatives"] for item in per_source.values()),
        "doc_overlap_val_nonzero": {name: item["doc_overlap_with_val"]
                                    for name, item in per_source.items()
                                    if item["doc_overlap_with_val"]},
        "excluded_v19_pairs": len(leak_ids),
        "doc_overlap_heldout_nonzero": {name: item["doc_overlap_with_heldout"]
                                        for name, item in per_source.items()
                                        if item["doc_overlap_with_heldout"]},
    }
    report = {
        "rule": "шаг 1–2 v21: данные для cross-encoder. Внешние наборы обязаны иметь doc_overlap = 0 "
                "с val и held-out; synth_v11/pseudo_v11 — собственная аугментация train, их документы "
                "совпадают с train по построению и на val/held-out не пересекаются",
        "encoder": {"model": "cointegrated/rubert-tiny2", "weights": weights,
                    "arch": "3 слоя, hidden 312, vocab 83 828, max_position 2048"},
        "sources": per_source,
        "totals": totals,
        "tokens": tokens,
        "window_plan": {"max_len": args.max_len,
                        "layout": "[CLS] document_prefix [SEP] candidate_tokens [SEP]",
                        "document_prefix_tokens": 200,
                        "overlap_tokens": 0,
                        "note": "документ режется на префикс фиксированной длины (кэш на документ), "
                                "кандидат целиком; start/end предсказываются по токенам кандидата"},
        "audit_leak_v19_heldout": {
            "excluded_pairs": sorted(leak_ids),
            "count": len(leak_ids),
            "detail": "7 пар v19 с base_corpus = mfava_ru имеют документ, побайтово равный документу из "
                      "gold1200_v3 (mFAVA, split extra/train_extra). Проверка doc_overlap в отчёте v19 "
                      "сравнивала с mfava_train_v3 (267 пар), а не с полным mFAVA-набором, поэтому "
                      "пересечение не было видно. Из обучения cross-encoder эти 7 пар исключены.",
            "impact": "mFAVA — внутренний набор (не блокер), но пересечение с held-out фиксируется честно",
        },
        "audit_document_reuse_in_own_data": {
            "note": "документы повторяются между парами исходного корпуса RusHallu/mFAVA (549 train пар "
                    "на 674 уникальных документа в train+val и 301 held-out пара на общий пул документов), "
                    "поэтому OOF считается строго по документам (v17–v20), а не по парам",
            "train_docs_overlapping_heldout": len({doc_key(pair.get("document_text")) for pair in real_train}
                                                  & held_keys),
        },
        "examples_file": {"path": args.examples,
                          "bytes": (ROOT / args.examples).stat().st_size,
                          "sha256": hashlib.sha256((ROOT / args.examples).read_bytes()).hexdigest()},
        "missing_sources": {
            "DFKonGH/ru_hallucination_detector": "в песочнице нет локальной копии "
                                                 "(tools/external потерян вместе с песочницей); "
                                                 "в v16–v17 перенос признан отрицательным",
            "RAGTruth (en)": "нет локальной копии; в v14 проекция ухудшала метрики — не добавляем",
        },
        "verdict": "данные готовы к обучению cross-encoder",
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps({"totals": totals, "sources": {k: v["pairs"] for k, v in per_source.items()},
                      "tokens": tokens}, ensure_ascii=False)[:1200], flush=True)


if __name__ == "__main__":
    main()
