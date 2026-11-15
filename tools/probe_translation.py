import argparse
import difflib
import json
import random
import statistics
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.ragtruth import chunks_from, context_text, load_jsonl, select_pairs, spans_from_labels
from tools.translate_ragtruth import MultiTranslator, NetworkBlocked, split_parts


def sample_chunks(rows, sources, count, seed):
    positives, _ = select_pairs(rows, 6000, 1500, seed)
    rng = random.Random(seed)
    rng.shuffle(positives)
    chunks = []
    documents = []
    for row, spans in positives:
        for chunk in chunks_from(row.get("response") or "", spans):
            if chunk["text"].strip():
                chunks.append({"kind": chunk["kind"], "text": chunk["text"], "pair_id": row.get("id")})
        source = sources.get(str(row.get("source_id"))) or {}
        text = context_text(source)
        if text.strip():
            documents.append({"kind": "document", "text": text[:4000], "pair_id": row.get("id")})
        if len(chunks) >= count * 2 and len(documents) >= count:
            break
    rng.shuffle(chunks)
    rng.shuffle(documents)
    mixed = (chunks + documents)[:count]
    rng.shuffle(mixed)
    return mixed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", default="Dataset/RAGTruth_en/response.jsonl")
    parser.add_argument("--sources", default="Dataset/RAGTruth_en/source_info.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--reverse", type=int, default=20)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default="artifacts/translate_probe.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows = load_jsonl(Path(args.responses))
    sources = {str(item.get("source_id")): item for item in load_jsonl(Path(args.sources))}
    samples = sample_chunks(rows, sources, args.count, args.seed)
    translator = MultiTranslator()
    report = {"count": len(samples), "channel_order": list(translator.errors), "translated": 0, "failed": 0, "parts": 0}
    translated = []
    failures = []
    for item in samples:
        report["parts"] += len(split_parts(item["text"]))
        try:
            value = translator.translate(item["text"])
            report["translated"] += 1
            translated.append({"text": item["text"], "translation": value, "kind": item["kind"], "pair_id": item["pair_id"]})
        except NetworkBlocked as error:
            report["failed"] += 1
            failures.append(str(error))
            if report["failed"] >= 3 and report["translated"] == 0:
                break
    report["channel_status"] = translator.status()
    report["failure_samples"] = sorted(set(failures))[:5]
    report["elapsed_s"] = round(time.perf_counter() - started, 1)

    if translated:
        ratios = [len(item["translation"]) / max(1, len(item["text"])) for item in translated]
        report["length_ratio_median"] = round(statistics.median(ratios), 4)
        report["length_ratio_min"] = round(min(ratios), 4)
        report["length_ratio_max"] = round(max(ratios), 4)
        reverse_scores = []
        for item in translated[: args.reverse]:
            try:
                back = translator.translate(item["translation"], "ru", "en")
            except NetworkBlocked:
                break
            reverse_scores.append(difflib.SequenceMatcher(None, item["text"], back).ratio())
        if reverse_scores:
            report["reverse_similarity_median"] = round(statistics.median(reverse_scores), 4)
            report["reverse_samples"] = len(reverse_scores)
        report["examples"] = [
            {"source": item["text"][:300], "translation": item["translation"][:300], "kind": item["kind"]}
            for item in translated[: args.examples]
        ]
    else:
        report["status"] = "ни один канал не ответил"
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
