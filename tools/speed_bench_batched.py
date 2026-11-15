import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, lexfeat, module
from spanver.evaluate import read_jsonl
from spanver.snap import snapped

LENGTHS = (200, 1000, 5000, 20000)
DOCS_PER_LENGTH = 5
TOKEN_THRESHOLD = 0.25
PAIR_GATE = 0.45
MIN_CHARS = 2
BATCH = 16


def build_pairs(gold, length):
    chosen = []
    for pair in gold:
        document = pair.get("document_text") or ""
        if len(document) >= length and pair.get("model_output"):
            chosen.append({"pair_id": pair.get("pair_id"), "model_output": pair.get("model_output"), "document_text": document[:length]})
        if len(chosen) >= DOCS_PER_LENGTH:
            return chosen
    pool = [pair for pair in gold if pair.get("model_output") and pair.get("document_text")]
    made = []
    index = 0
    while len(made) < DOCS_PER_LENGTH:
        base = pool[index % len(pool)]
        text = base.get("document_text") or ""
        repeated = (text + " ") * ((length // max(1, len(text))) + 1)
        made.append({"pair_id": f"ext-{length}-{index}", "model_output": base.get("model_output"), "document_text": repeated[:length]})
        index += 1
    return made


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/cv/feat5/fold_1")
    parser.add_argument("--source", default="models/rubert-tiny2")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--report", default="artifacts/speed_report_batched.json")
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args(argv)

    gold = read_jsonl(Path(args.gold))
    tokenizer, model = lexfeat.load_saved(args.checkpoint, args.source)
    rows = []
    for length in LENGTHS:
        pairs = build_pairs(gold, length)
        clock = time.perf_counter()
        cache = lexfeat.batched_probabilities(tokenizer, model, pairs, top=4, batch=args.batch)
        model_seconds = time.perf_counter() - clock
        clock = time.perf_counter()
        lexical = {}
        for pair in pairs:
            answer = pair.get("model_output") or ""
            lexical[pair.get("pair_id")] = snapped(module.verify(answer, (pair.get("document_text") or "")[:length])["spans"], answer, "expand")
        lexical_seconds = time.perf_counter() - clock
        for pair in pairs:
            cached = cache.get(pair.get("pair_id"))
            answer = pair.get("model_output") or ""
            spans = []
            if cached:
                top = max((value for _, _, value in cached["pieces"]), default=0.0)
                if top >= PAIR_GATE:
                    flags = decode.flags_for(cached["pieces"], TOKEN_THRESHOLD, None)
                    spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, MIN_CHARS)
                snapped(spans, answer, "expand")
        docs = len(pairs)
        chars = sum(len((pair.get("document_text") or "")[:length]) for pair in pairs)
        answer_chars = sum(len(pair.get("model_output") or "") for pair in pairs)
        rows.append(
            {
                "length": length,
                "documents": docs,
                "doc_chars": chars,
                "answer_chars": answer_chars,
                "batch": args.batch,
                "model_ms_per_1000_doc": round(model_seconds * 1000.0 / (chars / 1000.0), 2),
                "ensemble_ms_per_1000_doc": round((model_seconds + lexical_seconds) * 1000.0 / (chars / 1000.0), 2),
                "model_ms_per_pair": round(model_seconds * 1000.0 / docs, 2),
                "ensemble_ms_per_pair": round((model_seconds + lexical_seconds) * 1000.0 / docs, 2),
                "model_ms_per_1000_answer": round(model_seconds * 1000.0 / max(1.0, answer_chars / 1000.0), 2),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"checkpoint": args.checkpoint, "batch": args.batch, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
