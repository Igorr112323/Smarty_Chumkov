import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode, module
from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from spanver.splits import load_splits
from spanver.span_detector import load_model, pair_probabilities

LENGTHS = (200, 1000, 5000, 20000)
DOCS_PER_LENGTH = 5
TOKEN_THRESHOLD = 0.25
PAIR_GATE = 0.45
MIN_CHARS = 2


def build_documents(gold, length):
    texts = []
    for pair in gold:
        document = pair.get("document_text") or ""
        if len(document) >= length:
            texts.append((document[:length], pair.get("model_output") or "", pair.get("document_text")))
        if len(texts) >= DOCS_PER_LENGTH:
            break
    if len(texts) < DOCS_PER_LENGTH:
        pool = [pair.get("document_text") or "" for pair in gold if pair.get("document_text")]
        joined = ""
        index = 0
        while len(texts) < DOCS_PER_LENGTH:
            while len(joined) < length:
                joined += pool[index % len(pool)]
                index += 1
            answer = gold[index % len(gold)].get("model_output") or ""
            texts.append((joined[:length], answer, joined[:length]))
            joined = joined[length:]
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/cv/base/fold_1")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--report", default="artifacts/speed_report.json")
    args = parser.parse_args(argv)

    gold = read_jsonl(Path(args.gold))
    torch, tokenizer, model = load_model(args.checkpoint)
    rows = []
    for length in LENGTHS:
        documents = build_documents(gold, length)
        model_seconds = 0.0
        lexical_seconds = 0.0
        answers = 0
        for document, answer, _ in documents:
            piece_clock = time.perf_counter()
            cached = pair_probabilities(tokenizer, model, torch, {"model_output": answer, "document_text": document}, top=4)
            model_seconds += time.perf_counter() - piece_clock
            lexical_clock = time.perf_counter()
            spans = module.verify(answer, document)["spans"]
            lexical_seconds += time.perf_counter() - lexical_clock
            tops = max((value for _, _, value in cached["pieces"]), default=0.0)
            if tops >= PAIR_GATE:
                flags = decode.flags_for(cached["pieces"], TOKEN_THRESHOLD, None)
                model_spans = decode.spans_from_flags(cached["answer"], cached["pieces"], flags, MIN_CHARS)
            else:
                model_spans = []
            model_spans = snapped(model_spans, answer, "expand")
            answers += len(answer)
        docs = len(documents)
        chars = length * docs
        rows.append(
            {
                "length": length,
                "documents": docs,
                "answer_chars": answers,
                "model_ms_per_1000_doc": round(model_seconds * 1000.0 / (chars / 1000.0), 2),
                "lexical_ms_per_1000_doc": round(lexical_seconds * 1000.0 / (chars / 1000.0), 2),
                "ensemble_ms_per_1000_doc": round((model_seconds + lexical_seconds) * 1000.0 / (chars / 1000.0), 2),
                "model_ms_per_1000_answer": round(model_seconds * 1000.0 / max(1.0, answers / 1000.0), 2),
                "ensemble_ms_per_1000_answer": round((model_seconds + lexical_seconds) * 1000.0 / max(1.0, answers / 1000.0), 2),
                "model_ms_per_pair": round(model_seconds * 1000.0 / docs, 2),
                "ensemble_ms_per_pair": round((model_seconds + lexical_seconds) * 1000.0 / docs, 2),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"checkpoint": args.checkpoint, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
