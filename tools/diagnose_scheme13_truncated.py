import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.span_detector import choose_chunks, encode_pair
from tools.train_scheme13_a1 import read_pairs


def main():
    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/rubert-tiny2"), local_files_only=True)
    train, held = read_pairs()
    result = []
    for pair in train + held:
        answer = pair["model_output"]
        full = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        chunks = choose_chunks(tokenizer, pair, top=1, max_length=512)
        encoded = encode_pair(tokenizer, chunks[0]["text"], answer, 512)
        offsets = encoded["answer_tokens"]
        if len(offsets) >= len(full):
            continue
        spans = []
        for span in pair["spans"]:
            left, right = span["start"], span["end"]
            original = [[a, b - 1] for a, b in full if max(a, left) < min(b, right + 1)]
            visible = [[a, b - 1] for _, a, b in offsets if max(a, left) < min(b, right + 1)]
            spans.append({"original_span": [left, right], "original_subword_offsets": original, "encoded_subword_offsets": visible, "original_subwords": len(original), "encoded_subwords": len(visible), "status": "lost" if original and not visible else "partial" if len(visible) < len(original) else "visible"})
        result.append({"pair_id": pair["pair_id"], "answer_subwords_full": len(full), "answer_subwords_encoded": len(offsets), "encoded_end_exclusive": max((b for _, _, b in offsets), default=0), "spans": spans})
    audit = json.loads((ROOT / "artifacts/diagnose_scheme13_before.json").read_text(encoding="utf-8"))
    if len(result) != audit["chunking"]["rag_pairs_with_truncated_answer_subwords"]:
        raise ValueError("выборка длинных ответов не покрыла все обрезанные пары")
    report = {"truncated_pairs": len(result), "partial_spans": sum(span["status"] == "partial" for row in result for span in row["spans"]), "seconds": round(time.perf_counter() - started, 3), "rows": result}
    (ROOT / "artifacts/diagnose_scheme13_truncated.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("truncated_pairs", "partial_spans")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
