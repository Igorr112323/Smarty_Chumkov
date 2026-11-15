import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, token_positions
from spanver.span_detector import answer_labels, choose_chunks, encode_pair
from tools.train_scheme13_a1 import read_pairs


def main():
    import torch
    from spanver import lexfeat

    started = time.perf_counter()
    torch.set_num_threads(2)
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
    del model
    torch.set_num_threads(2)
    train, held = read_pairs()
    pairs = train + held
    result = {"pairs": len(pairs), "old_subword_differences": 0, "fixed_subword_differences": 0, "subwords_in_positive_pairs_checked": 0, "pairs_with_old_subword_differences": 0, "old_metric_word_differences": 0, "fixed_metric_word_differences": 0, "pairs_with_old_metric_differences": 0, "old_subword_examples": [], "old_metric_examples": []}
    for number, pair in enumerate(pairs, 1):
        text = pair["model_output"]
        gold = pair["spans"]
        words = token_positions(text)
        old = {index for index, (start, end) in enumerate(words) if any(span["start"] <= start <= span["end"] or span["start"] <= end <= span["end"] for span in gold)}
        correct = {index for index, (start, end) in enumerate(words) if any(max(start, span["start"]) <= min(end, span["end"]) for span in gold)}
        found = covered_tokens(text, [(span["start"], span["end"]) for span in gold])
        differences = old.symmetric_difference(correct)
        result["old_metric_word_differences"] += len(differences)
        result["fixed_metric_word_differences"] += len(found.symmetric_difference(correct))
        result["pairs_with_old_metric_differences"] += bool(differences)
        if differences and len(result["old_metric_examples"]) < 10:
            index = min(differences)
            left, right = words[index]
            result["old_metric_examples"].append({"pair_id": pair["pair_id"], "word_inclusive": [left, right], "text": text[left:right + 1], "spans": [[span["start"], span["end"]] for span in gold if max(left, span["start"]) <= min(right, span["end"])], "old": int(index in old), "fixed": int(index in found)})
        if not gold:
            continue
        row = lexfeat.build_row(tokenizer, pair)
        chunks = choose_chunks(tokenizer, pair, top=1, max_length=512)
        encoded = encode_pair(tokenizer, chunks[0]["text"], text, 512)
        reference = answer_labels(encoded, gold)
        differences = []
        for index, start, end in encoded["answer_tokens"]:
            previous = int(any(min(end, span["end"]) - max(start, span["start"]) >= 0 for span in gold))
            if previous != reference[index]:
                differences.append({"subword_half_open": [start, end], "text": text[start:end], "old": previous, "fixed": row["labels"][index]})
            result["fixed_subword_differences"] += row["labels"][index] != reference[index]
        result["old_subword_differences"] += len(differences)
        result["pairs_with_old_subword_differences"] += bool(differences)
        result["subwords_in_positive_pairs_checked"] += len(encoded["answer_tokens"])
        if differences and len(result["old_subword_examples"]) < 10:
            result["old_subword_examples"].append({"pair_id": pair["pair_id"], "spans": [[span["start"], span["end"]] for span in gold], "differences": differences[:5]})
        if number % 2000 == 0:
            print(f"проверено {number}/{len(pairs)}", flush=True)
    result["seconds"] = round(time.perf_counter() - started, 3)
    (ROOT / "artifacts/diagnose_scheme13_boundaries.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
