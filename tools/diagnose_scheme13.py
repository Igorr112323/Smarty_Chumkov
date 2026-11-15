import argparse
import gzip
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, token_positions
from spanver.span_detector import answer_labels, choose_chunks, encode_pair
from tools.preflight_13 import digest, our_rows
from tools.train_scheme13_a1 import read_pairs

SEED = 28


def encode_with_offsets(tokenizer, pair):
    chunks = choose_chunks(tokenizer, pair, top=1, max_length=512)
    return encode_pair(tokenizer, chunks[0]["text"] if chunks else "", pair["model_output"], 512)


def compare_pair(tokenizer, pair, lexfeat):
    answer = pair["model_output"]
    row = lexfeat.build_row(tokenizer, pair)
    encoded = encode_with_offsets(tokenizer, pair)
    if row is None or encoded is None or row["input_ids"] != encoded["input_ids"]:
        raise ValueError(f"несовпадение токенизации {pair['pair_id']}")
    words = token_positions(answer)
    gold = covered_tokens(answer, [(span["start"], span["end"]) for span in pair["spans"]])
    positive = set()
    visible = set()
    for index, start, end in encoded["answer_tokens"]:
        for word_index, (left, right) in enumerate(words):
            if left >= end:
                break
            if start <= right and end > left:
                visible.add(word_index)
                if row["labels"][index] == 1:
                    positive.add(word_index)
    mismatch = positive.symmetric_difference(gold & visible)
    reference = answer_labels(encoded, pair["spans"])
    subword_diffs = []
    for index, start, end in encoded["answer_tokens"]:
        if row["labels"][index] != reference[index]:
            subword_diffs.append({"start": start, "end_exclusive": end, "text": answer[start:end], "train": row["labels"][index], "strict_overlap": reference[index]})
    differences = [{"start": words[i][0], "end_inclusive": words[i][1], "text": answer[words[i][0]:words[i][1] + 1], "train_projection": int(i in positive), "eval": int(i in gold)} for i in sorted(mismatch)]
    return {
        "pair_id": pair["pair_id"], "train_subwords": sum(v != -100 for v in row["labels"]),
        "train_positive_subwords": sum(v == 1 for v in row["labels"]),
        "eval_words_total": len(words), "eval_positive_words_total": len(gold),
        "visible_words": len(visible), "unrepresented_words": len(words) - len(visible),
        "train_positive_visible_words": len(positive), "eval_positive_visible_words": len(gold & visible),
        "disagreements": len(mismatch), "disagreement_fraction": len(mismatch) / len(visible) if visible else 0.0,
        "matched_positive_positions": [{"start": words[i][0], "end_inclusive": words[i][1], "text": answer[words[i][0]:words[i][1] + 1], "train_projection": int(i in positive), "eval": int(i in gold)} for i in sorted(positive & gold)[:3]],
        "differences": differences[:10], "strict_subword_disagreements": len(subword_diffs), "strict_subword_examples": subword_diffs[:5],
    }


def loss_probe(torch, tokenizer, model, training, lexfeat):
    from tools.train_scheme13_full import loss_batch
    order = list(range(len(training)))
    random.Random(2).shuffle(order)
    pairs = [training[i] for i in order[:16]]
    rows = [lexfeat.build_row(tokenizer, pair) for pair in pairs]
    counts = {"zero": sum(label == 0 for row in rows for label in row["labels"]), "one": sum(label == 1 for row in rows for label in row["labels"])}

    class Constant(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.logit = math.log(p / (1.0 - p))

        def forward(self, input_ids, attention_mask, features):
            shape = (*input_ids.shape, 2)
            logits = torch.zeros(shape, dtype=torch.float)
            logits[:, :, 1] = self.logit
            return logits

    result = {}
    with torch.no_grad():
        for name, candidate in (("trained_e2", model), ("all_positive_p099", Constant(0.99)), ("all_negative_p001", Constant(0.01)), ("all_positive_p026", Constant(0.26))):
            token_losses, pair_losses = [], []
            probs = {0: [], 1: []}
            for start in range(0, len(rows), 4):
                batch = lexfeat.collate(rows[start:start + 4], tokenizer.pad_token_id)
                token_loss, pair_loss = loss_batch(torch, candidate, batch)
                token_losses.append(float(token_loss))
                pair_losses.append(float(pair_loss))
                logits = candidate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], features=batch["features"])
                p = torch.softmax(logits, dim=-1)[:, :, 1]
                for cls in (0, 1):
                    probs[cls].extend(p[batch["labels"] == cls].tolist())
            summary = {}
            for cls in (0, 1):
                values = probs[cls]
                summary[str(cls)] = {"mean_probability": sum(values) / len(values) if values else None, "fraction_ge_025": sum(v >= 0.25 for v in values) / len(values) if values else None, "fraction_ge_050": sum(v >= 0.5 for v in values) / len(values) if values else None}
            result[name] = {"focal_token_loss": sum(token_losses) / len(token_losses), "pair_loss": sum(pair_losses) / len(pair_losses), "total_loss": (sum(token_losses) + 0.3 * sum(pair_losses)) / len(token_losses), "probabilities_by_true_class": summary}
    return {"pair_ids": [pair["pair_id"] for pair in pairs], "labels": counts, "counterfactual_probability_note": "все единицы/нули заданы конечными вероятностями 0,99/0,01; loss зависит от уверенности", "values": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-50-only", action="store_true")
    parser.add_argument("--output", default="artifacts/diagnose_scheme13_before.json")
    args = parser.parse_args()
    import torch
    from spanver import lexfeat

    started = time.perf_counter()
    training, heldout = read_pairs()
    rag = training + heldout
    if len(training) != 6042 or len(heldout) != 1458:
        raise ValueError("разбиение не совпадает")
    torch.set_num_threads(2)
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
    model.base.config._attn_implementation = "eager"
    torch.set_num_threads(2)
    sample = random.Random(SEED).sample(rag, 50)
    audit = [compare_pair(tokenizer, pair, lexfeat) for pair in sample]
    report = {"seed": SEED, "audit_method": "метки обучения проецируются по пересечению subword на токены метрики; сравниваются только видимые слова; отсутствующие слова учтены отдельно", "audit_50": audit, "audit_totals": {key: sum(item[key] for item in audit) for key in ("train_positive_subwords", "eval_positive_words_total", "visible_words", "unrepresented_words", "train_positive_visible_words", "eval_positive_visible_words", "disagreements", "strict_subword_disagreements")}}
    if args.audit_50_only:
        report["seconds"] = round(time.perf_counter() - started, 3)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report["audit_totals"], ensure_ascii=False), flush=True)
        return
    all_sets = (("ragtruth_all", rag), ("ours_699", our_rows()))
    statistics_out = {}
    negs = [row for row in training if not row["spans"]]
    negative_ids = {row["pair_id"] for row in negs}
    negative_sample = {row["pair_id"] for row in random.Random(SEED).sample(negs, 20)}
    negative_results = []
    negative_nonzero = []
    long_examples = []
    answer_token_lengths = []
    truncated_pairs = 0
    spans_total = invisible = partial = no_subwords = 0
    for name, pairs in all_sets:
        sub_tokens = sub_positive = word_tokens = word_positive = 0
        for i, pair in enumerate(pairs, 1):
            row = lexfeat.build_row(tokenizer, pair)
            if row is None:
                raise ValueError(f"пустая строка {pair['pair_id']}")
            positives = sum(label == 1 for label in row["labels"])
            sub_positive += positives
            sub_tokens += sum(label != -100 for label in row["labels"])
            answer = pair["model_output"]
            words = token_positions(answer)
            gold = covered_tokens(answer, [(span["start"], span["end"]) for span in pair["spans"]])
            word_tokens += len(words)
            word_positive += len(gold)
            if name == "ragtruth_all":
                if pair["pair_id"] in negative_sample:
                    negative_results.append({"pair_id": pair["pair_id"], "positive_subwords": positives})
                if pair["pair_id"] in negative_ids and positives:
                    negative_nonzero.append(pair["pair_id"])
                full_offsets = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                answer_token_lengths.append(len(full_offsets))
                if sum(row["answer_mask"]) < len(full_offsets):
                    truncated_pairs += 1
                long_candidate = len(long_examples) < 20 or len(full_offsets) > long_examples[-1]["answer_subwords_full"]
                if long_candidate or pair["spans"]:
                    encoded = encode_with_offsets(tokenizer, pair)
                    offsets = encoded["answer_tokens"]
                    max_end = max((end for _, _, end in offsets), default=0)
                    info = []
                    for span in pair["spans"]:
                        spans_total += 1
                        start, end = span["start"], span["end"]
                        visible = [(left, right - 1) for _, left, right in offsets if min(right, end + 1) > max(left, start)]
                        original = [(left, right - 1) for left, right in full_offsets if min(right, end + 1) > max(left, start)]
                        absent = len(visible) == 0 and bool(original)
                        partly = 0 < len(visible) < len(original)
                        invisible += absent
                        partial += partly
                        no_subwords += not original
                        info.append({"original_span": [start, end], "visible_positive_subword_offsets": visible, "original_subword_count": len(original), "visible_subword_count": len(visible), "status": "no_subwords" if not original else "lost" if absent else "partial" if partly else "visible"})
                    if long_candidate:
                        long_examples.append({"pair_id": pair["pair_id"], "answer_chars": len(answer), "answer_subwords_full": len(full_offsets), "answer_subwords_encoded": len(offsets), "visible_chars_end_exclusive": max_end, "spans": info})
                        long_examples.sort(key=lambda item: -item["answer_subwords_full"])
                        del long_examples[20:]
            if i % 1000 == 0:
                print(f"{name}: {i}/{len(pairs)}", flush=True)
        statistics_out[name] = {"pairs": len(pairs), "train_subwords": sub_tokens, "train_positive_subwords": sub_positive, "train_positive_per_pair": sub_positive / len(pairs), "train_positive_fraction": sub_positive / sub_tokens, "eval_whitespace_tokens": word_tokens, "eval_positive_tokens": word_positive, "eval_positive_per_pair": word_positive / len(pairs), "eval_positive_fraction": word_positive / word_tokens}
    report["token_stats"] = statistics_out
    report["negatives"] = {"train_negative_count": len(negs), "negatives_with_positive_labels": len(negative_nonzero), "examples_20": sorted(negative_results, key=lambda row: row["pair_id"])}
    report["chunking"] = {"implementation": "документ разбивается на чанки; ответ не разбивается на окна, longest_first обрезает пару до 512 токенов", "rag_pairs_checked": len(rag), "rag_pairs_with_answer_over_512_subwords": sum(n > 512 for n in answer_token_lengths), "answer_subword_median": statistics.median(answer_token_lengths), "answer_subword_max": max(answer_token_lengths), "rag_pairs_with_truncated_answer_subwords": truncated_pairs, "positive_spans_checked": spans_total, "positive_spans_fully_unrepresented": invisible, "positive_spans_partially_represented": partial, "positive_spans_without_subwords": no_subwords, "long_examples_note": "20 самых длинных ответов; ответов длиннее 512 subword может быть меньше 20", "long_examples": long_examples}
    from tools.train_scheme13_full import load_state
    folder = ROOT / "artifacts/checkpoints/stage_a_full_e2"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    state = load_state(folder, manifest["training_state_parts"], torch)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.eval()
    report["loss_batch"] = loss_probe(torch, tokenizer, model, training, lexfeat)
    report["checkpoint_sha256"] = manifest["fp16"]["sha256"]
    report["seconds"] = round(time.perf_counter() - started, 3)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": report["audit_totals"], "token_stats": statistics_out, "negatives": report["negatives"], "chunking": {k: v for k, v in report["chunking"].items() if k != "long_examples"}, "loss_batch": report["loss_batch"]["values"], "seconds": report["seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
