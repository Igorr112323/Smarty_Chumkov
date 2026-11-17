import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.span_detector import answer_labels, choose_chunks, encode_pair
from tools.preflight_13 import digest, id_digest
from tools.train_scheme13_a1 import read_pairs


def encode(tokenizer, pair):
    chunks = choose_chunks(tokenizer, pair, top=1, max_length=512)
    return encode_pair(tokenizer, chunks[0]["text"] if chunks else "", pair["model_output"], 512)


def audit(tokenizer, name, pairs):
    total = positive = mismatch = invalid = 0
    differences = []
    spans = 0
    for index, pair in enumerate(pairs, 1):
        full = tokenizer(pair["model_output"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        encoded = encode(tokenizer, pair)
        actual = [(start, end) for _, start, end in encoded["answer_tokens"]]
        if actual != [tuple(offset) for offset in full]:
            mismatch += 1
            if len(differences) < 10:
                differences.append({"pair_id": pair["pair_id"], "full": len(full), "encoded": len(actual)})
        labels = answer_labels(encoded, pair["spans"])
        total += sum(label != -100 for label in labels)
        positive += sum(label == 1 for label in labels)
        for span in pair["spans"]:
            spans += 1
            if not 0 <= span["start"] <= span["end"] < len(pair["model_output"]):
                invalid += 1
        if index % 1000 == 0:
            print(f"{name}: {index}/{len(pairs)}", flush=True)
    return {"pairs": len(pairs), "pair_ids_sha256": id_digest(pairs), "positive_spans": spans, "invalid_spans": invalid, "answers_with_missing_subwords": mismatch, "examples": differences, "positive_subwords": positive, "negative_subwords": total - positive, "positive_fraction": positive / total, "negative_to_positive_ratio": (total - positive) / positive}


def control(torch, tokenizer, training):
    from spanver import lexfeat
    from tools.train_scheme13_full import load_state

    folder = ROOT / "artifacts/checkpoints/stage_a_full_e2"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
    model.base.config._attn_implementation = "eager"
    torch.set_num_threads(2)
    state = load_state(folder, manifest["training_state_parts"], torch)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.eval()
    order = list(range(len(training)))
    random.Random(2).shuffle(order)
    pairs = [training[index] for index in order[:16]]
    rows = [lexfeat.build_row(tokenizer, pair) for pair in pairs]
    counts = {"positive": sum(label == 1 for row in rows for label in row["labels"]), "negative": sum(label == 0 for row in rows for label in row["labels"])}
    weight = torch.tensor([11.0])
    total = {"model": 0.0, "constant_near_zero": 0.0, "constant_099": 0.0, "constant_05": 0.0}
    probabilities = {"constant_near_zero": 1e-9, "constant_099": 0.99, "constant_05": 0.5}
    count = 0
    with torch.no_grad():
        for start in range(0, len(rows), 4):
            batch = lexfeat.collate(rows[start:start + 4], tokenizer.pad_token_id)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], features=batch["features"])
            valid = batch["labels"] != -100
            targets = batch["labels"][valid].to(torch.float)
            scores = (logits[:, :, 1] - logits[:, :, 0])[valid]
            total["model"] += float(torch.nn.functional.binary_cross_entropy_with_logits(scores, targets, pos_weight=weight, reduction="sum"))
            for name, p in probabilities.items():
                constant = torch.full_like(scores, math.log(p / (1.0 - p)))
                total[name] += float(torch.nn.functional.binary_cross_entropy_with_logits(constant, targets, pos_weight=weight, reduction="sum"))
            count += len(targets)
    means = {name: value / count for name, value in total.items()}
    return {"model": "stage_a_full_e2, focal; веса fp32", "batch_pair_ids": [pair["pair_id"] for pair in pairs], "labels": counts, "pos_weight": 11, "loss": means, "constant_zero_exact_loss": "infinity при наличии положительных меток", "near_zero_probability": 1e-9, "strictly_below_all": all(means["model"] < means[name] for name in probabilities), "model_checkpoint_fp16_sha256": manifest["fp16"]["sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "control"), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/rubert-tiny2"), local_files_only=True)
    training, heldout = read_pairs()
    preflight = json.loads((ROOT / "artifacts/preflight_13.json").read_text(encoding="utf-8"))
    if id_digest(training) != preflight["rag_train"]["id_sha256"] or id_digest(heldout) != preflight["rag_holdout"]["id_sha256"]:
        raise ValueError("разбиение не соответствует preflight")
    report = {"mode": args.mode, "code_sha256": digest(Path(__file__)), "encoder_code_sha256": digest(ROOT / "spanver/span_detector.py")}
    if args.mode == "audit":
        report["train"] = audit(tokenizer, "train", training)
        report["holdout"] = audit(tokenizer, "holdout", heldout)
    else:
        report["control"] = control(torch, tokenizer, training)
    report["seconds"] = round(time.perf_counter() - started, 3)
    path = ROOT / f"artifacts/bce29_{args.mode}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "control" and not report["control"]["strictly_below_all"]:
        return 2
    if args.mode == "audit" and any(report[part]["answers_with_missing_subwords"] or report[part]["invalid_spans"] for part in ("train", "holdout")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
