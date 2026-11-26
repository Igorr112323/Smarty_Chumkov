"""Оценка чекпоинта v9 на полном множестве кандидатов (dev только)."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from oracle_recall_v4 import candidates_v4
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from train_neural_pointer_v9 import Pointer, doc_chunks, evidence_for, cell

CKPT = ROOT / "artifacts/neural_pointer_v9/best"


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT), local_files_only=True)
    base = AutoModel.from_pretrained(str(CKPT), local_files_only=True)
    model = Pointer(base).to(device)
    state = torch.load(str(CKPT / "head.pt"), map_location=device)
    model.head.load_state_dict(state["head"])
    model.eval()
    return tokenizer, model


def score_pairs(pairs, tokenizer, model, device, max_len=160, batch=48):
    meta = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        chunks = doc_chunks(document)
        for start, end, text in candidates_v4(answer):
            fragment = (text or "")[:320]
            meta.append((pair["pair_id"], start, end, fragment,
                         evidence_for(document, fragment, chunks=chunks)))
    print("scored candidates", len(meta), flush=True)
    rows = []
    started = time.time()
    for offset in range(0, len(meta), 1000):
        chunk = meta[offset:offset + 1000]
        encoded = tokenizer(
            [(item[3], item[4]) for item in chunk],
            truncation=True, max_length=max_len, padding=False,
        )
        for item, index in zip(chunk, range(len(chunk))):
            rows.append((item[0], item[1], item[2], {
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
                "token_type_ids": encoded["token_type_ids"][index],
            }))
        print("tokenized", offset + len(chunk), "sec", round(time.time() - started, 1), flush=True)
    result = {pair["pair_id"]: [] for pair in pairs}
    with torch.no_grad():
        for offset in range(0, len(rows), batch):
            chunk = rows[offset:offset + batch]
            size = max(len(item[3]["input_ids"]) for item in chunk)
            tensors = {}
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                tensors[key] = torch.tensor(
                    [item[3][key] + [tokenizer.pad_token_id] * (size - len(item[3][key])) for item in chunk],
                    dtype=torch.long, device=device,
                )
            logits = model(**tensors)
            for item, value in zip(chunk, logits.cpu().numpy()):
                result[item[0]].append((item[1], item[2], float(1 / (1 + math.exp(-float(value))))))
            if offset and offset % (batch * 100) == 0:
                print(offset, "sec", round(time.time() - started, 1), flush=True)
    return result


def decode(pair, scored, threshold, max_spans, nms_iou):
    candidates = sorted(scored.get(pair["pair_id"], []), key=lambda row: -row[2])
    chosen = []
    for start, end, risk in candidates:
        if risk < threshold or len(chosen) >= max_spans:
            break
        if any(iou(start, end, other[0], other[1]) > nms_iou for other in chosen):
            continue
        chosen.append((start, end))
    return [{"start": a, "end": b, "risk": 1.0,
             "type": "unsupported_addition", "confidence": 1.0} for a, b in chosen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/neural_pointer_v9/eval_full_dev.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cpu")
    tokenizer, model = load_model(device)
    val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    dev = val + mfava
    scored = score_pairs(dev, tokenizer, model, device)
    rows = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99):
        for max_spans in (1, 2, 3, 5):
            def predictor(pair):
                return decode(pair, scored, threshold, max_spans, 0.10)
            result = metrics(dev, predictor)
            rows.append({"threshold": threshold, "max_spans": max_spans,
                         "dev": cell(result), "val": cell(metrics(val, predictor))})
    rows.sort(key=lambda row: (row["dev"]["span"], row["dev"]["token"]), reverse=True)
    passing = [row for row in rows if row["dev"]["span"] >= 0.5 and row["dev"]["token"] >= 0.5
               and row["dev"]["pair"] >= 0.5 and row["dev"]["fpr"] <= 0.4]
    report = {
        "rule": "full candidates_v4; val+mFAVA dev only; held-out не читается",
        "ckpt_sha256": hashlib.sha256((CKPT / "model.safetensors").read_bytes()).hexdigest(),
        "best": rows[:10], "passing": passing[:10], "configs": rows,
    }
    target = ROOT / args.output
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("best", rows[0], "passing", len(passing), flush=True)


if __name__ == "__main__":
    main()
