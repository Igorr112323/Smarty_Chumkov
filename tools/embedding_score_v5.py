"""Контекстные эмбеддинговые признаки v5 (только train/val/mFAVA-train).

Для каждого ответа считаем hidden states локального RuBERT и для каждого
токена ответа — максимальный cosine similarity к токенам документа. Это
сигнал локальной подтверждённости, отсутствующий в лексических признаках.

Кэш сохраняется в artifacts/embedding_v5_cache.pkl и используется другими
скриптами v5. Held-out не читается.
"""
import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import load_split, read_jsonl
from transformers import AutoModel, AutoTokenizer

MODEL = ROOT / "models/rubert-tiny2"


def load_pairs():
    return (
        load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
        + load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
        + read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    )


def encode(text, tokenizer, model, device):
    if not text.strip():
        return np.zeros((0, 312), dtype=np.float32), []
    encoded = tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=512, return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state[0].cpu().numpy()
    keep = [i for i, (start, end) in enumerate(offsets) if end > start]
    return hidden[keep].astype(np.float32), [offsets[i] for i in keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="artifacts/embedding_v5_cache.pkl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModel.from_pretrained(str(MODEL)).eval()
    device = torch.device("cpu")
    pairs = load_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    cache = {}
    started = time.time()
    for number, pair in enumerate(pairs, 1):
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        aemb, aoff = encode(answer, tokenizer, model, device)
        demb, _ = encode(document, tokenizer, model, device)
        if len(aemb) and len(demb):
            a = aemb / (np.linalg.norm(aemb, axis=1, keepdims=True) + 1e-9)
            d = demb / (np.linalg.norm(demb, axis=1, keepdims=True) + 1e-9)
            sims = (a @ d.T).max(axis=1).astype(np.float32)
        else:
            sims = np.zeros(len(aemb), dtype=np.float32)
        cache[pair["pair_id"]] = {"offsets": aoff, "sim": sims}
        if number % 100 == 0:
            print("encoded", number, "sec", round(time.time() - started, 1), flush=True)
    target = ROOT / args.cache
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        pickle.dump({"pairs": cache, "model_sha": hashlib.sha256(
            (MODEL / "pytorch_model.bin").read_bytes()).hexdigest()}, handle)
    print("saved", target, target.stat().st_size)


if __name__ == "__main__":
    main()
