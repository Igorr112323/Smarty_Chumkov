"""V9: документно-обусловленный start/end pointer на rubert-tiny2.

Принципиальные отличия от v6 QA-pointer и v7 синтетики:
  * обучение на RAGTruth_ru (6000 фразовых hallucination span + 150 mFAVA +
    549 RusHallu train) с точными автоматическими offsets;
  * на вход подается не весь DOC_LIMIT, а релевантный evidence-фрагмент,
    выбранный по лексическому совпадению с кандидатом;
  * hard negatives — кандидаты с высоким baseline score, а не случайные;
  *pair gate не замораживается: порог калибруется только на val+mFAVA.

Ни held-out, ни test301 в обучении и подборе не участвуют.
"""
import argparse
import collections
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from oracle_recall_v4 import candidates_v4
from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans, document_context

MODEL_DIR = ROOT / "models/rubert-tiny2"
OUT = ROOT / "artifacts/neural_pointer_v9"
MAX_LEN = 160
CAND_PER_PAIR = 12
BATCH = 16
EPOCHS = 2
LR = 2e-5
SEED = 9

ranker._WEIGHTS = json.loads((ROOT / "spanver/ranker_weights.json").read_text(encoding="utf-8"))
TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)


class Pointer(nn.Module):
    """Скоринг пары (start,end): CLS-представление фрагмента и документа."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.base(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        doc_mask = (token_type_ids * attention_mask).unsqueeze(-1)
        candidate = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        evidence = (hidden * doc_mask).sum(1) / doc_mask.sum(1).clamp(min=1)
        return self.head(torch.cat((candidate, evidence), dim=-1)).squeeze(-1)


def doc_chunks(document, limit=520):
    doc = document or ""
    return [doc[start:start + limit].lower() for start in range(0, max(1, len(doc)), 240)]


def evidence_for(document, candidate_text, limit=520, chunks=None):
    """Дешёвый выбор релевантного фрагмента документа (SET substring lookup)."""
    if chunks is None:
        chunks = doc_chunks(document, limit)
    if not chunks:
        return ""
    words = {w.lower() for w in candidate_text.split() if len(w) > 2}
    if not words:
        return chunks[0][:limit]
    best, best_score = chunks[0], -1.0
    for chunk in chunks:
        score = sum(1 for word in words if word in chunk)
        if score > best_score:
            best, best_score = chunk, score
    return best


def select_candidates(pair, rng=None, gold_aware=True):
    """Короткий лист кандидатов: дешёвый support → baseline logit → top-K.

    На inference gold_aware=False: gold-кандидаты не добавляются.
    """
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    doc_tokens, _stems, _grams = document_context(document)
    gold = pair.get("spans") or []
    pool = candidates_v4(answer)
    base_spans = list(candidate_spans(answer))
    shortlist = {}
    for start, end, text in pool:
        words = [w.lower() for w in text.split() if len(w) > 2]
        cheap = sum(w in doc_tokens for w in words[:20]) / max(1, len(words[:20]))
        shortlist[(start, end)] = (cheap, text)
    for start, end, text in base_spans:
        if (start, end) not in shortlist:
            shortlist[(start, end)] = (1.0, text)
    ranked = sorted(shortlist.items(), key=lambda row: -row[1][0])[:48]
    chosen_spans = {(key[0], key[1]) for key, _ in ranked}
    if gold_aware:
        for span in gold:
            for key, (_cheap, text) in shortlist.items():
                if iou(key[0], key[1], span["start"], span["end"]) >= 0.5:
                    chosen_spans.add(key)
    scored = []
    for start, end in chosen_spans:
        text = answer[start:end + 1]
        x = ranker.features(text, start, end, answer, document, doc_tokens, _grams)
        score = ranker._logit(x, ranker._WEIGHTS["cand"], ranker.FEATURE_COUNT)
        hit = gold_aware and any(iou(start, end, g["start"], g["end"]) >= 0.5 for g in gold)
        scored.append((score, start, end, text, hit))
    scored.sort(key=lambda row: -row[0])
    result = scored[:CAND_PER_PAIR]
    if rng and gold_aware:
        rng.shuffle(result)
    return result


def tokenize(pair, candidate, chunks=None):
    _score, start, end, text, label = candidate
    candidate_text = (text or "")[:320]
    evidence = evidence_for(pair.get("document_text") or "", candidate_text, chunks=chunks)
    return TOKENIZER(
        candidate_text, evidence, truncation=True, max_length=MAX_LEN,
        padding=False, return_tensors=None,
    ), (start, end, int(label))


def collate(rows, device):
    size = max(len(row["encoding"]["input_ids"]) for row in rows)
    batch = {}
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        batch[key] = torch.tensor(
            [row["encoding"][key] + [TOKENIZER.pad_token_id] * (size - len(row["encoding"][key])) for row in rows],
            dtype=torch.long, device=device,
        )
    return batch


def build_rows(pairs, rng, limit_pairs=0):
    rows = []
    started = time.time()
    for index, pair in enumerate(pairs):
        if limit_pairs and index >= limit_pairs:
            break
        chunks = doc_chunks(pair.get("document_text") or "")
        for candidate in select_candidates(pair, rng):
            encoding, (start, end, label) = tokenize(pair, candidate, chunks)
            rows.append({"encoding": encoding, "start": start, "end": end, "label": label})
        if (index + 1) % 500 == 0:
            print("rows", len(rows), "pairs", index + 1, "sec", round(time.time() - started, 1), flush=True)
    return rows


def score_pairs(model, pairs, device, batch_size=24):
    model.eval()
    result = {pair["pair_id"]: [] for pair in pairs}
    rows = []
    for pair in pairs:
        chunks = doc_chunks(pair.get("document_text") or "")
        for candidate in select_candidates(pair, None, gold_aware=False):
            encoding, (start, end, _label) = tokenize(pair, candidate, chunks)
            rows.append((pair["pair_id"], start, end, encoding))
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset:offset + batch_size]
            batch = collate([{"encoding": item[3]} for item in chunk], device)
            logits = model(**batch)
            for item, value in zip(chunk, logits.cpu().numpy()):
                result[item[0]].append((item[1], item[2], float(1 / (1 + math.exp(-float(value))))))
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


def cell(result):
    return {
        "span": result["span_level"]["f1"],
        "token": result["token_level"]["f1"],
        "pair": result["pair_level"]["f1"],
        "fpr": result["fpr"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-limit", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--no-dev-eval", action="store_true")
    parser.add_argument("--out", default="artifacts/neural_pointer_v9")
    args = parser.parse_args()
    global OUT
    OUT = ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)
    device = torch.device("cpu")
    torch.set_num_threads(2)

    held = [p for p in read_jsonl(str(ROOT / "artifacts/gold1200_v3.jsonl"))
            if p.get("split") in ("test", "extra")]
    held_docs = {hashlib.sha256((p.get("document_text") or "").encode()).hexdigest() for p in held}
    rag = [p for p in read_jsonl(str(ROOT / "Dataset/RAGTruth_ru/ragtruth_ru.jsonl"))
           if hashlib.sha256((p.get("document_text") or "").encode()).hexdigest() not in held_docs]
    train = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "train")
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    val = load_split(str(ROOT / "artifacts/gold_v3.jsonl"), "val")
    dev = val + mfava
    print("data", len(rag), len(train), len(mfava), len(val), flush=True)

    rows = build_rows(rag[:args.rag_limit] + train + mfava, rng)
    print("train rows", len(rows), "positive", sum(r["label"] for r in rows), flush=True)

    base = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = Pointer(base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positive = sum(r["label"] for r in rows)
    weight = min(6.0, (len(rows) - positive) / max(1, positive))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    history = []
    best = {"score": -1.0}
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(rows)
        started = time.time()
        losses = []
        for offset in range(0, len(rows), args.batch):
            chunk = rows[offset:offset + args.batch]
            batch = collate(chunk, device)
            labels = torch.tensor([r["label"] for r in chunk], dtype=torch.float32, device=device)
            logits = model(**batch)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        print("epoch", epoch, "loss", round(float(np.mean(losses)), 4),
              "sec", round(time.time() - started, 1), flush=True)
        if not args.no_dev_eval:
            scored = score_pairs(model, dev, device)
        for threshold in (() if args.no_dev_eval else (0.35, 0.45, 0.55, 0.65, 0.75)):
            for max_spans in (1, 2, 3):
                def predictor(pair):
                    return decode(pair, scored, threshold, max_spans, 0.10)
                combined = metrics(dev, predictor)
                v = metrics(val, predictor)
                row = {
                    "epoch": epoch, "threshold": threshold, "max_spans": max_spans,
                    "dev": cell(combined), "val": cell(v),
                }
                history.append(row)
                score = min(combined["span_level"]["f1"], 1.0) + min(combined["token_level"]["f1"], 1.0)
                if combined["fpr"] <= 0.4 and v["fpr"] <= 0.4 and score > best["score"]:
                    best = {"score": score, "row": row, "epoch": epoch}
                    model.base.save_pretrained(str(OUT / "best"))
                    TOKENIZER.save_pretrained(str(OUT / "best"))
                    torch.save({
                        "head": model.head.state_dict(),
                        "hidden": int(base.config.hidden_size),
                    }, str(OUT / "best/head.pt"))
        if args.no_dev_eval:
            target_dir = OUT / "last"
            model.base.save_pretrained(str(target_dir))
            TOKENIZER.save_pretrained(str(target_dir))
            torch.save({"head": model.head.state_dict(), "hidden": int(base.config.hidden_size)},
                       str(target_dir / "head.pt"))
            print("checkpoint", target_dir, "epoch", epoch, flush=True)
        print("best", best.get("row"), flush=True)

    report = {
        "architecture": "rubert-tiny2 + evidence-conditioned start/end pointer (CLS of candidate x CLS of evidence)",
        "rule": "RAGTruth_ru (phrase spans) + v3 RusHallu train + mFAVA train; val/mFAVA dev only; held не читается",
        "rag_pairs": len(rag[:args.rag_limit]), "train_pairs": len(train), "mfava_pairs": len(mfava),
        "no_dev_eval": args.no_dev_eval,
        "rows": len(rows), "positive_rows": positive, "pos_weight": weight,
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "max_len": MAX_LEN,
        "best": best.get("row"), "history": history,
        "checkpoint": str(OUT / "best"),
        "weights_sha256": hashlib.sha256((OUT / "last/model.safetensors").read_bytes()).hexdigest()
        if (OUT / "last/model.safetensors").exists() else (hashlib.sha256((OUT / "best/pytorch_model.bin").read_bytes()).hexdigest()
        if (OUT / "best/pytorch_model.bin").exists() else None),
        "head_sha256": hashlib.sha256((OUT / "last/head.pt").read_bytes()).hexdigest()
        if (OUT / "last/head.pt").exists() else (hashlib.sha256((OUT / "best/head.pt").read_bytes()).hexdigest()
        if (OUT / "best/head.pt").exists() else None),
    }
    if args.no_dev_eval and not (OUT / "best/head.pt").exists():
        model.base.save_pretrained(str(OUT / "best"))
        TOKENIZER.save_pretrained(str(OUT / "best"))
        torch.save({"head": model.head.state_dict(), "hidden": int(base.config.hidden_size)},
                   str(OUT / "best/head.pt"))
    target = OUT / ("report_full.json" if args.no_dev_eval else "report.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", target, flush=True)


if __name__ == "__main__":
    main()
