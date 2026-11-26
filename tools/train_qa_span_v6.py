"""QA-style start/end span extractor v6.

Принципиальное отличие от v3 token classifier и v4/v5 candidate rankers:
  - нет кандидатов и BIO-меток; модель предсказывает start/end logits напрямую;
  - несколько gold span кодируются как multi-label BCE;
  - pair gate baseline замораживается, поэтому FPR/pair не переобучается;
  - early stopping и все пороги выбираются только на val+mFAVA-train.

Веса и логи сохраняются в artifacts/qa_span_v6/.
"""
import argparse
import collections
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import iou, load_split, metrics, read_jsonl

MODEL_DIR = ROOT / "models/rubert-tiny2"
OUT = ROOT / "artifacts/qa_span_v6"
WINDOW = 1200
STEP = 900
DOC_LIMIT = 1500
MAX_LEN = 512
TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL_DIR))


class QASpan(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.start = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.base(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        return self.start(hidden).squeeze(-1), self.end(hidden).squeeze(-1)


def windows(pair):
    answer = pair.get("model_output") or ""
    doc = (pair.get("document_text") or "")[:DOC_LIMIT]
    gold = pair.get("spans") or []
    out = []
    for start in range(0, max(1, len(answer)), STEP):
        chunk = answer[start:start + WINDOW]
        if not chunk.strip():
            continue
        enc = TOKENIZER(
            chunk, doc, truncation=True, max_length=MAX_LEN,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        positions = []
        indices = []
        starts = []
        ends = []
        for index, (left, right), segment in zip(
            range(len(enc["input_ids"])), offsets, enc["token_type_ids"]
        ):
            if segment == 1 or right <= left:
                continue
            char_start = start + int(left)
            char_end = start + int(right) - 1
            positions.append((char_start, char_end))
            indices.append(index)
            starts.append(any(
                g["start"] <= char_start <= g["end"] for g in gold
            ))
            ends.append(any(
                g["start"] <= char_end <= g["end"] for g in gold
            ))
        if not positions:
            continue
        out.append({
            "encoding": {k: v for k, v in enc.items()},
            "positions": positions,
            "indices": indices,
            "starts": starts,
            "ends": ends,
            "window_start": start,
            "window_end": start + len(chunk) - 1,
        })
    return out


def collate(items, device):
    max_len = max(len(item["encoding"]["input_ids"]) for item in items)
    batch = {}
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        values = []
        for item in items:
            row = item["encoding"][key]
            values.append(row + [TOKENIZER.pad_token_id] * (max_len - len(row)))
        batch[key] = torch.tensor(values, dtype=torch.long, device=device)
    starts = torch.zeros(len(items), max_len, dtype=torch.float32, device=device)
    ends = torch.zeros(len(items), max_len, dtype=torch.float32, device=device)
    mask = torch.zeros(len(items), max_len, dtype=torch.bool, device=device)
    for i, item in enumerate(items):
        length = len(item["encoding"]["input_ids"])
        mask[i, :length] = True
        indices = torch.tensor(item["indices"], dtype=torch.long, device=device)
        starts[i, indices] = torch.tensor(item["starts"], dtype=torch.float32, device=device)
        ends[i, indices] = torch.tensor(item["ends"], dtype=torch.float32, device=device)
    return batch, starts, ends, mask


def predict_pair(model, pair, device):
    model.eval()
    items = windows(pair)
    starts = []
    ends = []
    with torch.no_grad():
        for i in range(0, len(items), 8):
            batch, _, _, mask = collate(items[i:i + 8], device)
            logits_s, logits_e = model(**batch)
            for j, item in enumerate(items[i:i + 8]):
                length = len(item["encoding"]["input_ids"])
                ps = torch.sigmoid(logits_s[j, :length]).cpu().numpy()
                pe = torch.sigmoid(logits_e[j, :length]).cpu().numpy()
                for (a, b), p_start, p_end in zip(item["positions"], ps, pe):
                    starts.append((a, b, float(p_start)))
                    ends.append((a, b, float(p_end)))
    return starts, ends


def decode(pair, data, min_len, max_len, top, threshold, nms, max_spans, flags, mode):
    pid = pair["pair_id"]
    if not flags.get(pid):
        return []
    starts, ends = data[pid]
    answer = pair.get("model_output") or ""
    starts = sorted(starts, key=lambda x: -x[2])[:top]
    ends = sorted(ends, key=lambda x: -x[2])[:top]
    candidates = []
    for a, _, ps in starts:
        for b, _, pe in ends:
            length = b - a + 1
            if not (min_len <= length <= max_len):
                continue
            score = ps * pe
            if score < threshold:
                continue
            candidates.append((score, a, b))
    candidates.sort(reverse=True)
    out = []
    for score, a, b in candidates:
        if any(iou(a, b, o["start"], o["end"]) > nms for o in out):
            continue
        out.append({
            "start": a, "end": b, "risk": score,
            "type": "unsupported_addition", "confidence": 1.0,
        })
        if len(out) >= max_spans:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    flags = pickle.load(open(ROOT / "artifacts/baseline_flags_v5.pkl", "rb"))
    device = torch.device("cpu")
    torch.manual_seed(0)
    base = AutoModel.from_pretrained(str(MODEL_DIR))
    model = QASpan(base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor(8.0, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cache = {p["pair_id"]: windows(p) for p in train + val + ex}
    print("windows", sum(len(v) for v in cache.values()), flush=True)
    train_items = [w for p in train for w in cache[p["pair_id"]]]
    np.random.seed(0)
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(train_items)
        np.random.shuffle(order)
        losses = []
        for i in range(0, len(order), args.batch):
            batch, starts, ends, mask = collate(order[i:i + args.batch], device)
            logits_s, logits_e = model(**batch)
            loss = loss_fn(logits_s[mask], starts[mask]) + loss_fn(logits_e[mask], ends[mask])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        model.eval()
        data = {}
        for p in val + ex:
            data[p["pair_id"]] = predict_pair(model, p, device)
        sweep = []
        for threshold in (0.05, 0.1, 0.2, 0.3):
            for max_spans in (1, 3, 5):
                def predict(pair):
                    return decode(pair, data, 5, 800, 20, threshold, 0.1, max_spans, flags, None)
                mv = metrics(val, predict)
                me = metrics(ex, predict)
                sweep.append({
                    "threshold": threshold, "max_spans": max_spans,
                    "val": mv, "mfava": me,
                })
        best = max(sweep, key=lambda x: (
            x["val"]["span_level"]["f1"] * 6 + x["mfava"]["span_level"]["f1"],
            x["val"]["token_level"]["f1"] * 6 + x["mfava"]["token_level"]["f1"],
        ))
        history.append({
            "epoch": epoch, "loss": round(float(np.mean(losses)), 4),
            "best": best, "sec": round(time.time() - started, 1),
        })
        print("epoch", epoch, "loss", round(float(np.mean(losses)), 4),
              "val", best["val"]["span_level"]["f1"],
              "mfava", best["mfava"]["span_level"]["f1"],
              "th", best["threshold"], "max", best["max_spans"], flush=True)
        head = {k: v for k, v in model.state_dict().items()
                if k.startswith("start.") or k.startswith("end.")}
        torch.save(head, str(OUT / "head.pt"))
    report = {
        "rule": "QA start/end; frozen pair gate; val+mFAVA selection only",
        "history": history,
        "config": {"window": WINDOW, "step": STEP, "doc_limit": DOC_LIMIT,
                   "max_len": MAX_LEN, "batch": args.batch, "lr": args.lr,
                   "epochs": args.epochs, "pos_weight": 8.0},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
