"""V15 шаг 2: LM-признаки (rubert-tiny2, веса восстановлены из int8 ONNX) на кандидатах.

Модель: `models/rubert-tiny2-deq` (312/3/12, vocab 83 828) — реконструкция из
`tools/external/rubert-turbo-payments/models/onnx_model/model_quantized.onnx`
(dequant w = (q − zp)·scale). Дообучения нет: только предобученный энкодер в eval-режиме.

Вход LM: [CLS] документ (≤ 380 wordpiece) [SEP] ответ (≤ 128 wordpiece) [SEP].
На каждый токен ответа считаются 5 числовых признаков (договор §4.1):

  1. attn_entropy   — нормированная энтропия внимания токена к документу;
  2. attn_doc_share — доля внимания токена, ушедшая в документ;
  3. attn_doc_peak  — максимальный вес внимания на один токен документа;
  4. cos_doc_mean   — косинус скрытого состояния токена и среднего по документу;
  5. cos_doc_max    — максимальный косинус к токену документа.

Для кандидата (start, end) берутся mean и max по перекрывающимся токенам → 10 колонок,
которые добавляются к 37 штатным признакам. Порядок кандидатов берётся из кэша v14
(`artifacts/cache_v14/*.npz`), поэтому матрицы выравнены построчно.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import v11_lab as L

MAX_DOC = 380
MAX_ANSWER = 128
BATCH = 8
NAMES = ["attn_entropy", "attn_doc_share", "attn_doc_peak", "cos_doc_mean", "cos_doc_max"]
COLUMNS = [f"{name}_{agg}" for name in NAMES for agg in ("mean", "max")]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


class Encoder:
    def __init__(self, model_dir):
        import torch
        from transformers import BertForSequenceClassification, BertTokenizerFast
        self.torch = torch
        self.tokenizer = BertTokenizerFast.from_pretrained(str(model_dir))
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        from transformers import BertConfig
        self.model = BertForSequenceClassification(
            BertConfig(**config, num_labels=9, output_attentions=True))
        state = torch.load(Path(model_dir) / "pytorch_model.bin", map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.config.output_attentions = True
        self.model.config.output_hidden_states = True

    def encode(self, documents, answers):
        torch = self.torch
        tok = self.tokenizer
        batch_ids, batch_types, offsets = [], [], []
        for document, answer in zip(documents, answers):
            doc = tok(document or "", add_special_tokens=False, truncation=True, max_length=MAX_DOC)
            enc = tok(answer or "", add_special_tokens=False, truncation=True,
                      max_length=MAX_ANSWER, return_offsets_mapping=True)
            ids = [tok.cls_token_id] + doc["input_ids"] + [tok.sep_token_id]
            types = [0] * len(ids)
            doc_slice = (1, 1 + len(doc["input_ids"]))
            ids += enc["input_ids"] + [tok.sep_token_id]
            types += [1] * (len(enc["input_ids"]) + 1)
            ans_slice = (doc_slice[1] + 1, doc_slice[1] + 1 + len(enc["input_ids"]))
            batch_ids.append(ids)
            batch_types.append(types)
            offsets.append((doc_slice, ans_slice, enc["offset_mapping"], len(doc["input_ids"])))
        width = max(len(row) for row in batch_ids)
        ids = torch.tensor([row + [0] * (width - len(row)) for row in batch_ids], dtype=torch.long)
        types = torch.tensor([row + [0] * (width - len(row)) for row in batch_types],
                             dtype=torch.long)
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in batch_ids],
                            dtype=torch.long)
        with torch.no_grad():
            out = self.model.bert(input_ids=ids, attention_mask=mask, token_type_ids=types,
                                  output_attentions=True, output_hidden_states=True)
        attn = torch.stack(out.attentions, 0).mean(0)  # (B, heads, S, S)
        hidden = out.hidden_states[-1]
        return attn, hidden, offsets

    def pair_features(self, attn, hidden, offsets_row):
        torch = self.torch
        doc_slice, ans_slice, offset_mapping, n_doc = offsets_row
        if n_doc == 0 or ans_slice[1] <= ans_slice[0]:
            return np.zeros((0, 5), dtype=np.float32), offset_mapping
        p_doc = attn[0][:, ans_slice[0]:ans_slice[1], doc_slice[0]:doc_slice[1]]
        safe = p_doc.clamp_min(1e-9)
        entropy = -(safe * safe.log()).sum(-1) / float(np.log(max(n_doc, 2)))
        entropy = entropy.mean(0)
        share = p_doc.sum(-1).mean(0)
        peak = p_doc.max(-1).values.mean(0)
        h = hidden[0]
        norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit = h / norm
        ans_unit = unit[ans_slice[0]:ans_slice[1]]
        doc_unit = unit[doc_slice[0]:doc_slice[1]]
        cos_matrix = ans_unit @ doc_unit.T
        doc_mean = doc_unit.mean(0, keepdim=True)
        doc_mean = doc_mean / doc_mean.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos_mean = (ans_unit @ doc_mean.T).squeeze(-1)
        cos_max = cos_matrix.max(-1).values
        feats = torch.stack([entropy, share, peak, cos_mean, cos_max], dim=-1)
        return feats.cpu().numpy().astype(np.float32), offset_mapping


def span_features(feats, offset_mapping, spans):
    """mean/max по токенам, перекрывающим кандидата (10 колонок)."""
    n_tokens = feats.shape[0]
    out = np.zeros((len(spans), 2 * len(NAMES)), dtype=np.float32)
    if n_tokens == 0:
        return out
    starts = np.asarray([row[0] for row in offset_mapping])
    ends = np.asarray([row[1] for row in offset_mapping])
    for index, (start, end) in enumerate(spans):
        mask = (starts < end) & (ends > start)
        if not mask.any():
            centers = (starts + ends) / 2.0
            pick = int(np.argmin(np.abs(centers - (start + end) / 2.0)))
            mask = np.zeros(n_tokens, dtype=bool)
            mask[pick] = True
        block = feats[mask]
        values = np.concatenate([block.mean(0), block.max(0)])
        out[index] = values
    return out


def feature_stats(matrix, labels):
    positives = labels == 1
    return {
        column: {"pos_mean": round(float(matrix[positives, index].mean()), 4) if positives.any() else None,
                 "neg_mean": round(float(matrix[~positives, index].mean()), 4) if (~positives).any() else None,
                 "std": round(float(matrix[:, index].std()), 4)}
        for index, column in enumerate(COLUMNS)}


def cache_spans(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    return meta, data["y"]


def group_matrix(encoder, pairs, cache_path, log_every=200, limit=0):
    meta, _ = cache_spans(cache_path)
    if limit:
        meta = meta[:limit]
    by_id = {pair["pair_id"]: pair for pair in pairs}
    rows = []
    started = time.time()
    buffer = []
    order = []
    for entry in meta:
        order.append(by_id[entry["pair_id"]])
        buffer.append((by_id[entry["pair_id"]], entry))
        if len(buffer) >= BATCH:
            rows.extend(process_batch(encoder, buffer))
            buffer = []
        if len(order) % log_every == 0:
            print(f"  {len(order)} пар, {time.time() - started:.1f}s", flush=True)
    if buffer:
        rows.extend(process_batch(encoder, buffer))
    matrix = np.vstack(rows) if rows else np.zeros((0, len(COLUMNS)), dtype=np.float32)
    return matrix, len(meta)


def process_batch(encoder, entries):
    """entries: [(pair, meta-запись)] — кандидаты берутся из meta (порядок v14)."""
    documents = [pair.get("document_text") or "" for pair, _ in entries]
    answers = [pair.get("model_output") or "" for pair, _ in entries]
    attn, hidden, offsets = encoder.encode(documents, answers)
    blocks = []
    for index, (_, entry) in enumerate(entries):
        feats, offset_mapping = encoder.pair_features(attn[index:index + 1],
                                                      hidden[index:index + 1], offsets[index])
        blocks.append(span_features(feats, offset_mapping, entry["spans"]))
    return blocks


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/rubert-tiny2-deq")
    parser.add_argument("--report", default="artifacts/lm_features_v15.json")
    parser.add_argument("--cache-in", default="artifacts/cache_v14")
    parser.add_argument("--cache-out", default="artifacts/cache_v15")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--groups", default="")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    train, val, mfava = L.load_all()
    groups = {
        "real": train,
        "synth": read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl"),
        "pseudo": read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl"),
        "ragtruth": read_jsonl(ROOT / "artifacts/ragtruth_proj_v14.jsonl"),
        "val": val,
        "mfava": mfava,
    }
    if args.groups:
        wanted = [name.strip() for name in args.groups.split(",") if name.strip()]
        groups = {name: groups[name] for name in wanted}
    encoder = None
    out_dir = ROOT / args.cache_out
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "rule": "предобученный энкодер rubert-tiny2 (dequant из int8 ONNX) в eval-режиме; "
                "дообучения нет; 5 признаков на токен ответа, mean/max по кандидату",
        "weights": "models/rubert-tiny2-deq (см. artifacts/weights_v15.json)",
        "columns": COLUMNS,
        "input": {"max_doc_tokens": MAX_DOC, "max_answer_tokens": MAX_ANSWER,
                  "template": "[CLS] документ [SEP] ответ [SEP]"},
        "groups": {},
    }
    for name, pairs in groups.items():
        started = time.time()
        cache_in = ROOT / args.cache_in / f"{name}.npz"
        target = out_dir / f"{name}_lm.npz"
        _, expected = cache_spans(cache_in)
        if target.exists() and not args.rebuild and not args.limit:
            cached = np.load(target)["X"]
            if cached.shape[0] == expected.shape[0]:
                print(f"{name}: уже готово ({cached.shape[0]} строк), пропуск", flush=True)
                y_cached = np.load(target)["y"]
                report["groups"][name] = {"candidates": int(cached.shape[0]),
                                          "positives": int((y_cached == 1).sum()),
                                          "cached": True,
                                          "feature_stats": feature_stats(cached, y_cached)}
                (ROOT / args.report).write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                continue
        if encoder is None:
            encoder = Encoder(ROOT / args.model_dir)
        matrix, n_pairs = group_matrix(encoder, pairs, cache_in, limit=args.limit)
        elapsed = time.time() - started
        _, y = cache_spans(cache_in)
        y = y[:matrix.shape[0]]
        if matrix.shape[0] != y.shape[0]:
            raise SystemExit(f"{name}: строк {matrix.shape[0]} != кандидатов {y.shape[0]}")
        np.savez_compressed(target, X=matrix, y=y,
                            columns=np.asarray(COLUMNS))
        positives = y == 1
        entry = {
            "pairs": n_pairs,
            "candidates": int(matrix.shape[0]),
            "positives": int(positives.sum()), "seconds": round(elapsed, 1),
            "seconds_per_1k_chars": round(
                elapsed / max(sum(len(p.get("document_text") or "") + len(p.get("model_output") or "")
                                  for p in pairs), 1) * 1000, 4),
            "feature_stats": feature_stats(matrix, y),
        }
        report["groups"][name] = entry
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        print(name, matrix.shape, {k: entry[k] for k in ("seconds", "seconds_per_1k_chars")},
              flush=True)
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
