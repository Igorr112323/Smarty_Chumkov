"""V17 шаг 1: span-level LM-признаки (кросс-внимание ответ→документ) для ранжирования.

Один forward rubert-tiny2 на пару даёт:
  * `attn` — среднее по слоям и головам внимание (S × S),
  * `hidden` — последний слой.

Для каждого кандидата (span) считаются 14 признаков, которых не было в 37 лексических и 10
per-token LM-признаках v15:

  кросс-внимание
    `x_in_doc_mean`   — средняя доля внимания токенов спана на документ,
    `x_in_doc_max`    — максимум по токенам спана,
    `x_out_doc_mean`  — то же для токенов ВНЕ спана (остальной ответ),
    `x_doc_ratio`     — in / (out + eps): «спан смотрит в документ сильнее остального ответа?»,
    `x_boundary`      — внимание на граничных токенах спана минус внутри (контраст границ),
  профиль энтропии
    `x_ent_in_mean`, `x_ent_in_var`, `x_ent_in_min`, `x_ent_out_mean`, `x_ent_ratio`,
  геометрия скрытых состояний
    `x_cos_doc_mean`  — косинус спана к среднему документу,
    `x_cos_peak`      — максимум косинуса к токену документа,
    `x_cos_high`      — доля токенов спана с max косинусом ≥ 0.9 («почти дословно»),
    `x_copy_frac`     — доля токенов спана, чей максимум внимания в документе попадает
                        на токен с тем же id (сигнал копирования/подмены сущности).

Порядок кандидатов совпадает с `spanver.module.candidate_spans` (как в кэшах v14/v16), поэтому
матрицы стыкуются построчно с 37 лексическими признаками и 10 признаками v15.
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
from spanver.module import candidate_spans

MAX_DOC = 380
MAX_ANSWER = 128
BATCH = 8
COLUMNS = ["x_in_doc_mean", "x_in_doc_max", "x_out_doc_mean", "x_doc_ratio", "x_boundary",
           "x_ent_in_mean", "x_ent_in_var", "x_ent_in_min", "x_ent_out_mean", "x_ent_ratio",
           "x_cos_doc_mean", "x_cos_peak", "x_cos_high", "x_copy_frac"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


class Encoder:
    def __init__(self, model_dir="models/rubert-tiny2-deq"):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        self.torch = torch
        self.tok = BertTokenizerFast.from_pretrained(str(ROOT / model_dir))
        config = json.loads((ROOT / model_dir / "config.json").read_text(encoding="utf-8"))
        self.model = BertForSequenceClassification(BertConfig(**config, num_labels=9))
        self.model.load_state_dict(
            torch.load(ROOT / model_dir / "pytorch_model.bin", map_location="cpu"), strict=True)
        self.model.eval()

    def forward(self, documents, answers):
        torch = self.torch
        tok = self.tok
        ids_all, types_all, meta = [], [], []
        for document, answer in zip(documents, answers):
            doc = tok(document or "", add_special_tokens=False, truncation=True,
                      max_length=MAX_DOC)
            enc = tok(answer or "", add_special_tokens=False, truncation=True,
                      max_length=MAX_ANSWER, return_offsets_mapping=True)
            ids = [tok.cls_token_id] + doc["input_ids"] + [tok.sep_token_id]
            types = [0] * len(ids)
            doc_slice = (1, 1 + len(doc["input_ids"]))
            ids += enc["input_ids"] + [tok.sep_token_id]
            types += [1] * (len(enc["input_ids"]) + 1)
            ans_slice = (doc_slice[1] + 1, doc_slice[1] + 1 + len(enc["input_ids"]))
            ids_all.append(ids)
            types_all.append(types)
            meta.append((doc_slice, ans_slice, enc["offset_mapping"], ids))
        width = max(len(row) for row in ids_all)
        ids = torch.tensor([row + [0] * (width - len(row)) for row in ids_all], dtype=torch.long)
        types = torch.tensor([row + [0] * (width - len(row)) for row in types_all],
                             dtype=torch.long)
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids_all],
                            dtype=torch.long)
        with torch.no_grad():
            out = self.model.bert(input_ids=ids, attention_mask=mask, token_type_ids=types,
                                  output_attentions=True, output_hidden_states=True)
        attn = torch.stack(out.attentions, 0).mean(0).mean(1).numpy()  # (B, S, S): слои и головы
        hidden = out.hidden_states[-1].numpy()
        return attn, hidden, meta


def span_block(attn, hidden, meta_row, spans):
    doc_slice, ans_slice, offsets, ids = meta_row
    doc_pos = np.arange(doc_slice[0], doc_slice[1])
    ans_pos = np.arange(ans_slice[0], ans_slice[1])
    if len(doc_pos) == 0 or len(ans_pos) == 0:
        return np.zeros((len(spans), len(COLUMNS)), dtype=np.float32)

    A = attn[np.ix_(ans_pos, doc_pos)]  # (A, D) внимание ответ→документ
    log_d = np.log(max(len(doc_pos), 2))
    ent = -(A * np.log(np.clip(A, 1e-9, None))).sum(1) / log_d
    doc_attn = A.sum(1)

    h = hidden
    norm = np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-6, None)
    unit = h / norm
    ans_unit = unit[ans_pos]
    doc_unit = unit[doc_pos]
    cos_matrix = ans_unit @ doc_unit.T
    cos_peak_tok = cos_matrix.max(1)
    cos_high_tok = np.clip((cos_peak_tok - 0.9) / 0.05, 0, 1)
    doc_mean = doc_unit.mean(0)
    doc_mean = doc_mean / max(np.linalg.norm(doc_mean), 1e-6)
    cos_mean_tok = ans_unit @ doc_mean

    top_doc = A.argmax(1)
    ans_ids = np.asarray([ids[position] for position in ans_pos])
    doc_ids = np.asarray([ids[position] for position in doc_pos])
    copy_tok = (doc_ids[top_doc] == ans_ids).astype(np.float32)

    starts = np.asarray([row[0] for row in offsets])
    ends = np.asarray([row[1] for row in offsets])
    out = np.zeros((len(spans), len(COLUMNS)), dtype=np.float32)
    for index, (start, end) in enumerate(spans):
        inside = (starts < end) & (ends > start)
        if not inside.any():
            continue
        outside = ~inside
        in_doc = doc_attn[inside]
        out_doc = doc_attn[outside] if outside.any() else np.zeros(1)
        boundary = np.zeros(0)
        idx = np.flatnonzero(inside)
        if len(idx) >= 3:
            boundary = np.concatenate([in_doc[:1], in_doc[-1:]])
            inner = in_doc[1:-1]
        else:
            inner = in_doc
        ent_in = ent[inside]
        ent_out = ent[outside] if outside.any() else np.zeros(1)
        out[index] = [
            float(in_doc.mean()),
            float(in_doc.max()),
            float(out_doc.mean()),
            float(in_doc.mean() / (out_doc.mean() + 1e-6)),
            float((boundary.mean() if boundary.size else in_doc.mean())
                  - (inner.mean() if inner.size else in_doc.mean())),
            float(ent_in.mean()),
            float(ent_in.var()),
            float(ent_in.min()),
            float(ent_out.mean()),
            float(ent_in.mean() / (ent_out.mean() + 1e-6)),
            float(cos_mean_tok[inside].mean()),
            float(cos_peak_tok[inside].max()),
            float(cos_high_tok[inside].mean()),
            float(copy_tok[inside].mean()),
        ]
    return out


def group_matrix(encoder, pairs, limit=0, log_every=2000):
    started = time.time()
    blocks = []
    buffer = []
    done = 0
    for pair in pairs[:limit] if limit else pairs:
        buffer.append(pair)
        if len(buffer) >= BATCH:
            blocks.extend(process(encoder, buffer))
            done += len(buffer)
            buffer = []
            if log_every and done % log_every == 0:
                print(f"  {done} пар, {time.time() - started:.1f}s", flush=True)
    if buffer:
        blocks.extend(process(encoder, buffer))
        done += len(buffer)
    matrix = np.vstack(blocks) if blocks else np.zeros((0, len(COLUMNS)), dtype=np.float32)
    return matrix, done


def process(encoder, pairs):
    documents = [pair.get("document_text") or "" for pair in pairs]
    answers = [pair.get("model_output") or "" for pair in pairs]
    attn, hidden, meta = encoder.forward(documents, answers)
    blocks = []
    for index, pair in enumerate(pairs):
        spans = [(start, end) for start, end, _ in candidate_spans(pair.get("model_output") or "")]
        blocks.append(span_block(attn[index], hidden[index], meta[index], spans))
    return blocks


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ranker_features_v17.json")
    parser.add_argument("--cache-out", default="artifacts/cache_v17")
    parser.add_argument("--sber-limit", type=int, default=20000)
    parser.add_argument("--groups", default="val,real,sber")
    args = parser.parse_args(argv)

    L.set_weights(L.weights())
    train, val, mfava = L.load_all()
    groups = {
        "val": val, "mfava": mfava, "real": train,
        "sber": read_jsonl(ROOT / "artifacts/sber_flip_v16.jsonl")[:args.sber_limit],
        "synth": read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl"),
        "pseudo": read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl"),
    }
    wanted = [name.strip() for name in args.groups.split(",") if name.strip()]
    encoder = Encoder()
    out_dir = ROOT / args.cache_out
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"rule": "span-level кросс-внимание rubert-tiny2 (dequant int8 ONNX → transformers, "
                      "eval без дообучения); порядок кандидатов как в spanver.candidate_spans",
              "columns": COLUMNS, "input": {"max_doc_tokens": MAX_DOC,
                                            "max_answer_tokens": MAX_ANSWER, "batch": BATCH},
              "groups": {}}
    for name in wanted:
        pairs = groups[name]
        started = time.time()
        matrix, done = group_matrix(encoder, pairs)
        elapsed = time.time() - started
        target = out_dir / f"{name}_span.npz"
        np.savez_compressed(target, X=matrix, columns=np.asarray(COLUMNS))
        chars = sum(len(p.get("document_text") or "") + len(p.get("model_output") or "")
                    for p in pairs[:done])
        report["groups"][name] = {
            "pairs": done, "candidates": int(matrix.shape[0]), "seconds": round(elapsed, 1),
            "seconds_per_1k_chars": round(elapsed / max(chars, 1) * 1000, 4),
            "mean": [round(float(value), 4) for value in matrix.mean(0)],
            "std": [round(float(value), 4) for value in matrix.std(0)],
        }
        print(name, matrix.shape, f"{elapsed:.1f}s", flush=True)
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
