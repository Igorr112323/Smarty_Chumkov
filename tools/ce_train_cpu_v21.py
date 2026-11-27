"""V21 (CPU-прогон): обучение cross-encoder rubert-tiny2 на 2 ядрах и честная проверка Span.

Отличия от GPU-версии (`ce_train_v21.py`) — только ради CPU-бюджета, методика та же:

  * окно `[CLS] doc_window [SEP] candidate [SEP]`, `max_len = 128`, doc_window = 64 токена;
  * `doc_window` выбирается по лексическому перекрытию с ответом ( cheap-RAG ), а не как префикс —
    так модель видит релевантный кусок документа;
  * в обучение берутся ВСЕ позитивы (IoU >= 0.5), ВСЕ hard negatives (0.3–0.7) и не более K=2
    лёгких негативов на пару → ~4 кандидата на пару вместо 16;
  * всё пре-токенизируется один раз (шаг обучения = только forward/backward).

Loss: CE(start) + CE(end) + BCE(risk). Батчи с балансом 50/50 позитив/негатив.

Арм ablation: real / v19 / real+v19 (+synth при `--include-synth`).
Проверка: val AUC риска, метрики договора через декадер v11 (пороги на val в бюджете FPR <= 0.40),
OOF по документам (`--oof N`, дообучение на фолде от общего чекпойнта).
Held-out не читается.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

import sweep_v16 as S
import v11_lab as L
from ce_model_v21 import build_model, load_tokenizer
from spanver.evaluate import iou
from spanver.module import candidate_spans


def read_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def doc_window(document, answer, doc_tokens=64):
    """Кусок документа максимально близкий к ответу (лексическое перекрытие), до `doc_tokens` слов."""
    sentences = [part.strip() for part in (document or "").replace("\n", " ").split(".") if part.strip()]
    if not sentences:
        return (document or "")[:doc_tokens * 6]
    answer_words = {word for word in answer.lower().split() if len(word) > 3}
    best, best_score, best_start = None, -1.0, 0
    for index in range(len(sentences)):
        chunk, score, cursor = "", 0.0, index
        while cursor < len(sentences) and len(chunk.split()) < doc_tokens:
            words = sentences[cursor].split()
            take = words[:doc_tokens - len(chunk.split())]
            piece = " ".join(take)
            if answer_words:
                overlap = len(answer_words & {word for word in piece.lower().split() if len(word) > 3})
                score += overlap + 0.5 * len(piece)
            else:
                score += len(piece)
            chunk = (chunk + " " + piece).strip()
            cursor += 1
        if score > best_score:
            best, best_score, best_start = chunk, score, index
    return best or (document or "")[:doc_tokens * 6]


def pair_candidates(pair, max_easy=2):
    answer = pair.get("model_output") or ""
    gold = pair.get("spans") or []
    positives, hard, easy = [], [], []
    for start, end, fragment in candidate_spans(answer):
        if not fragment.strip():
            continue
        score = max([iou(start, end, span["start"], span["end"]) for span in gold] or [0.0])
        item = {"start": start, "end": end, "text": fragment,
                "label": 1 if score >= 0.5 else 0, "iou": score}
        if score >= 0.5:
            positives.append(item)
        elif 0.3 <= score < 0.7:
            hard.append(item)
        else:
            easy.append(item)
    return positives + hard + (easy if max_easy < 0 else easy[:max_easy])


def pack(pairs, tokenizer, doc_tokens=64, max_len=128, max_easy=2, limit=0):
    """Пре-токенизация: возвращает тензоры и метаданные (порядок = порядок пар и кандидатов)."""
    ids_list, types_list, mask_list, cand_mask_list = [], [], [], []
    start_list, end_list, risk_list, meta = [], [], [], []
    for pair in (pairs[:limit] if limit else pairs):
        answer = pair.get("model_output") or ""
        if not answer:
            continue
        window = doc_window(pair.get("document_text") or "", answer, doc_tokens)
        doc_ids = tokenizer(window, add_special_tokens=False)["input_ids"][:doc_tokens]
        for candidate in pair_candidates(pair, max_easy):
            encoded = tokenizer(candidate["text"], add_special_tokens=False,
                                return_offsets_mapping=True,
                                truncation=True, max_length=max_len - len(doc_ids) - 3)
            cand_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if not cand_ids:
                continue
            ids = [tokenizer.cls_token_id] + doc_ids + [tokenizer.sep_token_id] + cand_ids + \
                  [tokenizer.sep_token_id]
            types = [0] * (len(doc_ids) + 2) + [1] * (len(cand_ids) + 1)
            cand_mask = [0] * (len(doc_ids) + 2) + [1] * len(cand_ids) + [0]
            start_index = end_index = None
            if candidate["label"] == 1:
                start_index = next((index for index, (left, right) in enumerate(offsets)
                                    if right > 0), None)
                end_index = next((index for index in range(len(offsets) - 1, -1, -1)
                                  if offsets[index][0] < len(candidate["text"])), None)
                if start_index is not None and end_index is not None and start_index > end_index:
                    start_index = end_index = None
            base = len(doc_ids) + 2
            pad = max_len - len(ids)
            ids_list.append(ids + [tokenizer.pad_token_id] * pad)
            types_list.append(types + [0] * pad)
            mask_list.append([1] * len(ids) + [0] * pad)
            cand_mask_list.append(cand_mask + [0] * pad)
            start_list.append(base + start_index if start_index is not None else -100)
            end_list.append(base + end_index if end_index is not None else -100)
            risk_list.append(1.0 if candidate["label"] == 1 else 0.0)
            meta.append({"pair_id": pair["pair_id"], "start": candidate["start"],
                         "end": candidate["end"], "label": candidate["label"]})
    tensors = {
        "input_ids": torch.tensor(ids_list, dtype=torch.int32),
        "token_type_ids": torch.tensor(types_list, dtype=torch.int8),
        "attention_mask": torch.tensor(mask_list, dtype=torch.int8),
        "candidate_mask": torch.tensor(cand_mask_list, dtype=torch.int8),
        "start": torch.tensor(start_list, dtype=torch.long),
        "end": torch.tensor(end_list, dtype=torch.long),
        "risk": torch.tensor(risk_list, dtype=torch.float32),
    }
    return tensors, meta


def batches(risk, batch, seed=21):
    positives = np.flatnonzero((risk == 1.0).numpy())
    negatives = np.flatnonzero((risk != 1.0).numpy())
    rng = random.Random(seed)
    positives = list(positives)
    negatives = list(negatives)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    half = max(batch // 2, 1)
    result, pos_index, neg_index = [], 0, 0
    for _ in range(math.ceil(len(risk) / batch)):
        chunk = []
        for _ in range(half):
            if pos_index >= len(positives):
                pos_index = 0
                rng.shuffle(positives)
            if not positives:
                break
            chunk.append(positives[pos_index])
            pos_index += 1
        for _ in range(batch - len(chunk)):
            if neg_index >= len(negatives):
                neg_index = 0
                rng.shuffle(negatives)
            if not negatives:
                break
            chunk.append(negatives[neg_index])
            neg_index += 1
        if chunk:
            result.append(chunk)
    return result


def run_epoch(model, tensors, optimizer, batch, grad_clip=1.0):
    model.train()
    losses = []
    for chunk in batches(tensors["risk"], batch):
        index = torch.tensor(chunk, dtype=torch.long)
        feeds = {"input_ids": tensors["input_ids"][index].long(),
                 "token_type_ids": tensors["token_type_ids"][index].long(),
                 "attention_mask": tensors["attention_mask"][index].long(),
                 "candidate_mask": tensors["candidate_mask"][index].long()}
        outputs = model(**feeds)
        valid = tensors["start"][index] >= 0
        if int(valid.sum()) > 0:
            span_loss = nn.functional.cross_entropy(outputs["start_logits"][valid],
                                                    tensors["start"][index][valid]) + \
                nn.functional.cross_entropy(outputs["end_logits"][valid],
                                            tensors["end"][index][valid])
        else:
            span_loss = outputs["start_logits"].sum() * 0.0
        risk_loss = nn.functional.binary_cross_entropy_with_logits(outputs["risk_logit"],
                                                                   tensors["risk"][index])
        loss = span_loss + risk_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss))
    return round(float(np.mean(losses)), 4) if losses else None


def scores(model, tensors, batch=32):
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, tensors["risk"].shape[0], batch):
            index = torch.arange(offset, min(offset + batch, tensors["risk"].shape[0]))
            feeds = {"input_ids": tensors["input_ids"][index].long(),
                     "token_type_ids": tensors["token_type_ids"][index].long(),
                     "attention_mask": tensors["attention_mask"][index].long(),
                     "candidate_mask": tensors["candidate_mask"][index].long()}
            outputs = model(**feeds)
            values.append(torch.sigmoid(outputs["risk_logit"]).numpy())
    return np.concatenate(values) if values else np.zeros(0)


def apply_to_tables(tables, meta, values, risk_key="risk_ce"):
    # кандидаты вне CE-подмножества остаются на штатном риске v11 (каскад, как в speed-анализе)
    for table in tables:
        for row in table["rows"]:
            row[risk_key] = row.get("risk_cand", 0.0)
    by_id = {table["pair_id"]: table for table in tables}
    for index, entry in enumerate(meta):
        table = by_id.get(entry["pair_id"])
        if table is None:
            continue
        for row in table["rows"]:
            if row["start"] == entry["start"] and row["end"] == entry["end"]:
                row[risk_key] = float(values[index])
                break


def contract(pairs, tables, risk_key="risk_ce"):
    rows = [{**L.cell(S.evaluate(pairs, tables, config, risk_key)), "config": config}
            for config in S.grid()]
    feasible = [row for row in rows if row["fpr"] <= 0.40]
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    best = feasible[0] if feasible else None
    if best is None:
        return None
    return {key: best[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")} | \
        {"config": best["config"]}


V19_HELDOUT_LEAK = {
    "ruclv19_000064", "ruclv19_000279", "ruclv19_000315", "ruclv19_000544",
    "ruclv19_000786", "ruclv19_000810", "ruclv19_001358",
}


def freeze_lower_layers(model, count):
    """Заморозка нижних слоёв энкодера (0 = полное дообучение, как в v21)."""
    if count <= 0:
        return
    layers = getattr(model.encoder, "encoder", None)
    layers = getattr(layers, "layer", None) or []
    for layer in layers[:count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    print(f"заморожено нижних слоёв: {min(count, len(layers))} из {len(layers)}", flush=True)


def train_arm(name, pairs, tokenizer, args, checkpoint=None):
    tensors, meta = pack(pairs, tokenizer, args.doc_tokens, args.max_len, args.max_easy)
    model = build_model(model_dir=args.encoder or None)
    freeze_lower_layers(model, args.freeze_layers)
    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history = []
    for epoch in range(args.epochs):
        started = time.time()
        loss = run_epoch(model, tensors, optimizer, args.batch)
        history.append({"epoch": epoch, "loss": loss, "seconds": round(time.time() - started, 1)})
        print(f"  {name} epoch {epoch}: loss {loss} за {history[-1]['seconds']} c", flush=True)
    return model, tensors, meta, history


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ce_train_v21.json")
    parser.add_argument("--checkpoint", default="artifacts/ce_model_cpu_v21.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--doc-tokens", type=int, default=64)
    parser.add_argument("--max-easy", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-synth", action="store_true")
    parser.add_argument("--arms", default="real,v19,real_plus_v19")
    parser.add_argument("--oof", type=int, default=0)
    parser.add_argument("--oof-epochs", type=int, default=1)
    parser.add_argument("--encoder", default=os.environ.get("CE_MODEL_DIR", ""),
                        help="каталог с весами энкодера (v22: rubert-base-cased), по умолчанию rubert-tiny2")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="заморозить нижние N слоёв энкодера (экономия памяти/времени на CPU)")
    args = parser.parse_args(argv)

    started_all = time.time()
    torch.set_num_threads(int(os.environ.get("CE_THREADS", "2")))
    tokenizer = load_tokenizer(args.encoder or None)
    train_pairs, val_pairs, mfava = L.load_all()
    v19_all = read_jsonl(ROOT / "artifacts/ru_clause_gen_v19.jsonl")
    # 7 пар v19 (base_corpus = mfava_ru) побайтово совпадают с документами held-out (см. ce_data_v21.json,
    # audit_leak_v19_heldout) — из обучения исключаются, чтобы ни один арм не видел эти документы
    v19 = [pair for pair in v19_all if pair.get("pair_id") not in V19_HELDOUT_LEAK]
    print(f"v19: {len(v19)} из {len(v19_all)} пар (исключено {len(v19_all) - len(v19)} с ликом в held-out)",
          flush=True)
    synth = read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl") if args.include_synth else []

    if args.limit and args.limit > 0:  # v22: дешёвый smoke-прогон на подмножестве пар
        train_pairs = train_pairs[:args.limit]
        print(f"ограничение выбора: {args.limit} пар на арм", flush=True)

    arms_data = {
        "real": train_pairs,
        "v19": v19,
        "real_plus_v19": train_pairs + v19,
        "real_plus_v19_plus_synth": train_pairs + v19 + synth,
    }

    tables_val = S.build_tables(val_pairs, L.weights(), ROOT / "artifacts/cache_v16/val_tables.json")
    val_tensors, val_meta = pack(val_pairs, tokenizer, args.doc_tokens, args.max_len, max_easy=-1)
    val_labels = [entry["label"] for entry in val_meta]

    report = {
        "rule": "v21 CPU-прогон: cross-encoder rubert-tiny2 (29.2M) на 2 ядрах; окно 128 токенов "
                "(документ-окно 64 по лексическому перекрытию); в батче 50/50 позитив/негатив; "
                "loss = CE(start) + CE(end) + BCE(risk); пороги декадера калибруются на val в "
                "бюджете FPR <= 0.40; held-out не читается",
        "device": {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                   "threads": torch.get_num_threads(),
                   "note": "GPU нет — обучение на CPU (2 ядра), это offline-процесс"},
        "window": {"max_len": args.max_len, "doc_tokens": args.doc_tokens,
                   "max_easy_negatives_per_pair": args.max_easy,
                   "layout": "[CLS] doc_window [SEP] candidate [SEP]"},
        "training": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr},
        "val_data": {"candidates": int(val_tensors["risk"].shape[0]),
                     "positives": int(sum(1 for label in val_labels if label == 1))},
        "arms": {},
    }

    best_arm, best_span, best_checkpoint_state = None, -1.0, None
    for arm in [name.strip() for name in args.arms.split(",") if name.strip()]:
        if arm not in arms_data:
            continue
        pairs = arms_data[arm]
        print(f"арм {arm}: {len(pairs)} пар", flush=True)
        model, tensors, meta, history = train_arm(arm, pairs, tokenizer, args)
        val_scores = scores(model, val_tensors)
        auc = float(roc_auc_score(val_labels, val_scores))
        apply_to_tables(tables_val, val_meta, val_scores)
        cell = contract(val_pairs, tables_val)
        entry = {"pairs": len(pairs), "examples": int(tensors["risk"].shape[0]),
                 "positives": int(float(tensors["risk"].sum())),
                 "history": history, "val_auc": round(auc, 4), "val": cell,
                 "within_pair": None}
        # within-pair accuracy
        by_pair = {}
        for index, item in enumerate(val_meta):
            by_pair.setdefault(item["pair_id"], []).append((item["label"], val_scores[index]))
        correct = total = 0
        for values in by_pair.values():
            positives = [score for label, score in values if label == 1]
            negatives = [score for label, score in values if label == 0]
            for left in positives:
                for right in negatives:
                    total += 1
                    correct += 1 if left > right else 0
        entry["within_pair"] = round(correct / total, 4) if total else None
        report["arms"][arm] = entry
        print(f"  → val AUC {entry['val_auc']} within {entry['within_pair']} "
              f"val {cell}", flush=True)

        if cell and cell["span"] > best_span:
            best_span, best_arm = cell["span"], arm
            best_checkpoint_state = {key: value.clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": model.state_dict(), "arm": arm, "args": vars(args)},
                       ROOT / args.checkpoint)
            report["checkpoint"] = {"path": args.checkpoint,
                                    "bytes": (ROOT / args.checkpoint).stat().st_size, "arm": arm}
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")

    report["best_arm"] = best_arm
    report["best_span"] = round(best_span, 4)

    if args.oof and best_arm:
        pairs = arms_data[best_arm]
        docs = sorted({L.doc_key(pair) for pair in train_pairs})
        np.random.RandomState(17).shuffle(docs)
        fold_of = {doc: index % args.oof for index, doc in enumerate(docs)}
        real_tables = S.build_tables(train_pairs, L.weights(), ROOT / "artifacts/cache_v16/real_tables.json")
        config = (report["arms"][best_arm]["val"] or {}).get("config")
        folds = []
        for fold in range(args.oof):
            held = [pair for pair in train_pairs if fold_of[L.doc_key(pair)] == fold]
            train_fold = [pair for pair in train_pairs if fold_of[L.doc_key(pair)] != fold]
            extra = [pair for pair in v19] if "v19" in best_arm else []
            model = build_model(model_dir=args.encoder or None)
            freeze_lower_layers(model, args.freeze_layers)
            model.load_state_dict(torch.load(ROOT / args.checkpoint, map_location="cpu")["state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 2, weight_decay=0.01)
            fold_tensors, _ = pack(train_fold + extra, tokenizer, args.doc_tokens, args.max_len,
                                   args.max_easy)
            for _ in range(args.oof_epochs):
                run_epoch(model, fold_tensors, optimizer, args.batch)
            held_tensors, held_meta = pack(held, tokenizer, args.doc_tokens, args.max_len,
                                           max_easy=-1)
            held_scores = scores(model, held_tensors)
            held_tables = [table for table in real_tables
                           if table["pair_id"] in {pair["pair_id"] for pair in held}]
            apply_to_tables(held_tables, held_meta, held_scores)
            cell = L.cell(S.evaluate(held, held_tables, config, "risk_ce"))
            cell.update({"fold": fold, "pairs": len(held)})
            folds.append(cell)
            print(f"  oof fold {fold}: span {cell['span']:.4f} pair {cell['pair']:.4f}", flush=True)
        mean = {key: round(float(np.mean([row[key] for row in folds])), 4)
                for key in ("span", "token", "pair", "fpr")}
        report["oof"] = {"folds": folds, "mean": mean, "baseline_span": 0.2437,
                         "required": 0.2637, "pass": bool(mean["span"] >= 0.2637)}

    val_cell = (report["arms"].get(best_arm) or {}).get("val") or {}
    report["guard"] = {
        "span": {"value": val_cell.get("span"), "need": 0.50,
                 "pass": bool((val_cell.get("span") or 0) >= 0.50)},
        "token": {"value": val_cell.get("token"), "need": 0.50,
                  "pass": bool((val_cell.get("token") or 0) >= 0.50)},
        "pair": {"value": val_cell.get("pair"), "need": 0.50,
                 "pass": bool((val_cell.get("pair") or 0) >= 0.50)},
        "fpr": {"value": val_cell.get("fpr"), "need": 0.40,
                "pass": bool((val_cell.get("fpr") or 1) <= 0.40)},
        "oof": {"value": (report.get("oof") or {}).get("mean", {}).get("span"), "need": 0.2637,
                "pass": bool((report.get("oof") or {}).get("pass"))},
    }
    report["guard"]["pass_all"] = all(item["pass"] for item in report["guard"].values())
    report["verdict"] = "candidate" if report["guard"]["pass_all"] else "reject_by_guard"
    report["total_seconds"] = round(time.time() - started_all, 1)
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best", best_arm, report["best_span"], json.dumps(report["guard"], ensure_ascii=False),
          report["verdict"], f"{report['total_seconds']} c", flush=True)


if __name__ == "__main__":
    main()
