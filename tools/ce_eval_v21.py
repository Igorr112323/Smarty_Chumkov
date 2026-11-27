"""V21 шаг 5: проверка cross-encoder на наших метриках договора (val + OOF по документам).

Что делает:
  * берёт чекпойнт cross-encoder (`artifacts/ce_model_v21.pt`) и считает risk-скор по кандидатам
    val-пар (AUC + within-pair accuracy);
  * подставляет скоры в декадер v11 (как в v17/v20) и калибрует пороги на val в бюджете FPR <= 0.40
    → Span / Token / Pair / FPR;
  * при `--oof N` честно переобучает модель по фолдам документов на реальных парах и считает OOF.

Без GPU (`torch.cuda.is_available() = False`) скрипт фиксирует блокер и не выдумывает метрики.
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
import torch
from sklearn.metrics import roc_auc_score

import sweep_v16 as S
import v11_lab as L
from ce_model_v21 import build_model, encode_batch, load_tokenizer
from ce_train_v21 import balanced_batches, flatten, read_jsonl, train_step


def candidate_examples(pairs, tokenizer, doc_tokens, max_len, sources=None):
    """Кандидаты всех пар в формате CE + ссылки (pair_id, start, end)."""
    from spanver.module import candidate_spans
    items, index = [], []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        gold = pair.get("spans") or []
        from spanver.evaluate import iou
        for start, end, fragment in candidate_spans(answer):
            if not fragment.strip():
                continue
            label = 1.0 if any(iou(start, end, span["start"], span["end"]) >= 0.5 for span in gold) else 0.0
            items.append((document, fragment, None, label))
            index.append({"pair_id": pair["pair_id"], "start": start, "end": end})
    return items, index


def scores(model, items, tokenizer, device, batch=16, max_len=384, doc_tokens=200):
    model.eval()
    result = []
    with torch.no_grad():
        for offset in range(0, len(items), batch):
            chunk = items[offset:offset + batch]
            tensors, start_targets, end_targets, risk_targets = encode_batch(
                tokenizer, chunk, doc_tokens=doc_tokens, max_len=max_len)
            tensors = {key: value.to(device) for key, value in tensors.items()}
            outputs = model(**tensors)
            result.append(torch.sigmoid(outputs["risk_logit"]).cpu().numpy())
    return np.concatenate(result) if result else np.zeros(0)


def apply_scores(tables, index, values, risk_key="risk_ce"):
    by_id = {table["pair_id"]: table for table in tables}
    for row_index, entry in enumerate(index):
        table = by_id.get(entry["pair_id"])
        if table is None:
            continue
        for row in table["rows"]:
            if row["start"] == entry["start"] and row["end"] == entry["end"]:
                row[risk_key] = float(values[row_index])
                break


def contract_cell(pairs, tables, risk_key="risk_ce"):
    rows = [{**L.cell(S.evaluate(pairs, tables, config, risk_key)), "config": config}
            for config in S.grid()]
    feasible = [row for row in rows if row["fpr"] <= 0.40]
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    return feasible[0] if feasible else None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/span_oof_v21.json")
    parser.add_argument("--checkpoint", default="artifacts/ce_model_v21.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--doc-tokens", type=int, default=200)
    parser.add_argument("--oof", type=int, default=0, help="число фолдов OOF (0 — не считать)")
    parser.add_argument("--epochs", type=int, default=1, help="эпох на фолд OOF")
    args = parser.parse_args(argv)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if (cuda and args.device != "cpu") else "cpu")
    checkpoint = ROOT / args.checkpoint
    report = {
        "rule": "шаг 5 v21: метрики договора для cross-encoder (val + OOF по документам), "
                "пороги калибруются на val в бюджете FPR <= 0.40, held-out не читается",
        "device": str(device), "cuda_available": cuda,
        "checkpoint": {"path": args.checkpoint, "exists": checkpoint.exists(),
                       "bytes": checkpoint.stat().st_size if checkpoint.exists() else 0},
    }

    if not checkpoint.exists():
        report["verdict"] = ("blocked_no_gpu: обученного чекпойнта нет "
                             "(в песочнице нет GPU → ce_train_v21.py не мог обучить модель). "
                             "Метрики val/OOF не подменяются — их просто нет")
        report["reference_v20"] = {"val": {"span": 0.3670, "token": 0.6453, "pair": 0.5195,
                                           "fpr": 0.1058},
                                   "oof_span": 0.2295, "oof_required": 0.2637}
        report["how_to_finish_on_gpu"] = [
            "python tools/ce_train_v21.py --device cuda --epochs 3 --batch 32 --amp",
            "python tools/ce_eval_v21.py --device cuda --oof 3 --epochs 1",
            "python tools/ce_export_v21.py --checkpoint artifacts/ce_model_v21.pt",
            "python tools/ce_speed_v21.py",
        ]
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        print(report["verdict"], flush=True)
        return

    tokenizer = load_tokenizer()
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])

    train_pairs, val_pairs, mfava = L.load_all()
    tables = S.build_tables(val_pairs, L.weights(), ROOT / "artifacts/cache_v16/val_tables.json")
    items, index = candidate_examples(val_pairs, tokenizer, args.doc_tokens, args.max_len)
    started = time.time()
    values = scores(model, items, tokenizer, device, args.batch, args.max_len, args.doc_tokens)
    report["val"] = {"candidates": len(items), "seconds": round(time.time() - started, 1),
                     "auc": round(float(roc_auc_score([item[3] for item in items], values)), 4)}
    apply_scores(tables, index, values)
    cell = contract_cell(val_pairs, tables)
    report["val"].update({"contract": {key: cell[key] for key in
                                       ("span", "token", "pair", "fpr")} if cell else None,
                          "config": cell["config"] if cell else None})
    print("val", report["val"]["auc"], report["val"]["contract"], flush=True)

    if args.oof:
        real = train_pairs
        docs = sorted({L.doc_key(pair) for pair in real})
        np.random.RandomState(17).shuffle(docs)
        fold_of = {doc: index % args.oof for index, doc in enumerate(docs)}
        real_tables = S.build_tables(real, L.weights(), ROOT / "artifacts/cache_v16/real_tables.json")
        config = cell["config"] if cell else {"mode_cut": 0.6, "gate": 0.4, "threshold": 0.6,
                                              "max_spans": 1, "local_threshold": 0.5,
                                              "local_max_spans": 3, "nms_iou": 0.1}
        folds = []
        for fold in range(args.oof):
            held = [pair for pair in real if fold_of[L.doc_key(pair)] == fold]
            train_fold = [pair for pair in real if fold_of[L.doc_key(pair)] != fold]
            model_fold = build_model().to(device)
            model_fold.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
            optimizer = torch.optim.AdamW(model_fold.parameters(), lr=2e-5)
            fold_items = flatten([{"document_text": pair.get("document_text") or "",
                                   "candidates": [{"start": start, "end": end, "text": text,
                                                   "label": 1 if any(
                                                       __import__("spanver.evaluate", fromlist=["iou"]).iou(
                                                           start, end, span["start"], span["end"]) >= 0.5
                                                       for span in (pair.get("spans") or [])) else 0}
                                                  for start, end, text in
                                                  __import__("spanver.module", fromlist=["candidate_spans"]).candidate_spans(
                                                      pair.get("model_output") or "")]}
                                  for pair in train_fold])
            for _ in range(args.epochs):
                model_fold.train()
                for chunk in balanced_batches(fold_items, args.batch):
                    tensors, start_targets, end_targets, risk_targets = encode_batch(
                        tokenizer, chunk, doc_tokens=args.doc_tokens, max_len=args.max_len)
                    tensors = {key: value.to(device) for key, value in tensors.items()}
                    loss, _ = train_step(model_fold, tensors, start_targets.to(device),
                                         end_targets.to(device), risk_targets.to(device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_fold.parameters(), 1.0)
                    optimizer.step()
            held_items, held_index = candidate_examples(held, tokenizer, args.doc_tokens, args.max_len)
            held_values = scores(model_fold, held_items, tokenizer, device, args.batch,
                                 args.max_len, args.doc_tokens)
            held_tables = [table for table in real_tables
                           if table["pair_id"] in {pair["pair_id"] for pair in held}]
            apply_scores(held_tables, held_index, held_values)
            fold_cell = L.cell(S.evaluate(held, held_tables, config, "risk_ce"))
            fold_cell.update({"fold": fold, "pairs": len(held)})
            folds.append(fold_cell)
            print("oof fold", fold, {key: round(fold_cell[key], 4) for key in ("span", "pair", "fpr")},
                  flush=True)
        mean = {key: round(float(np.mean([row[key] for row in folds])), 4)
                for key in ("span", "token", "pair", "fpr")}
        report["oof"] = {"folds": folds, "mean": mean, "baseline_span": 0.2437,
                         "required": 0.2637, "pass": bool(mean["span"] >= 0.2637)}

    val_cell = report["val"].get("contract") or {}
    report["guard"] = {
        "span": {"value": val_cell.get("span"), "need": 0.50, "pass": bool((val_cell.get("span") or 0) >= 0.50)},
        "token": {"value": val_cell.get("token"), "need": 0.50, "pass": bool((val_cell.get("token") or 0) >= 0.50)},
        "pair": {"value": val_cell.get("pair"), "need": 0.50, "pass": bool((val_cell.get("pair") or 0) >= 0.50)},
        "fpr": {"value": val_cell.get("fpr"), "need": 0.40, "pass": bool((val_cell.get("fpr") or 1) <= 0.40)},
        "oof": {"value": (report.get("oof") or {}).get("mean", {}).get("span"), "need": 0.2637,
                "pass": bool((report.get("oof") or {}).get("pass"))},
    }
    report["guard"]["pass_all"] = all(item["pass"] for item in report["guard"].values())
    report["verdict"] = "candidate" if report["guard"]["pass_all"] else "reject_by_guard"
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report["guard"], ensure_ascii=False), report["verdict"], flush=True)


if __name__ == "__main__":
    main()
