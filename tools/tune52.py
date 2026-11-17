"""Стадия 3, шаг 1: перенос выбора порога внутрь обучения.

Проблема, которую закрывает скрипт: сохранённые `inner_grid` из ablation51
считались при фиксированном `gate = 0.45`, поэтому подобрать gate вместе с
порогом по ним нельзя, а пресет `artifacts/tiny_head.json` берёт порог от
ансамбля пяти сидов, хотя в поставку идёт один сид.

Что делает `tune`:
  * берёт пары, на которых модель обучалась (train минус внутренний фолд),
    и внутренний фолд, который тоже не видел;
  * считает вероятности ОДНИМ сидом на обоих наборах;
  * перебирает (gate, start, extend) на внутреннем фолде при ограничениях
    FPR <= 0.40, выбирает максимум token F1;
  * проверяет, что выбранная точка не подгоняет FPR на обучающих парах
    (расхождение выводится отдельно, но не блокирует выбор).

`tune` требует только инференса — обучение не запускается.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import lexfeat
from spanver.decode import spans_from_flags
from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from tools.evaluate_snap42 import compute
from tools.predict_frozen38 import load_state
from tools.train_base33 import grouped_folds

GATES = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
STARTS = (0.40, 0.45, 0.50, 0.55, 0.60)
RATIOS = (0.5, 0.75, 1.0)
FPR_LIMIT = 0.40
MAX_SPANS = 3


def decode(pieces, answer, start, extend, gate, max_spans=MAX_SPANS):
    if not pieces:
        return []
    if max(piece[2] for piece in pieces) < gate:
        return []
    flags = []
    active = False
    for _, _, score in pieces:
        active = score >= (extend if active else start)
        flags.append(active)
    spans = spans_from_flags(answer, pieces, flags, 2)
    ordered = sorted(snapped(spans, answer, "expand"), key=lambda item: (-item["risk"], item["start"]))
    return ordered[:max_spans]


def probabilities(tokenizer, model, pairs, top):
    cache = {}
    for pair in pairs:
        value = lexfeat.pair_probabilities(tokenizer, model, pair, top=top)
        cache[pair["pair_id"]] = value["pieces"] if value and value["pieces"] else []
    return cache


def evaluate(pairs, cache, start, extend, gate):
    rows = [{
        "pair_id": pair["pair_id"],
        "answer": pair["model_output"],
        "gold_spans": pair["spans"],
        "predicted_spans": decode(cache[pair["pair_id"]], pair["model_output"], start, extend, gate),
    } for pair in pairs]
    return compute(rows)


def sweep(pairs, cache):
    table = []
    for gate in GATES:
        for start in STARTS:
            for ratio in RATIOS:
                result = evaluate(pairs, cache, start, start * ratio, gate)
                table.append({
                    "gate": gate,
                    "start": start,
                    "extend": round(start * ratio, 3),
                    "token_f1": result["token_f1"],
                    "fpr_token": result["fpr_token"],
                    "fpr_pair": result["fpr_pair"],
                    "iou_f1": result["iou_f1"],
                    "type_accuracy": result["type_accuracy"],
                })
    allowed = [row for row in table if row["fpr_token"] <= FPR_LIMIT and row["fpr_pair"] <= FPR_LIMIT]
    best = max(allowed, key=lambda row: row["token_f1"]) if allowed else None
    return table, best, len(allowed)


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--out", default="artifacts/tune52.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    folds = grouped_folds(pairs, 5)
    held = folds[args.outer - 1]
    inner = folds[args.inner - 1]
    train = [row for index, bucket in enumerate(folds) if index != args.outer - 1 for row in bucket]
    train = [row for row in train if row["pair_id"] not in {inner_pair["pair_id"] for inner_pair in inner}]

    manifest = json.loads((ROOT / "weights/final/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["checkpoints"][0]
    if entry["seed"] != args.seed:
        raise ValueError("манифест должен начинаться с запрошенного сида")
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", labels=2, seed=entry["seed"])
    model.base.config._attn_implementation = "eager"
    load_state(torch, model, ROOT / "weights/final" / f"seed_{entry['seed']}", entry)

    inner_cache = probabilities(tokenizer, model, inner, args.top)
    train_cache = probabilities(tokenizer, model, train, args.top)
    table, best, allowed_count = sweep(inner, inner_cache)
    if best is None:
        raise SystemExit("ни одна точка сетки не прошла ограничения FPR")

    fit_check = evaluate(train, train_cache, best["start"], best["extend"], best["gate"])
    report = {
        "status": "готово",
        "scope": "один сид seed 0, только инференс; внутренний фолд выбран до просмотра метрик внешнего",
        "outer_fold": args.outer,
        "inner_fold": args.inner,
        "pairs_inner": len(inner),
        "pairs_train": len(train),
        "grid_points": len(table),
        "allowed_points": allowed_count,
        "best": best,
        "train_fit_check": {
            "token_f1": fit_check["token_f1"],
            "fpr_token": fit_check["fpr_token"],
            "fpr_pair": fit_check["fpr_pair"],
        },
        "table": table,
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "allowed": allowed_count, "grid": len(table)}, ensure_ascii=False, indent=1))
    print("на обучающих парах:", json.dumps(report["train_fit_check"], ensure_ascii=False))


if __name__ == "__main__":
    main()
