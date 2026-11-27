"""V21: как использовать обученный cross-encoder в метриках договора.

Варианты:
  * `ce_only` — риск CE заменяет штатный риск кандидата (декадер v11 сохранён: mode-гейт,
    pair-гейт, local_hgb, NMS), пороги калибруются на val в бюджете FPR <= 0.40;
  * `blend`   — квантильная смесь риска CE со штатным риском v11, вес подбирается на val
    (как в v20 `blend_v20.py`), baseline при w = 0 воспроизводится точно.

Почему не используем голову start/end для уточнения границ: gold-спаны в v3-разметке
(правило C1) — это ровно минимальный фрагмент из `candidate_spans`, поэтому границы кандидата
и gold совпадают; уточнение границ может только испортить IoU. Голова start/end полезна как
вспомогательный сигнал обучения, в декодировании не нужна.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from sklearn.metrics import roc_auc_score

import sweep_v16 as S
import v11_lab as L
import ce_train_cpu_v21 as C
from ce_model_v21 import build_model, load_tokenizer


def rank_norm(values):
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks / max(len(order) - 1, 1)


def quantile_map(base, scores):
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    mapped = np.empty(len(order), dtype=np.float64)
    mapped[order] = np.sort(np.asarray(base, dtype=np.float64))
    return mapped


def with_risk(pairs, tables, meta, values, risk_key="risk_new"):
    for table in tables:
        for row in table["rows"]:
            row[risk_key] = row.get("risk_cand", 0.0)
    C.apply_to_tables(tables, meta, values, risk_key)
    return [{**L.cell(S.evaluate(pairs, tables, config, risk_key)), "config": config}
            for config in S.grid()]


def best_feasible(rows):
    feasible = [row for row in rows if row["fpr"] <= 0.40]
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    if not feasible:
        return None
    best = feasible[0]
    return {key: best[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")} | \
        {"config": best["config"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/ce_model_cpu_v21.pt")
    parser.add_argument("--report", default="artifacts/ce_decode_v21.json")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--doc-tokens", type=int, default=64)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args(argv)

    tokenizer = load_tokenizer()
    train_pairs, val_pairs, _ = L.load_all()
    model = build_model()
    model.load_state_dict(__import__("torch").load(ROOT / args.checkpoint, map_location="cpu")["state_dict"])
    model.eval()

    tensors, meta = C.pack(val_pairs, tokenizer, args.doc_tokens, args.max_len, max_easy=-1)
    values = C.scores(model, tensors)
    labels = [entry["label"] for entry in meta]
    tables = S.build_tables(val_pairs, L.weights(), ROOT / "artifacts/cache_v16/val_tables.json")
    base = []
    by_id = {table["pair_id"]: table for table in tables}
    for entry in meta:
        row = next(row for row in by_id[entry["pair_id"]]["rows"]
                   if row["start"] == entry["start"] and row["end"] == entry["end"])
        base.append(row["risk_cand"])
    base = np.asarray(base, dtype=np.float64)

    report = {
        "rule": "v21: применение cross-encoder к метрикам договора. Пороги калибруются на val в "
                "бюджете FPR <= 0.40, held-out не читается",
        "checkpoint": args.checkpoint,
        "val": {"candidates": len(meta), "positives": int(sum(labels)),
                "ce_auc": round(float(roc_auc_score(labels, values)), 4),
                "v11_auc": round(float(roc_auc_score(labels, base)), 4)},
        "variants": {},
    }

    report["variants"]["ce_only"] = best_feasible(with_risk(val_pairs, tables, meta, values))

    # пара-гейт из CE: max(risk CE) по кандидатам пары, приведённый квантильно к шкале pair_prob v11
    def apply_pair_gate(tables, meta, values, mode="max"):
        by_id = {table["pair_id"]: table for table in tables}
        per_pair = {}
        for index, entry in enumerate(meta):
            per_pair.setdefault(entry["pair_id"], []).append(values[index])
        raw = {pair_id: (max(scores_) if mode == "max" else float(np.mean(sorted(scores_)[-3:])))
               for pair_id, scores_ in per_pair.items()}
        current = np.asarray([table.get("pair_prob", 0.0) for table in tables], dtype=np.float64)
        order = np.argsort(np.asarray([raw.get(table["pair_id"], 0.0) for table in tables]))
        mapped = np.empty(len(tables), dtype=np.float64)
        mapped[order] = np.sort(current)
        for index, table in enumerate(tables):
            table["pair_prob"] = float(mapped[index])
        return by_id

    original_pair_prob = [table["pair_prob"] for table in tables]

    def restore_pair_prob():
        for table, value in zip(tables, original_pair_prob):
            table["pair_prob"] = value

    for mode in ("max", "top3"):
        apply_pair_gate(tables, meta, values, mode)
        report["variants"][f"ce_pair_gate_{mode}_v11_cand"] = best_feasible(
            with_risk(val_pairs, tables, meta, base))
        print(f"ce_pair_gate_{mode}", report["variants"][f"ce_pair_gate_{mode}_v11_cand"], flush=True)
        restore_pair_prob()
    print("ce_only", report["variants"]["ce_only"], flush=True)

    restore_pair_prob()
    mapped = quantile_map(base, values)
    blend_rows = []
    for weight in np.round(np.arange(0.0, 1.0001, args.step), 2):
        mixed = (1 - weight) * base + weight * mapped
        cell = best_feasible(with_risk(val_pairs, tables, meta, mixed))
        if cell is None:
            continue
        blend_rows.append({"weight_ce": round(float(weight),
                                             2),
                           "auc": round(float(roc_auc_score(labels, mixed)), 4), **cell})
    report["variants"]["blend_grid"] = blend_rows
    best_blend = max(blend_rows, key=lambda row: (row["span"], -row["fpr"]))
    report["variants"]["blend_best"] = best_blend
    print("blend_best", best_blend, flush=True)

    guard_span = max([entry["span"] for key, entry in report["variants"].items()
                      if isinstance(entry, dict) and "span" in entry] or [0.0])
    report["guard_partial"] = {"best_val_span": guard_span, "need": 0.50,
                               "pass": bool(guard_span >= 0.50)}
    report["verdict"] = "span_candidate" if guard_span >= 0.50 else "span_not_reached"
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best val span", guard_span, report["verdict"], flush=True)


if __name__ == "__main__":
    main()
