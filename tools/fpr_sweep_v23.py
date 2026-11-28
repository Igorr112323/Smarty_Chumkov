"""V23: неиспользованные рычаги — полный FPR-бюджет (0.40 вместо 0.096) и мульти-спан декод.

Что делает скрипт:

  1. собирает val-таблицы v11 (признаки кандидатов, режим-гейт, пара-гейт, local_hgb) и скоры
     обученного cross-encoder (v21-конфигурация, tiny2);
  2. строит варианты риска кандидата: `v11` (штатный), `ce`, `blend` (квантильная смесь),
     `rank_*` (ранг внутри пары), `z_*` (z-score внутри пары) — шаг 4 v23 (калибровка по паре);
  3. строит варианты пара-гейта: v11 `pair_prob`, `ce_max`, `ce_top3`, `mix`;
  4. прогоняет расширенную сетку декадера: порог от 0.0 до 0.6, `max_spans` до «все выше порога»,
     gate от 0.0, mode_cut вплоть до отключения local-ветки;
  5. снимает FPR-кривую: для бюджетов 0.10…0.40 выбирает лучший по Span конфиг;
  6. опционально считает OOF по документам (3 фолда) для выбранного конфига;
  7. пишет `artifacts/fpr_sweep_v23.json` и `artifacts/span_oof_v23.json`.

Дисциплина прежняя: пороги калибруются только на val, held-out не читается, метрики не правятся.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from sklearn.metrics import roc_auc_score

import ce_train_cpu_v21 as C
import sweep_v16 as S
import v11_lab as L
from spanver.evaluate import metrics
from spanver.module import confidence, predict_type
from spanver.snap import snapped

BUDGETS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

GRID = {
    "mode_cut": (0.6, 0.7, 1.01),
    "gate": (0.0, 0.2, 0.3, 0.4, 0.5),
    "threshold": (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6),
    "max_spans": (1, 2, 3, 5, 10, 999),
    "local_threshold": (0.3, 0.4, 0.5),
    "local_max_spans": (1, 3, 999),
}


def configs():
    keys = list(GRID)
    for values in itertools.product(*(GRID[key] for key in keys)):
        config = dict(zip(keys, values))
        config["nms_iou"] = 0.1
        yield config


def iou(left, right):
    start = max(left["start"], right["start"])
    end = min(left["end"], right["end"])
    overlap = max(0, end - start + 1)
    union = (left["end"] - left["start"] + 1) + (right["end"] - right["start"] + 1) - overlap
    return overlap / union if union > 0 else 0.0


def decode(table, config, risk_key, answers, snap_mode="none"):
    """Расширенная версия `sweep_v16.decode`: мульти-спан до «все выше порога» + снэп границ."""
    rows = table["rows"]
    if not rows:
        return []
    if table["mode_prob"] >= config["mode_cut"]:
        threshold = config["local_threshold"]
        max_spans = config["local_max_spans"]
        whole_end = rows[0]["end"]
        pool = [row for row in rows if not (row["start"] == 0 and row["end"] == whole_end)]
        key = "risk_local"
    else:
        if table["pair_prob"] < config["gate"]:
            return []
        threshold = config["threshold"]
        max_spans = config["max_spans"]
        pool = rows
        key = risk_key
    order = sorted(range(len(pool)), key=lambda index: (-pool[index][key], pool[index]["start"]))
    accepted = []
    for index in order:
        if pool[index][key] < threshold:
            break
        candidate = pool[index]
        if any(iou(candidate, other) > config["nms_iou"] for other in accepted):
            continue
        accepted.append(candidate)
        if len(accepted) >= max_spans:
            break
    result = [{"start": row["start"], "end": row["end"], "risk": round(row[key], 4),
               "type": predict_type(row["text"], 0.0), "confidence": confidence(row[key])}
              for row in accepted]
    if snap_mode != "none" and result:
        result = snapped(result, answers.get(table["pair_id"], ""), snap_mode)
    return result


def apply_local_risk(tables, risk_local):
    """Подменяет риск local-ветки один раз на вариант риска, а не на каждый конфиг сетки."""
    if risk_local is None:
        return
    for table in tables:
        for row in table["rows"]:
            row["risk_local"] = risk_local.get((table["pair_id"], row["start"], row["end"]),
                                               row.get("risk_local", 0.0))


def evaluate(pairs, tables, config, risk_key, answers=None, snap_mode="none"):
    answers = answers or {}
    picked = {table["pair_id"]: decode(table, config, risk_key, answers, snap_mode) for table in tables}
    return L.cell(metrics(pairs, lambda pair: picked[pair["pair_id"]]))


def rank_within_pairs(meta, values):
    by_pair = {}
    for index, entry in enumerate(meta):
        by_pair.setdefault(entry["pair_id"], []).append(index)
    result = np.zeros(len(meta), dtype=np.float64)
    for indexes in by_pair.values():
        scores = np.asarray([values[index] for index in indexes], dtype=np.float64)
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(len(scores), dtype=np.float64)
        normalised = ranks / max(len(scores) - 1, 1)
        for position, index in enumerate(indexes):
            result[index] = normalised[position]
    return result


def zscore_within_pairs(meta, values):
    by_pair = {}
    for index, entry in enumerate(meta):
        by_pair.setdefault(entry["pair_id"], []).append(index)
    result = np.zeros(len(meta), dtype=np.float64)
    for indexes in by_pair.values():
        scores = np.asarray([values[index] for index in indexes], dtype=np.float64)
        std = scores.std()
        if std < 1e-9:
            result[indexes] = 0.5
            continue
        z = (scores - scores.mean()) / std
        result[indexes] = (z - z.min()) / max(z.max() - z.min(), 1e-9)
    return result


def quantile_map(base, scores):
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    mapped = np.empty(len(order), dtype=np.float64)
    mapped[order] = np.sort(np.asarray(base, dtype=np.float64))
    return mapped


def pair_gate_values(meta, values, mode):
    per_pair = {}
    for index, entry in enumerate(meta):
        per_pair.setdefault(entry["pair_id"], []).append(values[index])
    if mode == "ce_max":
        return {pair_id: float(np.max(scores)) for pair_id, scores in per_pair.items()}
    if mode == "ce_top3":
        return {pair_id: float(np.mean(sorted(scores)[-3:])) for pair_id, scores in per_pair.items()}
    raise ValueError(mode)


def apply_pair_gate(tables, raw):
    current = np.asarray([table.get("pair_prob", 0.0) for table in tables], dtype=np.float64)
    order = np.argsort(np.asarray([raw.get(table["pair_id"], 0.0) for table in tables]))
    mapped = np.empty(len(tables), dtype=np.float64)
    mapped[order] = np.sort(current)
    for index, table in enumerate(tables):
        table["pair_prob"] = float(mapped[index])


def apply_pair_gate_mix(tables, raw, weight=0.5):
    current = np.asarray([table.get("pair_prob", 0.0) for table in tables], dtype=np.float64)
    order = np.argsort(np.asarray([raw.get(table["pair_id"], 0.0) for table in tables]))
    mapped = np.empty(len(tables), dtype=np.float64)
    mapped[order] = np.sort(current)
    for index, table in enumerate(tables):
        table["pair_prob"] = float((1 - weight) * current[index] + weight * mapped[index])



def nms_order(pool, key, nms_iou=0.1):
    """Жадный NMS-порядок не зависит от порога: при пороге t берётся префикс из элементов с risk >= t.
    Считаем его один раз на пару и вариант риска — иначе перебор 7k конфигов идёт часами."""
    order = sorted(range(len(pool)), key=lambda index: (-pool[index][key], pool[index]["start"]))
    accepted = []
    for index in order:
        candidate = pool[index]
        if any(iou(candidate, other) > nms_iou for other in accepted):
            continue
        accepted.append(candidate)
    return [(row[key], row["start"], row["end"]) for row in accepted]


def build_orders(tables, risk_key):
    """Для каждой пары: NMS-порядок кандидатов (риск риска) и порядок local-ветки."""
    result = {}
    for table in tables:
        rows = table["rows"]
        if not rows:
            result[table["pair_id"]] = {"cand": [], "local": [], "mode_prob": 0.0, "pair_prob": 0.0}
            continue
        whole_end = rows[0]["end"]
        local_pool = [row for row in rows if not (row["start"] == 0 and row["end"] == whole_end)]
        result[table["pair_id"]] = {
            "cand": nms_order(rows, risk_key),
            "local": nms_order(local_pool, "risk_local") if local_pool else [],
            "mode_prob": table["mode_prob"], "pair_prob": table["pair_prob"],
        }
    return result


def take(order, threshold, max_spans):
    picked = []
    for risk, start, end in order:
        if risk < threshold:
            break
        picked.append((start, end))
        if len(picked) >= max_spans:
            break
    return picked


def fast_stats(pairs, orders, config):
    """Быстрая версия метрик без token F1: правило счёта повторяет spanver.evaluate.metrics
    (span tp/fp/fn, pair tp/fn/fp по парам, FPR = доля чистых пар с любым предсказанием)."""
    from spanver.evaluate import match_spans

    tp = fp = fn = 0
    pair_tp = pair_fn = pair_fp = 0
    negatives = flagged = 0
    for pair in pairs:
        info = orders[pair["pair_id"]]
        gold = pair.get("spans") or []
        if info["mode_prob"] >= config["mode_cut"]:
            picked = take(info["local"], config["local_threshold"], config["local_max_spans"])
        elif info["pair_prob"] < config["gate"]:
            picked = []
        else:
            picked = take(info["cand"], config["threshold"], config["max_spans"])
        predicted = [{"start": start, "end": end} for start, end in picked]
        matched = 0
        if predicted and gold:
            matched, _, _ = match_spans(predicted, gold)
        tp += matched
        fp += len(predicted) - matched
        if gold:
            fn += len(gold) - matched
            if matched:
                pair_tp += 1
            else:
                pair_fn += 1
        else:
            negatives += 1
            if predicted:
                pair_fp += 1
                flagged += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    span_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pair_precision = pair_tp / (pair_tp + pair_fp) if pair_tp + pair_fp else 0.0
    pair_recall = pair_tp / (pair_tp + pair_fn) if pair_tp + pair_fn else 0.0
    pair_f1 = (2 * pair_precision * pair_recall / (pair_precision + pair_recall)
               if pair_precision + pair_recall else 0.0)
    fpr = flagged / negatives if negatives else 0.0
    return {"span": round(span_f1, 4), "pair": round(pair_f1, 4), "fpr": round(fpr, 4),
            "span_p": round(precision, 4), "span_r": round(recall, 4)}


def sweep(pairs, tables, risk_variants, gate_variants, answers, local_overrides=None):
    grid = list(configs())
    rows = []
    gate_names = ("v11", "ce_max", "ce_top3", "mix")
    for risk_name, risk_key in risk_variants.items():
        for gate_name in gate_names:
            if gate_name == "ce_max":
                apply_pair_gate(tables, gate_variants["ce_max"])
            elif gate_name == "ce_top3":
                apply_pair_gate(tables, gate_variants["ce_top3"])
            elif gate_name == "mix":
                apply_pair_gate_mix(tables, gate_variants["ce_max"])
            apply_local_risk(tables, (local_overrides or {}).get(risk_name))
            orders = build_orders(tables, risk_key)
            # быстрый проход по всей сетке: span/pair/FPR без token F1
            fast_rows = []
            for config in grid:
                stats = fast_stats(pairs, orders, config)
                fast_rows.append({"risk": risk_name, "gate": gate_name, **stats, "config": config})
            feasible = [row for row in fast_rows if row["fpr"] <= 0.40]
            feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
            # точные метрики (с token F1) только для лучших конфигов каждого бюджета
            finalists = feasible[:60]
            for row in finalists:
                cell = evaluate(pairs, tables, row["config"], risk_key, answers=answers)
                row.update(cell)
                rows.append(row)
            best = finalists[0] if finalists else None
            print(f"  {risk_name} × {gate_name}: best span {best['span'] if best else None} "
                  f"(fpr {best['fpr'] if best else None})", flush=True)
            for table, value in zip(tables, gate_variants["v11_values"]):
                table["pair_prob"] = value
    return rows


def select(rows, budget):
    feasible = [row for row in rows if row["fpr"] <= budget + 1e-9]
    if not feasible:
        return None
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    best = feasible[0]
    return {"risk": best["risk"], "gate": best["gate"],
            **{key: best[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")},
            "config": best["config"]}


def oof_ce_variant(risk_key, gate_name, config, tokenizer, args, folds=3):
    train_pairs, val_pairs, _ = L.load_all()
    docs = sorted({L.doc_key(pair) for pair in train_pairs})
    np.random.RandomState(17).shuffle(docs)
    fold_of = {doc: index % folds for index, doc in enumerate(docs)}
    results = []
    for fold in range(folds):
        held = [pair for pair in train_pairs if fold_of[L.doc_key(pair)] == fold]
        train_fold = [pair for pair in train_pairs if fold_of[L.doc_key(pair)] != fold]
        model = C.build_model()
        import torch

        model.load_state_dict(torch.load(ROOT / args.checkpoint, map_location="cpu")["state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 2, weight_decay=0.01)
        tensors, _ = C.pack(train_fold, tokenizer, args.doc_tokens, args.max_len, args.max_easy)
        for _ in range(args.oof_epochs):
            C.run_epoch(model, tensors, optimizer, args.batch)
        held_tensors, held_meta = C.pack(held, tokenizer, args.doc_tokens, args.max_len, max_easy=-1)
        values = C.scores(model, held_tensors)
        labels = [entry["label"] for entry in held_meta]
        tables = S.build_tables(held, L.weights(),
                                ROOT / f"artifacts/cache_v16/oof_v23_fold{fold}.json")
        base = []
        by_id = {table["pair_id"]: table for table in tables}
        for entry in held_meta:
            row = next(row for row in by_id[entry["pair_id"]]["rows"]
                       if row["start"] == entry["start"] and row["end"] == entry["end"])
            base.append(row["risk_cand"])
        base = np.asarray(base, dtype=np.float64)
        gate_values = {"ce_max": pair_gate_values(held_meta, values, "ce_max"),
                       "ce_top3": pair_gate_values(held_meta, values, "ce_top3")}
        C.apply_to_tables(tables, held_meta, values, "risk_new")
        risk_map = {"v11": "risk_cand", "ce": "risk_new",
                    "blend_0.4": "risk_blend", "rank_blend": "risk_rank", "z_blend": "risk_z"}
        mapped = quantile_map(base, values)
        blend = 0.6 * base + 0.4 * mapped
        C.apply_to_tables(tables, held_meta, blend, "risk_blend")
        C.apply_to_tables(tables, held_meta, rank_within_pairs(held_meta, blend), "risk_rank")
        C.apply_to_tables(tables, held_meta, zscore_within_pairs(held_meta, blend), "risk_z")
        row = {"fold": fold, "pairs": len(held), "auc_ce": round(float(roc_auc_score(labels, values)), 4)}
        original = [table["pair_prob"] for table in tables]
        if gate_name in gate_values:
            apply_pair_gate(tables, gate_values[gate_name])
        elif gate_name == "mix":
            apply_pair_gate_mix(tables, gate_values["ce_max"])
        row["ce_variant"] = evaluate(held, tables, config, risk_map[risk_key])
        for table, value in zip(tables, original):
            table["pair_prob"] = value
        row["v11_reference"] = evaluate(held, tables, config, "risk_cand")
        results.append(row)
        print(f"  oof fold {fold}: ce {row['ce_variant']['span']} v11 {row['v11_reference']['span']}",
              flush=True)
    mean = {key: round(float(np.mean([row["ce_variant"][key] for row in results])), 4)
            for key in ("span", "token", "pair", "fpr")}
    return {"folds": results, "mean": mean, "baseline_span": 0.2437, "required": 0.2637,
            "pass": bool(mean["span"] >= 0.2637)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/ce_model_cpu_v23.pt")
    parser.add_argument("--report", default="artifacts/fpr_sweep_v23.json")
    parser.add_argument("--oof-report", default="artifacts/span_oof_v23.json")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--doc-tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-easy", type=int, default=2)
    parser.add_argument("--oof", type=int, default=0)
    parser.add_argument("--oof-epochs", type=int, default=1)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(2)
    tokenizer = C.load_tokenizer()
    train_pairs, val_pairs, _ = L.load_all()
    model = C.build_model()
    model.load_state_dict(torch.load(ROOT / args.checkpoint, map_location="cpu")["state_dict"])
    model.eval()

    tensors, meta = C.pack(val_pairs, tokenizer, args.doc_tokens, args.max_len, max_easy=-1)
    values = C.scores(model, tensors)
    labels = [entry["label"] for entry in meta]
    tables = S.build_tables(val_pairs, L.weights(), ROOT / "artifacts/cache_v16/val_tables.json")
    answers = {pair["pair_id"]: pair.get("model_output") or "" for pair in val_pairs}

    base = []
    by_id = {table["pair_id"]: table for table in tables}
    for entry in meta:
        row = next(row for row in by_id[entry["pair_id"]]["rows"]
                   if row["start"] == entry["start"] and row["end"] == entry["end"])
        base.append(row["risk_cand"])
    base = np.asarray(base, dtype=np.float64)
    mapped = quantile_map(base, values)
    blend = 0.6 * base + 0.4 * mapped

    C.apply_to_tables(tables, meta, values, "risk_new")
    C.apply_to_tables(tables, meta, blend, "risk_blend")
    C.apply_to_tables(tables, meta, rank_within_pairs(meta, blend), "risk_rank")
    C.apply_to_tables(tables, meta, zscore_within_pairs(meta, blend), "risk_z")

    risk_variants = {"v11": "risk_cand", "ce": "risk_new", "blend_0.4": "risk_blend",
                     "rank_blend": "risk_rank", "z_blend": "risk_z",
                     "blend_0.4+local": "risk_blend", "ce+local": "risk_new"}
    local_overrides = {
        "blend_0.4+local": {(entry["pair_id"], entry["start"], entry["end"]): float(blend[index])
                            for index, entry in enumerate(meta)},
        "ce+local": {(entry["pair_id"], entry["start"], entry["end"]): float(values[index])
                     for index, entry in enumerate(meta)},
    }
    gate_variants = {"ce_max": pair_gate_values(meta, values, "ce_max"),
                     "ce_top3": pair_gate_values(meta, values, "ce_top3"),
                     "v11_values": [table["pair_prob"] for table in tables]}

    print("сетка:", len(list(configs())), "конфигов ×", len(risk_variants), "рисков × 3 гейта", flush=True)
    rows = sweep(val_pairs, tables, risk_variants, gate_variants, answers, local_overrides)

    curve = {}
    for budget in BUDGETS:
        curve[f"{budget:.2f}"] = select(rows, budget)
    best = select(rows, 0.40)

    # снэп границ (adjuster) применяем только к выбранному конфигу — это отдельная абляция
    snap_rows = {}
    for mode in ("none", "expand", "shrink", "nearest"):
        snap_rows[mode] = evaluate(val_pairs, tables, best["config"], risk_variants[best["risk"]],
                                   answers=answers, snap_mode=mode)

    # мульти-спан абляция на выбранном риске/гейте
    multi = {}
    for max_spans in (1, 2, 3, 5, 999):
        config = dict(best["config"]) | {"max_spans": max_spans}
        multi[str(max_spans)] = evaluate(val_pairs, tables, config, risk_variants[best["risk"]],
                                         answers=answers)

    report = {
        "rule": ("v23: используем полный FPR-бюджет договора (<= 0.40) и мульти-спан декод. "
                 "Пороги калибруются только на val, held-out не читается, метрики не правятся"),
        "checkpoint": args.checkpoint,
        "val": {"pairs": len(val_pairs), "candidates": len(meta), "positives": int(sum(labels)),
                "ce_auc": round(float(roc_auc_score(labels, values)), 4),
                "v11_auc": round(float(roc_auc_score(labels, base)), 4)},
        "grid": {"size": len(list(configs())), "risks": list(risk_variants),
                 "gates": ["v11", "ce_max", "ce_top3", "mix"],
                 "thresholds": list(GRID["threshold"]), "max_spans": list(GRID["max_spans"])},
        "fpr_curve": curve,
        "best_under_0.40": best,
        "snap_ablation": snap_rows,
        "multi_span_ablation": multi,
        "rows_total": len(rows),
        "rows_top": sorted(rows, key=lambda row: (-row["span"], row["fpr"]))[:40],
        "reference": {"v11_v21": {"span": 0.3621, "token": 0.6247, "pair": 0.5122, "fpr": 0.1442},
                      "ce_v21_best": {"span": 0.3818, "token": 0.6365, "pair": 0.5455, "fpr": 0.0962},
                      "oracle_ceiling_span": 0.6165},
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best under 0.40:", json.dumps(best, ensure_ascii=False), flush=True)
    for budget, row in curve.items():
        print(f"  budget {budget}: span {row['span'] if row else None} fpr {row['fpr'] if row else None}",
              flush=True)

    oof = None
    if args.oof:
        oof = oof_ce_variant(best["risk"], best["gate"], best["config"], tokenizer, args, args.oof)
        print("oof:", json.dumps(oof["mean"], ensure_ascii=False), flush=True)

    guard = {
        "span": {"value": best["span"], "need": 0.50, "pass": bool(best["span"] >= 0.50)},
        "token": {"value": best["token"], "need": 0.50, "pass": bool(best["token"] >= 0.50)},
        "pair": {"value": best["pair"], "need": 0.50, "pass": bool(best["pair"] >= 0.50)},
        "fpr": {"value": best["fpr"], "need": 0.40, "pass": bool(best["fpr"] <= 0.40)},
        "oof": {"value": (oof or {}).get("mean", {}).get("span"), "need": 0.2637,
                "pass": bool((oof or {}).get("pass"))},
    }
    guard["pass_all"] = all(item["pass"] for item in guard.values())
    span_oof = {
        "rule": report["rule"],
        "val": report["val"],
        "ablations": {
            "v11_alone_best_under_0.40": select([row for row in rows if row["risk"] == "v11"], 0.40),
            "ce_alone_best_under_0.40": select([row for row in rows if row["risk"] == "ce"], 0.40),
            "v11_plus_ce_best_under_0.40": best,
            "multi_span": multi,
            "snap_adjuster": snap_rows,
        },
        "fpr_curve": curve,
        "oof": oof,
        "guard": guard,
        "verdict": "candidate" if guard["pass_all"] else "reject_by_guard",
    }
    (ROOT / args.oof_report).write_text(json.dumps(span_oof, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print("guard:", json.dumps(guard, ensure_ascii=False), span_oof["verdict"], flush=True)
    return report


if __name__ == "__main__":
    main()
