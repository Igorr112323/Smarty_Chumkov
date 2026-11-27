"""V20 шаг 3–4: смесь штатного риска v11 с ранжировщиком (квантильное выравнивание шкал)
и честный OOF-замер переноса v19 на реальные пары.

Проблема наивной смеси: скоры модели живут в другой шкале, а пороги декадера v11 калиброваны
под `risk_cand`. Поэтому скор модели переводится в распределение `risk_cand` (квантильное
выравнивание по рангу), после чего смесь `(1-w)*base + w*mapped` остаётся в той же шкале:
при w = 0 воспроизводится ровно baseline v11 (Span 0.3621).

Проверяется: (1) воспроизводимость baseline, (2) сетка w × конфиги декадера в бюджете FPR ≤ 0.40,
(3) OOF по документам на реальных парах для «без v19» и «с v19» при одном конфиге.
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
from sklearn.metrics import roc_auc_score

import sweep_v16 as S
import v11_lab as L
import train_rank_v20 as T

REFERENCE = {"span": 0.3621, "token": 0.6247, "pair": 0.5122, "fpr": 0.1442}
REFERENCE_CONFIG = {"mode_cut": 0.6, "gate": 0.4, "threshold": 0.6, "max_spans": 1,
                    "local_threshold": 0.5, "local_max_spans": 3, "nms_iou": 0.1}


def load_groups():
    L.set_weights(L.weights())
    train, val, mfava = L.load_all()
    groups = {}
    plan = (("real", train, T.CACHE20 / "real.npz"),
            ("synth", T.read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl"), T.CACHE20 / "synth.npz"),
            ("pseudo", T.read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl"), T.CACHE20 / "pseudo.npz"),
            ("v19", T.read_jsonl(ROOT / "artifacts/ru_clause_gen_v19.jsonl"), T.CACHE20 / "v19.npz"),
            ("val", val, T.CACHE20 / "val.npz"),
            ("mfava", mfava, T.CACHE20 / "mfava.npz"))
    for name, pairs, cache in plan:
        groups[name] = T.build_group(name, pairs, cache)
    return groups


def quantile_map(base, scores):
    """Перевод скоров модели в распределение base (по рангу)."""
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    ordered_base = np.sort(np.asarray(base, dtype=np.float64))
    mapped = np.empty(len(order), dtype=np.float64)
    mapped[order] = ordered_base
    return mapped


def attach(groups, tables, values):
    data = groups["val"]
    index = 0
    by_id = {table["pair_id"]: table for table in tables}
    for entry in data["meta"]:
        table = by_id.get(entry["pair_id"])
        if table is None:
            index += entry["n"]
            continue
        for row in table["rows"]:
            row["risk_new"] = float(values[index])
            index += 1


def best_cell(groups, val, tables, values):
    attach(groups, tables, values)
    rows = [{**L.cell(S.evaluate(val, tables, config, "risk_new")), "config": config}
            for config in S.grid()]
    feasible = [row for row in rows if row["fpr"] <= 0.40]
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    best = feasible[0] if feasible else None
    if best is None:
        return None
    return {key: best[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")} | \
        {"config": best["config"]}


def sweep_variant(groups, tables, val, label, model, kind, base, weights_grid):
    model_scores = T.score_rows(model, groups["val"]["X"], kind)
    mapped = quantile_map(base, model_scores)
    y = groups["val"]["y"].astype(int)
    rows = []
    for weight in weights_grid:
        values = (1 - weight) * base + weight * mapped
        cell = best_cell(groups, val, tables, values)
        if cell is None:
            continue
        rows.append({"weight_model": round(float(weight), 2),
                     "auc": round(float(roc_auc_score(y, values)), 4), **cell})
        print(f"  {label} w={weight:.2f} auc={rows[-1]['auc']} "
              f"span={cell['span']:.4f} pair={cell['pair']:.4f} fpr={cell['fpr']:.4f}", flush=True)
    best = max(rows, key=lambda row: (row["span"], -row["fpr"]))
    return {"variant": label, "kind": kind, "grid": rows, "best": best}


def oof_transfer(groups, label, parts, config, weight, folds=3, seed=17):
    """OOF по документам на реальных парах: как v19 в обучении влияет на реальные пары."""
    data = groups["real"]
    docs = sorted({L.doc_key(pair) for pair in data["pairs"]})
    np.random.RandomState(seed).shuffle(docs)
    fold_of = {doc: index % folds for index, doc in enumerate(docs)}
    by_id = {pair["pair_id"]: pair for pair in data["pairs"]}
    row_fold = []
    for entry in data["meta"]:
        row_fold.extend([fold_of[L.doc_key(by_id[entry["pair_id"]])]] * entry["n"])
    row_fold = np.asarray(row_fold)
    Xs, ys = T.stack(groups, [part for part in parts if part != "real"])
    tables = S.build_tables(data["pairs"], L.weights(), T.CACHE16 / "real_tables.json")
    cells = []
    for fold in range(folds):
        train_mask = row_fold != fold
        held_mask = ~train_mask
        if train_mask.sum() == 0 or held_mask.sum() == 0:
            continue
        model = T.fit_hgb(np.vstack([Xs, data["X"][train_mask]]),
                          np.concatenate([ys, data["y"][train_mask]]))
        scores = model.predict_proba(data["X"][held_mask])[:, 1]
        held_ids = {pair["pair_id"] for pair in data["pairs"]
                    if fold_of[L.doc_key(pair)] == fold}
        held_pairs = [pair for pair in data["pairs"] if pair["pair_id"] in held_ids]
        held_tables = [table for table in tables if table["pair_id"] in held_ids]
        index = 0
        base = np.zeros(int(held_mask.sum()), dtype=np.float64)
        for entry in data["meta"]:
            if entry["pair_id"] not in held_ids:
                continue
            base[index:index + entry["n"]] = [row["risk_cand"] for row in
                                              {table["pair_id"]: table for table in held_tables}
                                              [entry["pair_id"]]["rows"]]
            index += entry["n"]
        values = (1 - weight) * base + weight * quantile_map(base, scores)
        index = 0
        by_table = {table["pair_id"]: table for table in held_tables}
        for entry in data["meta"]:
            if entry["pair_id"] not in held_ids:
                continue
            table = by_table[entry["pair_id"]]
            for row in table["rows"]:
                row["risk_new"] = float(values[index])
                index += 1
        cell = L.cell(S.evaluate(held_pairs, held_tables, config, "risk_new"))
        cell.update({"fold": fold, "pairs": len(held_pairs)})
        cells.append(cell)
        print(f"  oof {label} fold {fold}: span={cell['span']:.4f} pair={cell['pair']:.4f} "
              f"fpr={cell['fpr']:.4f}", flush=True)
    mean = {key: round(float(np.mean([cell[key] for cell in cells])), 4)
            for key in ("span", "token", "pair", "fpr")} if cells else None
    return {"variant": label, "parts": parts, "config": config, "weight_model": weight,
            "folds": cells, "mean": mean}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/blend_v20.json")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--skip-oof", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    groups = load_groups()
    train, val, _ = L.load_all()
    tables_val = S.build_tables(val, L.weights(), T.CACHE16 / "val_tables.json")
    base = T.base_risks(groups, tables_val)
    print(f"группы готовы ({time.time() - started:.0f} c)", flush=True)

    reference = best_cell(groups, val, tables_val, base)
    print("reference v11:", reference, flush=True)

    specs = (("control_37", ["real", "synth", "pseudo"], False, "hgb"),
             ("v19_plus_real", ["real", "synth", "pseudo", "v19"], False, "hgb"),
             ("v19_plus_real_hardneg", ["real", "synth", "pseudo", "v19"], True, "hgb"),
             ("full_pairwise", ["real", "synth", "pseudo", "v19"], True, "pairwise"))
    weights_grid = np.round(np.arange(0.0, 1.0001, args.step), 2)
    results = {}
    models = {}
    for label, parts, use_hard, kind in specs:
        if kind == "pairwise":
            X, y, weights = T.pairwise_dataset(groups, parts)
            model = __import__("sklearn.linear_model", fromlist=["LogisticRegression"]) \
                .LogisticRegression(max_iter=3000, C=0.5).fit(X, y, sample_weight=weights)
        else:
            X, y = T.stack(groups, parts)
            weight = None
            if use_hard:
                weight = np.concatenate([T.hard_weights(groups[part])[0] for part in parts])
            model = T.fit_hgb(X, y, weight)
        models[label] = (model, kind)
        results[label] = sweep_variant(groups, tables_val, val, label, model, kind, base,
                                       weights_grid)

    best_variant = max(results, key=lambda key: (results[key]["best"]["span"],
                                                 -results[key]["best"]["fpr"]))
    report = {
        "rule": "смесь штатного риска v11 с ранжировщиком через квантильное выравнивание шкал; "
                "пороги декадера калибруются на val в бюджете FPR ≤ 0.40; held-out не читается",
        "reference_v11_reproduced": reference,
        "reference_v11_expected": REFERENCE,
        "weights_grid": [float(value) for value in weights_grid],
        "variants": results,
        "best": {"variant": best_variant, **results[best_variant]["best"]},
    }
    print("best blend:", best_variant, results[best_variant]["best"], flush=True)

    if not args.skip_oof:
        best_cfg = results[best_variant]["best"]["config"]
        best_w = results[best_variant]["best"]["weight_model"]
        report["oof"] = {
            "config": best_cfg,
            "weight_model": best_w,
            "without_v19": oof_transfer(groups, "without_v19", ["real", "synth", "pseudo"],
                                        best_cfg, best_w),
            "with_v19": oof_transfer(groups, f"with_v19_{best_variant}",
                                     dict((label, parts) for label, parts, _, _ in specs)[best_variant],
                                     best_cfg, best_w),
        }
        without = report["oof"]["without_v19"]["mean"]
        with_ = report["oof"]["with_v19"]["mean"]
        report["oof"]["delta_span"] = round((with_ or {}).get("span", 0) - (without or {}).get("span", 0), 4)
        report["oof"]["baseline_oof_span_v14"] = T.BASELINE_OOF_SPAN
        report["oof"]["pass"] = bool(with_ and with_["span"] >= T.BASELINE_OOF_SPAN + 0.02)
        print("oof", report["oof"]["without_v19"]["mean"], report["oof"]["with_v19"]["mean"],
              "delta", report["oof"]["delta_span"], flush=True)

    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"готово за {time.time() - started:.0f} c", flush=True)


if __name__ == "__main__":
    main()
