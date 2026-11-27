"""V16 (доп. эксперимент): относительный риск кандидата внутри пары.

Диагностика показала: у **каждого** gold-спана на val есть кандидат с IoU = 1.0 (генерация
идеальна), но только 20/54 gold-спанов стоят на первом месте по абсолютному риску v11.
Значит узкое место — выбор кандидата ВНУТРИ пары, а не порог по абсолютному риску.

Поэтому: к 37 штатным признакам добавляются z-оценки внутри пары, ранг по риску v11 и отступ
от лучшего кандидата; HGB учится на новых русских данных (`artifacts/sber_flip_v16.jsonl`,
doc_overlap = 0) + v11-смеси. Декодер v11 (mode-гейт, pair-гейт, local_hgb) сохраняется,
калибруются только пороги. Held-out не читается.
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
from sklearn.ensemble import HistGradientBoostingClassifier

import sweep_v16 as S
import v11_lab as L
from spanver.evaluate import metrics
from train_pair_classifier_v12 import export_trees, tree_probability

CACHE = ROOT / "artifacts/cache_v16"


def relative_features(X, meta, base_risk=None, start=0):
    """Дополнительные признаки: z-оценки по паре, ранг и отступ от лучшего."""
    blocks = []
    offset = 0
    for entry in meta:
        n = entry["n"]
        if n == 0:
            continue
        block = X[offset:offset + n]
        mean = block.mean(0)
        std = block.std(0) + 1e-6
        z = (block - mean) / std
        if base_risk is not None:
            risk = np.asarray(base_risk[offset:offset + n], dtype=np.float64)
        else:
            risk = block[:, 1]  # support как прокси, если риск не передан
        order = np.argsort(-risk)
        rank = np.empty(n)
        rank[order] = np.arange(n)
        margin = risk - risk.max()
        extra = np.stack([rank / max(n - 1, 1), margin, np.full(n, np.log1p(n))], axis=1)
        blocks.append(np.hstack([z, extra]))
        offset += n
    return np.vstack(blocks) if blocks else np.zeros((0, 0), dtype=np.float32)


def table_relative(tables):
    rows = []
    for table in tables:
        local_risk = [row["risk_cand"] for row in table["rows"]]
        if not table["rows"]:
            continue
        X = np.asarray([row["features"] for row in table["rows"]], dtype=np.float32)
        rel = relative_features(X, [{"n": X.shape[0]}], local_risk)
        rows.append(rel)
    return np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def fit(X, y, trees, leaves, lr):
    model = HistGradientBoostingClassifier(
        max_iter=trees, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=10,
        l2_regularization=1.0, random_state=16,
    ).fit(X, y)
    return export_trees(model)


def attach_risk(tables, model):
    for table in tables:
        if not table["rows"]:
            continue
        X = np.asarray([row["features"] for row in table["rows"]], dtype=np.float32)
        risk = [row["risk_cand"] for row in table["rows"]]
        rel = relative_features(X, [{"n": X.shape[0]}], risk)
        for index, row in enumerate(table["rows"]):
            row["risk_new"] = tree_probability(model, list(X[index]) + list(rel[index]))
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/span_oof_v16.json")
    parser.add_argument("--model", default="artifacts/span_model_v16.json")
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--fpr-max", type=float, default=0.40)
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    train, val, mfava = L.load_all()
    cache = CACHE

    groups = {}
    for name, path in (("sber", "sber37.npz"), ("synth", "synth37.npz"),
                       ("pseudo", "pseudo37.npz"), ("real", "real37.npz"),
                       ("val", "val37.npz"), ("mfava", "mfava37.npz")):
        data = np.load(cache / path, allow_pickle=True)
        groups[name] = (data["X"], data["y"], json.loads(str(data["meta"])))
        print(name, groups[name][0].shape, int(groups[name][1].sum()), flush=True)

    tables = {}
    for name, pairs in (("val", val), ("mfava", mfava), ("real", train)):
        tables[name] = S.build_tables(pairs, params, cache / f"{name}_tables.json")

    def matrices(parts):
        X_parts, y_parts, rel_parts = [], [], []
        for part in parts:
            X, y, meta = groups[part]
            X_parts.append(X)
            y_parts.append(y)
            rel_parts.append(relative_features(X, meta))
        return np.vstack(X_parts), np.concatenate(y_parts), np.vstack(rel_parts)

    variants = {
        "rel_sber+synth+pseudo+real": ["sber", "synth", "pseudo", "real"],
        "rel_sber+real": ["sber", "real"],
        "rel_sber+synth+pseudo": ["sber", "synth", "pseudo"],
    }
    report = {
        "rule": "относительный риск внутри пары (z-оценки 37 признаков, ранг v11, отступ) + "
                "новые русские данные; декодер v11 сохранён; калибровка порогов на val в "
                "бюджете FPR ≤ 0.40",
        "features": "37 штатных + 40 относительных (37 z + ранг + отступ + log n)",
        "baseline_v11": {"span": 0.3363, "token": 0.6191, "pair": 0.4872, "fpr": 0.125},
        "variants": {},
    }
    best = None
    models = {}
    for name, parts in variants.items():
        X, y, rel = matrices(parts)
        started = time.time()
        model = fit(np.hstack([X, rel]), y, args.trees, args.leaves, args.lr)
        models[name] = model
        fit_seconds = time.time() - started
        attach_risk(tables["val"], model)
        attach_risk(tables["mfava"], model)
        rows = []
        for config in S.grid():
            cell = L.cell(S.evaluate(val, tables["val"], config, "risk_new"))
            rows.append({**cell, "config": config})
        feasible = [row for row in rows if row["fpr"] <= args.fpr_max]
        feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
        top = feasible[0] if feasible else None
        if top is None:
            continue
        mfava_cell = L.cell(S.evaluate(mfava, tables["mfava"], top["config"], "risk_new"))
        report["variants"][name] = {
            "parts": parts, "fit_seconds": round(fit_seconds, 1),
            "val": {key: top[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")},
            "config": top["config"], "mfava": mfava_cell,
            "nodes": sum(len(tree) for tree in model["trees"]),
        }
        print(name, {key: round(top[key], 4) for key in ("span", "token", "pair", "fpr")},
              "mfava", round(mfava_cell["span"], 4), flush=True)
        if best is None or top["span"] > best[0]:
            best = (top["span"], name, top["config"])

    if best is None:
        raise SystemExit("ни один вариант не уложился в бюджет FPR")

    # 3-fold OOF по документам на реальных русских парах
    docs = sorted({L.doc_key(pair) for pair in train})
    np.random.RandomState(16).shuffle(docs)
    fold_of = {doc: index % 3 for index, doc in enumerate(docs)}
    folds = []
    parts = [part for part in report["variants"][best[1]]["parts"] if part != "real"]
    for fold in range(3):
        held = [pair for pair in train if fold_of[L.doc_key(pair)] == fold]
        held_ids = {pair["pair_id"] for pair in held}
        X, y, rel = matrices(parts)
        Xr, yr, relr = groups["real"][0], groups["real"][1], relative_features(
            groups["real"][0], groups["real"][2])
        block = [X, Xr]
        block_rel = [rel, relr]
        y_all = [y, yr]
        X_all = np.vstack(block)
        rel_all = np.vstack(block_rel)
        y_all = np.concatenate(y_all)
        keep = np.ones(X_all.shape[0], dtype=bool)
        offset = X.shape[0]
        for entry in groups["real"][2]:
            if entry["pair_id"] in held_ids:
                keep[offset:offset + entry["n"]] = False
            offset += entry["n"]
        model = fit(np.hstack([X_all[keep], rel_all[keep]]), y_all[keep], args.trees,
                    args.leaves, args.lr)
        held_tables = [table for table in tables["real"] if table["pair_id"] in held_ids]
        attach_risk(held_tables, model)
        cell = L.cell(S.evaluate(held, held_tables, best[2], "risk_new"))
        cell["pairs"] = len(held)
        folds.append(cell)
        print("oof fold", fold, {key: round(cell[key], 4) for key in ("span", "token", "pair", "fpr")},
              flush=True)
    oof = {key: round(float(np.mean([row[key] for row in folds])), 4)
           for key in ("span", "token", "pair", "fpr")}
    report["oof"] = {"rule": "3 фолда по документам, реальные русские пары",
                     "folds": folds, "mean": oof, "baseline_oof_span_v14": 0.2437,
                     "pass": bool(oof["span"] >= 0.2437 + 0.02)}
    entry = report["variants"][best[1]]
    guard = (entry["val"]["span"] >= 0.50 and entry["val"]["token"] >= 0.50
             and entry["val"]["pair"] >= 0.50 and entry["val"]["fpr"] <= 0.40)
    report["best_val"] = {"variant": best[1], "config": best[2], "val": entry["val"]}
    report["guard"] = {"val_span": 0.50, "val_token": 0.50, "val_pair": 0.50,
                       "val_fpr_max": 0.40, "pass": bool(guard),
                       "oof_pass": report["oof"]["pass"]}
    report["verdict"] = "success" if guard else "reject"

    model = models[best[1]]
    (ROOT / args.model).write_text(json.dumps(model, ensure_ascii=False) + "\n", encoding="utf-8")
    report["model_bytes"] = (ROOT / args.model).stat().st_size
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best", best[1], entry["val"], "verdict", report["verdict"], flush=True)


if __name__ == "__main__":
    main()
