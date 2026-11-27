"""V15 шаг 3: дообучение только GBDT-части на 37 штатных + 10 LM-признаках.

Сравнение честное, без held-out:
  * `control_37` — тот же пайплайн v14 на 37 признаках (контроль);
  * `v15_47`     — 37 + 10 LM-признаков (`artifacts/cache_v15/*_lm.npz`);
  * `v15_47_real+synth` — без pseudo/ragtruth, чтобы видеть вклад LM отдельно.

Порог и max_spans калибруются на val (как в v14), mFAVA — отдельный набор, held-out не
читается. Дополнительно 3-fold OOF по парам обучающей части: сравниваем со средним OOF
v14 (0.2437). Guard v15: Span ≥ 0.40, Token ≥ 0.50, Pair ≥ 0.45, FPR ≤ 0.40,
mFAVA ≥ 0.65, OOF ≥ baseline-OOF.
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
from spanver.evaluate import metrics
from train_span_v14 import decode_pair, evaluate_split, fit, scores_for, sweep

GUARD = {"span": 0.40, "token": 0.50, "pair": 0.45, "fpr_max": 0.40, "mfava_span": 0.65,
         "oof_span": 0.2437}
BASELINE = {"val": {"span": 0.3363, "token": 0.6191, "pair": 0.4872, "fpr": 0.125},
            "mfava_span": 0.6895, "oof_span": 0.2437}
GRIDS = [{"threshold": threshold, "max_spans": max_spans}
         for threshold in (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
         for max_spans in (1, 2, 3)]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_group(name, train, val, mfava, extra):
    pairs = {"real": train, "synth": extra["synth"], "pseudo": extra["pseudo"],
             "ragtruth": extra["ragtruth"], "val": val, "mfava": mfava}[name]
    base = np.load(ROOT / "artifacts/cache_v14" / f"{name}.npz", allow_pickle=True)
    lm = np.load(ROOT / "artifacts/cache_v15" / f"{name}_lm.npz")
    meta = json.loads(str(base["meta"]))
    X37 = base["X"]
    X47 = np.hstack([X37, lm["X"].astype(np.float32)])
    total = sum(entry["n"] for entry in meta)
    if X47.shape[0] != total:
        raise SystemExit(f"{name}: строк {X47.shape[0]} != кандидатов в meta {total}")
    return {"X37": X37, "X47": X47, "y": base["y"], "meta": meta, "pairs": pairs}


def pair_normalize(X47, meta, start=37):
    """Z-нормировка LM-признаков внутри пары: абсолютные значения внимания зависят от
    длины и домена пары, а сравнение кандидатов — внутри одной пары."""
    out = X47.copy()
    offset = 0
    for entry in meta:
        n = entry["n"]
        if n > 1:
            block = X47[offset:offset + n, start:]
            mean = block.mean(0)
            std = block.std(0)
            out[offset:offset + n, start:] = (block - mean) / (std + 1e-6)
        offset += n
    return out


def fold_matrix(groups, parts, folds, fold, train_mask=True):
    X37, X47, y, pairs, meta = [], [], [], [], []
    for name in parts:
        data = groups[name]
        offset = 0
        for index, entry in enumerate(data["meta"]):
            tag = f"{name}:{entry['pair_id']}"
            hit = folds[tag] != fold
            if hit != train_mask:
                offset += entry["n"]
                continue
            slice_ = slice(offset, offset + entry["n"])
            X37.append(data["X37"][slice_])
            X47.append(data["X47"][slice_])
            y.append(data["y"][slice_])
            pairs.append(data["pairs"][index])
            meta.append(entry)
            offset += entry["n"]
    stack = lambda chunks, dim: (np.vstack(chunks) if chunks else np.zeros((0, dim), np.float32))
    return (stack(X37, 37), stack(X47, 47), (np.concatenate(y) if y else np.zeros(0, np.int8)),
            pairs, meta)


def normalize_rows_by_pair(X47, groups, parts, folds, fold, train_mask):
    offset = 0
    out = X47.copy()
    for name in parts:
        data = groups[name]
        for entry in data["meta"]:
            hit = folds[f"{name}:{entry['pair_id']}"] != fold
            if hit != train_mask:
                continue
            n = entry["n"]
            if n > 1:
                block = X47[offset:offset + n, 37:]
                out[offset:offset + n, 37:] = (block - block.mean(0)) / (block.std(0) + 1e-6)
            offset += n
    return out


def make_folds(pairs_by_group, seed=15):
    rng = np.random.RandomState(seed)
    tags = []
    for name, pairs in pairs_by_group.items():
        for pair in pairs:
            tags.append(f"{name}:{pair['pair_id']}")
    tags = sorted(set(tags))
    rng.shuffle(tags)
    return {tag: index % 3 for index, tag in enumerate(tags)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/span_oof_v15.json")
    parser.add_argument("--model", default="artifacts/span_model_v15.json")
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--leaves", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args(argv)

    L.set_weights(L.weights())
    train, val, mfava = L.load_all()
    extra = {
        "synth": read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl"),
        "pseudo": read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl"),
        "ragtruth": read_jsonl(ROOT / "artifacts/ragtruth_proj_v14.jsonl"),
    }
    groups = {name: load_group(name, train, val, mfava, extra)
              for name in ("real", "synth", "pseudo", "ragtruth", "val", "mfava")}
    print("данные:", {name: (int(data["X47"].shape[0]), int(data["y"].sum()))
                      for name, data in groups.items()}, flush=True)

    report = {
        "rule": "GBDT (HistGradientBoosting, экспорт деревьев в JSON) на 37 признаках v11 + "
                "10 LM-признаках rubert-tiny2; LM не дообучается; порог калибруется на val; "
                "held-out не читается",
        "baseline": BASELINE,
        "guard": GUARD,
        "model": {"trees": args.trees, "leaves": args.leaves, "lr": args.lr,
                  "pos_weight": args.pos_weight},
        "data": {name: {"candidates": int(data["X47"].shape[0]),
                        "positives": int(data["y"].sum())}
                 for name, data in groups.items()},
        "ablations": {},
    }

    for name, data in groups.items():
        data["X47n"] = pair_normalize(data["X47"], data["meta"])

    variants = {
        "control_37": ("37", ["real", "synth", "pseudo", "ragtruth"]),
        "v15_47": ("47", ["real", "synth", "pseudo", "ragtruth"]),
        "v15_47_pairnorm": ("47n", ["real", "synth", "pseudo", "ragtruth"]),
        "v15_47_pairnorm_real": ("47n", ["real"]),
        "v15_lm_only": ("lm", ["real", "synth", "pseudo", "ragtruth"]),
    }
    best = None
    models = {}
    for name, (kind, parts) in variants.items():
        if kind == "37":
            X = np.vstack([groups[part]["X37"] for part in parts])
        elif kind == "47":
            X = np.vstack([groups[part]["X47"] for part in parts])
        elif kind == "47n":
            X = np.vstack([groups[part]["X47n"] for part in parts])
        else:
            X = np.vstack([groups[part]["X47n"][:, 37:] for part in parts])
        y = np.concatenate([groups[part]["y"] for part in parts])
        started = time.time()
        model = fit(X, y, args.trees, args.leaves, args.lr, args.pos_weight)
        fit_seconds = time.time() - started
        val_X = groups["val"][{"37": "X37", "47": "X47", "47n": "X47n"}.get(kind, "X47n")]
        if kind == "lm":
            val_X = groups["val"]["X47n"][:, 37:]
        val_scores = scores_for(model, val_X)
        val_rows, val_best = sweep(groups["val"]["pairs"], groups["val"]["meta"],
                                   val_scores, GRIDS)
        mfava_X_pre = groups["mfava"][{"37": "X37", "47": "X47", "47n": "X47n"}.get(kind, "X47n")]
        if kind == "lm":
            mfava_X_pre = groups["mfava"]["X47n"][:, 37:]
        mfava_scores = scores_for(model, mfava_X_pre)
        combined = sorted(
            ({"config": row["config"],
              "span": round((row["span"] + L.cell(evaluate_split(
                  groups["mfava"]["pairs"], groups["mfava"]["meta"], mfava_scores,
                  row["config"]))["span"]) / 2.0, 4)}
             for row in val_rows), key=lambda item: -item["span"])
        config = val_best["config"]
        mfava_metrics = L.cell(evaluate_split(groups["mfava"]["pairs"], groups["mfava"]["meta"],
                                              mfava_scores, config))
        val_at_config = L.cell(evaluate_split(groups["val"]["pairs"], groups["val"]["meta"],
                                              val_scores, config))
        models[name] = (model, config, kind)
        guard_pass = (val_at_config["span"] >= GUARD["span"]
                      and val_at_config["token"] >= GUARD["token"]
                      and val_at_config["pair"] >= GUARD["pair"]
                      and val_at_config["fpr"] <= GUARD["fpr_max"]
                      and mfava_metrics["span"] >= GUARD["mfava_span"])
        report["ablations"][name] = {
            "features": kind, "parts": parts,
            "train_rows": int(X.shape[0]), "train_positives": int(y.sum()),
            "fit_seconds": round(fit_seconds, 1),
            "val_best_single_split": val_best, "val_at_combined_config": val_at_config,
            "combined_config": combined[0]["config"],
            "val_span_at_combined_config": combined[0]["span"], "all_configs": val_rows, "mfava": mfava_metrics,
            "nodes": sum(len(tree) for tree in model["trees"]),
            "guard_pass_without_oof": guard_pass,
        }
        print(name, "val", {k: val_at_config[k] for k in ("span", "token", "pair", "fpr")},
              "mfava", round(mfava_metrics["span"], 4), "config", config, flush=True)
        if best is None or val_at_config["span"] > best[0]:
            best = (val_at_config["span"], name)

    # 3-fold OOF по парам обучающей части для лучшего варианта и контроля
    # OOF честно считается на реальных русских парах: OOF v14 (0.2437) измерен так же.
    # Контроль 37 признаков и вариант с LM — на одних и тех же фолдах.
    report["oof"] = {
        "rule": "3 фолда по парам, конфиг берётся из val; отдельно реальные русские пары "
                "(с ними сравнивается OOF v14 = 0.2437) и полная обучающая смесь",
        "baseline_mean_span": BASELINE["oof_span"], "sets": {}}
    oof_sets = (("real", ["real"]), ("mixture", ["real", "synth", "pseudo", "ragtruth"]))
    for set_name, parts in oof_sets:
        pairs_by_group = {name: groups[name]["pairs"] for name in parts}
        folds = make_folds(pairs_by_group)
        report["oof"]["sets"][set_name] = {"parts": parts, "variants": {}}
        for oof_name in ("control_37", best[1]):
            kind = models[oof_name][2]
            config = models[oof_name][1]
            fold_rows = []
            for fold in range(args.folds):
                X_train, X_hold, y_train, _, _ = fold_matrix(groups, parts, folds, fold,
                                                             train_mask=True)
                _, _, y_hold, pairs_hold, meta_hold = fold_matrix(groups, parts, folds, fold,
                                                                  train_mask=False)
                if kind == "47n":
                    # нормировка внутри пары считается отдельно для train и для холда
                    X_train = normalize_rows_by_pair(X_train, groups, parts, folds, fold, True)
                    X_hold = normalize_rows_by_pair(X_hold, groups, parts, folds, fold, False)
                if X_train.shape[0] == 0 or X_hold.shape[0] == 0:
                    continue
                model = fit(X_train, y_train, args.trees, args.leaves, args.lr, args.pos_weight)
                cell = L.cell(evaluate_split(pairs_hold, meta_hold,
                                             scores_for(model, X_hold), config))
                cell.update({"config": config, "rows": int(X_hold.shape[0]),
                             "positives": int(y_hold.sum())})
                fold_rows.append(cell)
                print("oof", set_name, oof_name, "fold", fold,
                      {k: round(cell[k], 4) for k in ("span", "token", "pair", "fpr")},
                      flush=True)
            oof = ({key: round(float(np.mean([row[key] for row in fold_rows])), 4)
                    for key in ("span", "token", "pair", "fpr")} if fold_rows else {})
            report["oof"]["sets"][set_name]["variants"][oof_name] = {
                "features": kind, "folds": fold_rows, "mean": oof,
                "delta_vs_baseline": (round(oof["span"] - BASELINE["oof_span"], 4)
                                      if oof else None),
                "pass": bool(oof and oof["span"] >= BASELINE["oof_span"])}
    report["oof"]["pass"] = all(
        report["oof"]["sets"][set_name]["variants"][best[1]]["pass"]
        for set_name, _ in oof_sets)

    best_name = best[1]
    entry = report["ablations"][best_name]
    entry["guard_pass"] = bool(entry["guard_pass_without_oof"] and report["oof"].get("pass"))
    report["best"] = {"ablation": best_name, "config": models[best_name][1],
                      "val": {k: entry["val_at_combined_config"][k]
                              for k in ("span", "token", "pair", "fpr")},
                      "mfava_span": entry["mfava"]["span"]}
    report["verdict"] = "success" if entry["guard_pass"] else "reject"
    model, config, kind = models[best_name]
    (ROOT / args.model).write_text(json.dumps(model, ensure_ascii=False) + "\n", encoding="utf-8")
    report["best"]["model_bytes"] = (ROOT / args.model).stat().st_size
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best", best_name, report["best"], "verdict", report["verdict"], flush=True)


if __name__ == "__main__":
    main()
