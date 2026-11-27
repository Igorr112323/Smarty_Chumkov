"""V16 шаги 3–5: GBDT-риск на новых русских данных + калибровка в бюджете FPR ≤ 0.40.

Что делаем:
  * кандидаты и 37 штатных признаков v11 считаются для нового датасета
    (`artifacts/sber_flip_v16.jsonl`), наших real/synth/pseudo пар — один раз, с кэшем;
  * HGB (HistGradientBoosting) учится на смеси, деревья экспортируются в чистый JSON
    (`toolless` рантайм, как v8/v12/v14);
  * декодер остаётся v11: mode-гейт, pair-гейт, local_hgb и boundary-логика не сбрасываются,
    меняется только скор кандидата и калибровочные пороги (шаг 5 договора — бюджет FPR до 0.40);
  * проверка: val (calibration) + 3-fold OOF по документам на реальных русских парах.

Held-out не читается.
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
from spanver import ranker
from spanver.evaluate import iou, metrics
from spanver.module import candidate_spans, document_context
from train_pair_classifier_v12 import export_trees, tree_probability

CACHE = ROOT / "artifacts/cache_v16"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def pair_rows(pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    if not answer:
        return []
    context = document_context(document)
    doc_tokens, doc_grams = context[0], context[2]
    result = []
    for start, end, fragment in candidate_spans(answer):
        result.append((start, end, ranker.features(fragment, start, end, answer, document,
                                                   doc_tokens, doc_grams)))
    return result


def build_matrix(pairs, cache_path, log_every=5000):
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], json.loads(str(data["meta"]))
    X, y, meta = [], [], []
    started = time.time()
    for index, pair in enumerate(pairs):
        rows = pair_rows(pair)
        gold = pair.get("spans") or []
        for start, end, vector in rows:
            hit = any(iou(start, end, span["start"], span["end"]) >= 0.5 for span in gold)
            X.append(vector)
            y.append(1 if hit else 0)
        meta.append({"pair_id": pair["pair_id"], "n": len(rows),
                     "spans": [[row[0], row[1]] for row in rows]})
        if log_every and (index + 1) % log_every == 0:
            print(f"  {index + 1}/{len(pairs)} пар, {time.time() - started:.1f}s", flush=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, meta=json.dumps(meta))
    return X, y, meta


def fit(X, y, trees, leaves, lr, pos_weight):
    weights = np.where(y == 1, pos_weight, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=trees, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=10,
        l2_regularization=1.0, random_state=16,
    ).fit(X, y, sample_weight=weights)
    return export_trees(model)


def tree_scores(model, X):
    return np.asarray([tree_probability(model, list(row)) for row in X])


def attach_risk(tables, model):
    for table in tables:
        for row in table["rows"]:
            row["risk_new"] = tree_probability(model, list(row["features"]))
    return tables


def calibrate(pairs, tables, fpr_max=0.40):
    rows = []
    for config in S.grid():
        cell = L.cell(S.evaluate(pairs, tables, config, "risk_new"))
        rows.append({**cell, "config": config})
    feasible = [row for row in rows if row["fpr"] <= fpr_max]
    if not feasible:
        return None, rows
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    return feasible[0], rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/span_oof_v16.json")
    parser.add_argument("--model", default="artifacts/span_model_v16.json")
    parser.add_argument("--sber-limit", type=int, default=25000)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    train, val, mfava = L.load_all()
    sber = read_jsonl(ROOT / "artifacts/sber_flip_v16.jsonl")
    synth = read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl")
    pseudo = read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl")

    if args.sber_limit:
        rng = np.random.RandomState(16)
        clean = [row for row in sber if not row["spans"]]
        flips = [row for row in sber if row["spans"]]
        keep_clean = clean
        take = max(args.sber_limit - len(keep_clean), 0)
        picked = rng.choice(len(flips), size=min(take, len(flips)), replace=False)
        sber = keep_clean + [flips[index] for index in sorted(picked)]
    print(f"данные: sber {len(sber)}, synth {len(synth)}, pseudo {len(pseudo)}, real {len(train)}",
          flush=True)

    groups = {}
    for name, pairs in (("sber", sber), ("synth", synth), ("pseudo", pseudo), ("real", train),
                        ("val", val), ("mfava", mfava)):
        groups[name] = build_matrix(pairs, CACHE / f"{name}37.npz")
        print(name, groups[name][0].shape, "positives", int(groups[name][1].sum()), flush=True)

    # таблицы со скорами v11 для val/mfava/real (нужны для декодера и OOF)
    tables = {}
    for name, pairs in (("val", val), ("mfava", mfava), ("real", train)):
        tables[name] = S.build_tables(pairs, params, CACHE / f"{name}_tables.json")

    variants = {
        "sber_only": ["sber"],
        "sber+synth+pseudo": ["sber", "synth", "pseudo"],
        "sber+synth+pseudo+real": ["sber", "synth", "pseudo", "real"],
        "sber+real": ["sber", "real"],
        "baseline_synth+pseudo+real": ["synth", "pseudo", "real"],
    }
    report = {
        "rule": "декодер v11 (mode-гейт, pair-гейт, local_hgb) сохранён; переобучается только "
                "скор кандидата (HGB на 37 признаках) и калибруются пороги; бюджет FPR ≤ 0.40; "
                "held-out не читается",
        "baseline_v11": {"span": 0.3363, "token": 0.6191, "pair": 0.4872, "fpr": 0.125},
        "data": {name: {"pairs": len(pairs) if isinstance(pairs, list) else None,
                        "candidates": int(groups[name][0].shape[0]),
                        "positives": int(groups[name][1].sum())}
                 for name, pairs in (("sber", sber), ("synth", synth), ("pseudo", pseudo),
                                     ("real", train), ("val", val), ("mfava", mfava))},
        "model": {"trees": args.trees, "leaves": args.leaves, "lr": args.lr,
                  "pos_weight": args.pos_weight, "features": 37, "sber_limit": args.sber_limit},
        "variants": {},
    }

    best = None
    models = {}
    for name, parts in variants.items():
        X = np.vstack([groups[part][0] for part in parts])
        y = np.concatenate([groups[part][1] for part in parts])
        started = time.time()
        model = fit(X, y, args.trees, args.leaves, args.lr, args.pos_weight)
        fit_seconds = time.time() - started
        models[name] = model
        attach_risk(tables["val"], model)
        attach_risk(tables["mfava"], model)
        best_cell, all_rows = calibrate(val, tables["val"], fpr_max=0.40)
        if best_cell is None:
            print(name, "нет конфигураций в бюджете FPR", flush=True)
            continue
        mfava_cell = L.cell(S.evaluate(mfava, tables["mfava"], best_cell["config"], "risk_new"))
        entry = {
            "parts": parts, "train_rows": int(X.shape[0]), "train_positives": int(y.sum()),
            "fit_seconds": round(fit_seconds, 1),
            "val": {key: best_cell[key] for key in ("span", "token", "pair", "fpr", "span_p",
                                                    "span_r")},
            "config": best_cell["config"], "mfava": mfava_cell,
            "nodes": sum(len(tree) for tree in model["trees"]),
        }
        report["variants"][name] = entry
        print(name, {key: round(entry["val"][key], 4) for key in ("span", "token", "pair", "fpr")},
              "mfava", round(mfava_cell["span"], 4), flush=True)
        if best is None or entry["val"]["span"] > best[0]:
            best = (entry["val"]["span"], name, best_cell["config"])

    report["best_val"] = {"variant": best[1], "config": best[2],
                          "val": report["variants"][best[1]]["val"]}

    # 3-fold OOF по документам на реальных русских парах
    docs = sorted({L.doc_key(pair) for pair in train})
    rng = np.random.RandomState(16)
    rng.shuffle(docs)
    fold_of = {doc: index % args.folds for index, doc in enumerate(docs)}
    folds = []
    for fold in range(args.folds):
        held = [pair for pair in train if fold_of[L.doc_key(pair)] == fold]
        rest = [pair for pair in train if fold_of[L.doc_key(pair)] != fold]
        held_ids = {pair["pair_id"] for pair in held}
        parts = [part for part in best_parts(report, best[1]) if part != "real"]
        X_parts = [groups[part][0] for part in parts] + [groups["real"][0]]
        y_parts = [groups[part][1] for part in parts] + [groups["real"][1]]
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        # исключаем строки held-пар из real-части
        mask = np.ones(X.shape[0], dtype=bool)
        offset = sum(groups[part][0].shape[0] for part in parts)
        for entry in groups["real"][2]:
            if entry["pair_id"] in held_ids:
                mask[offset:offset + entry["n"]] = False
            offset += entry["n"]
        model = fit(X[mask], y[mask], args.trees, args.leaves, args.lr, args.pos_weight)
        held_tables = [table for table in tables["real"] if table["pair_id"] in held_ids]
        attach_risk(held_tables, model)
        cell = L.cell(S.evaluate(held, held_tables, best[2], "risk_new"))
        cell["pairs"] = len(held)
        folds.append(cell)
        print("oof fold", fold, {key: round(cell[key], 4) for key in ("span", "token", "pair", "fpr")},
              flush=True)
    oof_mean = {key: round(float(np.mean([row[key] for row in folds])), 4)
                for key in ("span", "token", "pair", "fpr")} if folds else {}
    report["oof"] = {"rule": "3 фолда по документам, реальные русские пары train; модель "
                             "переобучается на каждом фолде без held-документов",
                     "folds": folds, "mean": oof_mean,
                     "baseline_oof_span_v14": 0.2437,
                     "pass": bool(oof_mean and oof_mean["span"] >= 0.2437 + 0.02)}

    entry = report["variants"][best[1]]
    guard = (entry["val"]["span"] >= 0.50 and entry["val"]["token"] >= 0.50
             and entry["val"]["pair"] >= 0.50 and entry["val"]["fpr"] <= 0.40)
    report["guard"] = {"val_span": 0.50, "val_token": 0.50, "val_pair": 0.50, "val_fpr_max": 0.40,
                       "pass": bool(guard), "oof_pass": report["oof"]["pass"]}
    report["verdict"] = "success" if guard else "reject"

    model = models[best[1]]
    (ROOT / args.model).write_text(json.dumps(model, ensure_ascii=False) + "\n", encoding="utf-8")
    report["model_bytes"] = (ROOT / args.model).stat().st_size
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best", best[1], report["best_val"], "verdict", report["verdict"], flush=True)


def best_parts(report, name):
    return report["variants"][name]["parts"]


if __name__ == "__main__":
    main()
