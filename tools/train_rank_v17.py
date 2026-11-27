"""V17 шаги 2–4: ранжировщик кандидатов внутри пары на новых русских данных.

Признаки кандидата (склейка по порядку `spanver.candidate_spans`):
  * 37 штатных лексических v11 (`spanver.ranker.features`);
  * 14 span-level кросс-внимание LM (`tools/span_features_v17.py`);
  * 10 per-token LM v15 (только для наших групп; у новых пар — NaN, HGB их сам обрабатывает).

Модели:
  * `control_37` — HGB на 37 (контроль, как в v16);
  * `hgb_51` — HGB на 37+14;
  * `hgb_61` — HGB на 37+14+10;
  * `pairwise_logit` — RankNet-подобная логистическая регрессия на разностях (pos−neg) внутри
    пары, hard negatives (IoU 0.3–0.7) с повышенным весом;
  * `pairwise_hgb` — HGB на тех же разностях.

Проверка: val (AUC, within-pair accuracy, метрики договора после калибровки порогов в бюджете
FPR ≤ 0.40) + 3-fold Group-CV по документам на новых парах и OOF по документам на реальных
русских парах. Held-out не читается.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sweep_v16 as S
import v11_lab as L
from spanver.evaluate import iou
from spanver.module import candidate_spans, document_context
from spanver import ranker

CACHE14 = ROOT / "artifacts/cache_v14"
CACHE15 = ROOT / "artifacts/cache_v15"
CACHE16 = ROOT / "artifacts/cache_v16"
CACHE17 = ROOT / "artifacts/cache_v17"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def lexical_matrix(pairs, cache_path):
    """37 лексических признаков в порядке candidate_spans (кэш на диск)."""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], json.loads(str(data["meta"]))
    X, y, meta = [], [], []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        carries = {}
        if answer:
            context = document_context(document)
            doc_tokens, doc_grams = context[0], context[2]
            gold = pair.get("spans") or []
            spans = []
            for start, end, fragment in candidate_spans(answer):
                X.append(ranker.features(fragment, start, end, answer, document, doc_tokens,
                                         doc_grams))
                y.append(1 if any(iou(start, end, span["start"], span["end"]) >= 0.5
                                  for span in gold) else 0)
                spans.append([start, end])
            carries["spans"] = spans
        meta.append({"pair_id": pair["pair_id"], "n": len(carries.get("spans", [])),
                     "spans": carries.get("spans", [])})
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, meta=json.dumps(meta))
    return X, y, meta


def v15_features(name):
    path = CACHE15 / f"{name}_lm.npz"
    if not path.exists():
        return None
    return np.load(path)["X"].astype(np.float32)


def span_features(name):
    path = CACHE17 / f"{name}_span.npz"
    if not path.exists():
        return None
    return np.load(path)["X"].astype(np.float32)


def build_group(name, pairs, lexical_cache, with_v15=True):
    X37, y, meta = lexical_matrix(pairs, lexical_cache)
    X17 = span_features(name)
    if X17 is None or X17.shape[0] != X37.shape[0]:
        raise SystemExit(f"{name}: span-признаки не совпали по строкам "
                         f"({None if X17 is None else X17.shape} vs {X37.shape})")
    if with_v15:
        X15 = v15_features(name)
        if X15 is None or X15.shape[0] != X37.shape[0]:
            X15 = np.full((X37.shape[0], 10), np.nan, dtype=np.float32)
    else:
        X15 = np.full((X37.shape[0], 10), np.nan, dtype=np.float32)
    return {"X37": X37, "X17": X17, "X15": X15, "y": y, "meta": meta, "pairs": pairs}


def stack(groups, parts, mode):
    X, y = [], []
    for part in parts:
        data = groups[part]
        if mode == "37":
            X.append(data["X37"])
        elif mode == "51":
            X.append(np.hstack([data["X37"], data["X17"]]))
        else:
            X.append(np.hstack([data["X37"], data["X17"], data["X15"]]))
        y.append(data["y"])
    return np.vstack(X), np.concatenate(y)


def fit_hgb(X, y, trees=250, leaves=31, lr=0.08):
    model = HistGradientBoostingClassifier(max_iter=trees, learning_rate=lr,
                                           max_leaf_nodes=leaves, min_samples_leaf=10,
                                           l2_regularization=1.0, random_state=17)
    return model.fit(X, y)


def pairwise_dataset(groups, parts, per_pair=6, hard_weight=2.0):
    """Разности (pos − neg) внутри пары; hard negatives (IoU 0.3–0.7) с большим весом."""
    diffs, labels, weights = [], [], []
    for part in parts:
        data = groups[part]
        offset = 0
        by_id = {pair["pair_id"]: pair for pair in data["pairs"]}
        for entry in data["meta"]:
            n = entry["n"]
            if n:
                X = np.hstack([data["X37"][offset:offset + n], data["X17"][offset:offset + n],
                               data["X15"][offset:offset + n]])
                y = data["y"][offset:offset + n]
                spans = entry["spans"]
                gold = (by_id.get(entry["pair_id"]) or {}).get("spans") or []
                ious = np.asarray([max([iou(spans[index][0], spans[index][1], span["start"],
                                            span["end"]) for span in gold] or [0.0])
                                   for index in range(n)])
                pos = np.flatnonzero(y == 1)
                neg = np.flatnonzero(y == 0)
                if len(pos) and len(neg):
                    hard = neg[(ious[neg] >= 0.3) & (ious[neg] < 0.7)]
                    easy = neg[~((ious[neg] >= 0.3) & (ious[neg] < 0.7))]
                    for p in pos:
                        chosen = list(hard[:per_pair]) + list(easy[:per_pair])
                        for q in chosen:
                            weight = hard_weight if q in hard else 1.0
                            # зеркальная пара: (pos − neg) → 1, (neg − pos) → 0
                            diffs.append(X[p] - X[q])
                            labels.append(1)
                            weights.append(weight)
                            diffs.append(X[q] - X[p])
                            labels.append(0)
                            weights.append(weight)
            offset += n
    X = np.nan_to_num(np.asarray(diffs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.asarray(labels), np.asarray(weights, dtype=np.float32)


def val_report(model, kind, groups, val, tables, risk_key="risk_new"):
    data = groups["val"]
    if kind == "37":
        X = data["X37"]
    elif kind in ("51", "pairwise_hgb"):
        X = np.hstack([data["X37"], data["X17"]])
    else:
        X = np.hstack([data["X37"], data["X17"], data["X15"]])
    if kind == "pairwise":
        scores = model.decision_function(X) if hasattr(model, "decision_function") else model.predict(X)
    else:
        scores = model.predict_proba(X)[:, 1]
    y = data["y"].astype(int)
    auc = roc_auc_score(y, scores)
    # within-pair pairwise accuracy
    correct = total = 0
    offset = 0
    for entry in data["meta"]:
        n = entry["n"]
        block = scores[offset:offset + n]
        labels = y[offset:offset + n]
        pos = np.flatnonzero(labels == 1)
        neg = np.flatnonzero(labels == 0)
        for p in pos:
            for q in neg:
                total += 1
                correct += 1 if block[p] > block[q] else 0
        offset += n
    # раскладываем скоры по таблицам (используем порядок meta)
    index = 0
    by_id = {table["pair_id"]: table for table in tables["val"]}
    for entry in data["meta"]:
        table = by_id.get(entry["pair_id"])
        if table is None:
            index += entry["n"]
            continue
        for row in table["rows"]:
            row[risk_key] = float(scores[index])
            index += 1
    rows = []
    for config in S.grid():
        rows.append({**L.cell(S.evaluate(val, tables["val"], config, risk_key)),
                     "config": config})
    feasible = [row for row in rows if row["fpr"] <= 0.40]
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    best = feasible[0] if feasible else None
    return {"auc": round(float(auc), 4),
            "within_pair_acc": round(correct / max(total, 1), 4),
            "val": {key: best[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")}
            if best else None,
            "config": best["config"] if best else None}


def group_cv(groups, parts, mode="61", folds=3, seed=17):
    """Group-CV по документам на новых парах: AUC на отложенных документах."""
    data = groups["sber"]
    docs = sorted({L.doc_key(pair) for pair in data["pairs"]})
    np.random.RandomState(seed).shuffle(docs)
    fold_of = {doc: index % folds for index, doc in enumerate(docs)}
    row_fold = []
    by_id = {pair["pair_id"]: pair for pair in data["pairs"]}
    for entry in data["meta"]:
        pair = by_id[entry["pair_id"]]
        row_fold.extend([fold_of[L.doc_key(pair)]] * entry["n"])
    row_fold = np.asarray(row_fold)
    X, y = stack(groups, ["sber"], mode)
    results = []
    for fold in range(folds):
        train_mask = row_fold != fold
        held_mask = ~train_mask
        if train_mask.sum() == 0 or held_mask.sum() == 0:
            continue
        model = fit_hgb(X[train_mask], y[train_mask])
        scores = model.predict_proba(X[held_mask])[:, 1]
        results.append({"fold": fold, "rows": int(held_mask.sum()),
                        "positives": int(y[held_mask].sum()),
                        "auc": round(float(roc_auc_score(y[held_mask], scores)), 4)})
    mean = round(float(np.mean([row["auc"] for row in results])), 4) if results else None
    return {"folds": results, "mean_auc": mean}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ranker_train_v17.json")
    parser.add_argument("--model", default="artifacts/ranker_model_v17.json")
    parser.add_argument("--sber-limit", type=int, default=6000)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--lr", type=float, default=0.08)
    args = parser.parse_args(argv)

    L.set_weights(L.weights())
    train, val, mfava = L.load_all()
    sber = read_jsonl(ROOT / "artifacts/sber_flip_v16.jsonl")[:args.sber_limit]
    groups = {}
    for name, pairs, cache in (
            ("sber", sber, CACHE17 / "sber37_aligned.npz"),
            ("real", train, CACHE14 / "real.npz"),
            ("synth", read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl"), CACHE14 / "synth.npz"),
            ("pseudo", read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl"), CACHE14 / "pseudo.npz"),
            ("val", val, CACHE14 / "val.npz"),
            ("mfava", mfava, CACHE14 / "mfava.npz")):
        groups[name] = build_group(name, pairs, cache, with_v15=(name != "sber"))
        print(name, groups[name]["X37"].shape, "pos", int(groups[name]["y"].sum()), flush=True)

    tables = {"val": S.build_tables(val, L.weights(), CACHE16 / "val_tables.json")}

    report = {
        "rule": "ранжировщик кандидатов внутри пары: 37 лексических + 14 span-level кросс-внимание "
                "+ 10 per-token LM; декодер v11 (mode-гейт, pair-гейт, local_hgb) сохранён; "
                "пороги калибруются на val в бюджете FPR ≤ 0.40; held-out не читается",
        "features": {"lexical": 37, "span_cross_attention": 14, "per_token_lm_v15": 10},
        "data": {name: {"pairs": len(data["pairs"]), "candidates": int(data["X37"].shape[0]),
                        "positives": int(data["y"].sum())}
                 for name, data in groups.items()},
        "hard_negatives": {"definition": "кандидат с IoU 0.3–0.7 к gold внутри пары",
                           "weight": 2.0},
        "variants": {},
    }

    parts_main = ["sber", "real", "synth", "pseudo"]
    results = {}
    for kind, mode, label in (("37", "37", "control_37"), ("51", "51", "hgb_51"),
                              ("61", "61", "hgb_61")):
        X, y = stack(groups, parts_main, mode)
        started = time.time()
        model = fit_hgb(X, y, args.trees, args.leaves, args.lr)
        fit_seconds = time.time() - started
        entry = val_report(model, mode if kind != "61" else "61", groups, val, tables)
        entry.update({"train_rows": int(X.shape[0]), "train_positives": int(y.sum()),
                      "fit_seconds": round(fit_seconds, 1)})
        report["variants"][label] = entry
        results[label] = (model, mode)
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        print(label, "auc", entry["auc"], "pairwise", entry["within_pair_acc"],
              "val", entry["val"], flush=True)

    Xp, yp, wp = pairwise_dataset(groups, parts_main)
    started = time.time()
    logit = LogisticRegression(max_iter=2000, C=0.5).fit(Xp, yp, sample_weight=wp)
    pairwise_logit_seconds = time.time() - started
    entry = val_report(logit, "pairwise", groups, val, tables)
    entry.update({"train_rows": int(Xp.shape[0]), "fit_seconds": round(pairwise_logit_seconds, 1)})
    report["variants"]["pairwise_logit"] = entry
    results["pairwise_logit"] = (logit, "pairwise")
    print("pairwise_logit", "auc", entry["auc"], "pairwise", entry["within_pair_acc"],
          "val", entry["val"], flush=True)

    Xp51 = Xp[:, :51]
    started = time.time()
    pairwise_hgb = fit_hgb(Xp51, yp, trees=min(args.trees, 150), leaves=15, lr=0.08)
    entry = val_report(pairwise_hgb, "pairwise_hgb", groups, val, tables)
    entry.update({"train_rows": int(Xp51.shape[0]),
                  "fit_seconds": round(time.time() - started, 1)})
    report["variants"]["pairwise_hgb_51"] = entry
    results["pairwise_hgb_51"] = (pairwise_hgb, "pairwise_hgb")
    print("pairwise_hgb", "auc", entry["auc"], "pairwise", entry["within_pair_acc"],
          "val", entry["val"], flush=True)

    try:
        report["group_cv_sber"] = group_cv(groups, ["sber"], mode="51")
    except Exception as error:  # noqa: BLE001 - диагностика не должна ронять отчёт
        report["group_cv_sber"] = {"error": f"{type(error).__name__}: {str(error)[:200]}"}
    print("group_cv_sber", report["group_cv_sber"], flush=True)

    best_label = max(report["variants"].items(),
                     key=lambda item: (item[1]["val"] or {"span": -1})["span"])[0]
    report["best"] = {"variant": best_label, **{key: report["variants"][best_label][key]
                                                for key in ("auc", "within_pair_acc", "val",
                                                            "config")}}
    model, mode = results[best_label]
    if mode in ("37", "51", "61"):
        model_json = None
        try:
            from train_pair_classifier_v12 import export_trees
            model_json = export_trees(model)
        except Exception as error:  # noqa: BLE001 - экспорт не критичен для отчёта
            report["export_error"] = str(error)[:200]
        if model_json is not None:
            (ROOT / args.model).write_text(json.dumps(model_json, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
            report["model_bytes"] = (ROOT / args.model).stat().st_size
    report["verdict"] = "candidate" if (report["best"]["val"] or {"span": 0})["span"] >= 0.50 \
        else "reject_by_span"
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print("best", best_label, report["best"]["val"], report["verdict"], flush=True)


if __name__ == "__main__":
    main()
