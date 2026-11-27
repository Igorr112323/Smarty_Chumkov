"""V20: ранжировщик на синтетическом датасете v19 (реальный клаузовый стиль) + честный
перенос на реальные русские пары.

Данные (шаг 1):
  * `real`   — реальный train из gold_v3 (549 пар, ручная разметка);
  * `synth`  — 3563 синтетических флипа v11;
  * `pseudo` — 84 pseudo-clause v11;
  * `v19`    — 1659 пар клаузовых галлюцинаций в реальном стиле (doc_overlap = 0);
  * `val`    — 150 реальных пар (кросс-домен), `mfava` — внутренний набор (не блокер).

Признаки: 37 штатных лексических v11 (`spanver.ranker.features`), декодер v11 сохранён
(mode-гейт, pair-гейт, local_hgb). Пороги калибруются на val в бюджете FPR <= 0.40.

Варианты: `control_37` (без v19) / `v19_only` / `v19_plus_real` / `v19_plus_real_hardneg`
(hard negatives IoU 0.3–0.7, вес 2.0) / `full_pairwise` (RankNet-подобные разности + hard).

Отчёты: `artifacts/ranker_train_v20.json`, `artifacts/span_oof_v20.json`.
Held-out (gold1200_v3) не читается.
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
CACHE16 = ROOT / "artifacts/cache_v16"
CACHE20 = ROOT / "artifacts/cache_v20"
BASELINE_OOF_SPAN = 0.2437


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def lexical_matrix(pairs, cache_path):
    """37 лексических признаков в порядке `candidate_spans` (кэш на диск)."""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], json.loads(str(data["meta"]))
    X, y, meta = [], [], []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        spans = []
        if answer:
            context = document_context(document)
            doc_tokens, doc_grams = context[0], context[2]
            gold = pair.get("spans") or []
            for start, end, fragment in candidate_spans(answer):
                X.append(ranker.features(fragment, start, end, answer, document, doc_tokens,
                                         doc_grams))
                y.append(1 if any(iou(start, end, span["start"], span["end"]) >= 0.5
                                  for span in gold) else 0)
                spans.append([start, end])
        meta.append({"pair_id": pair["pair_id"], "n": len(spans), "spans": spans})
    X = np.asarray(X, dtype=np.float32).reshape(-1, 37)
    y = np.asarray(y, dtype=np.int8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, meta=json.dumps(meta))
    return X, y, meta


def candidate_ious(pair, spans):
    gold = pair.get("spans") or []
    return np.asarray([max([iou(start, end, span["start"], span["end"]) for span in gold] or [0.0])
                       for start, end in spans], dtype=np.float32)


def build_group(name, pairs, cache):
    X, y, meta = lexical_matrix(pairs, cache)
    by_id = {pair["pair_id"]: pair for pair in pairs}
    offset, ious, pair_rows = 0, np.zeros(len(y), dtype=np.float32), []
    for entry in meta:
        pair = by_id[entry["pair_id"]]
        n = entry["n"]
        ious[offset:offset + n] = candidate_ious(pair, entry["spans"])
        pair_rows.append(offset)
        offset += n
    return {"X": X, "y": y, "meta": meta, "pairs": pairs, "ious": ious, "offset": np.asarray(pair_rows)}


def hard_weights(data, weight=2.0):
    """Вес 2.0 кандидатам-негативам с IoU 0.3–0.7 (hard negatives)."""
    hard = (data["y"] == 0) & (data["ious"] >= 0.3) & (data["ious"] < 0.7)
    return np.where(hard, weight, 1.0).astype(np.float32), int(hard.sum())


def fit_hgb(X, y, sample_weight=None, trees=250, leaves=31, lr=0.08, seed=17):
    model = HistGradientBoostingClassifier(max_iter=trees, learning_rate=lr,
                                           max_leaf_nodes=leaves, min_samples_leaf=10,
                                           l2_regularization=1.0, random_state=seed)
    return model.fit(X, y, sample_weight=sample_weight)


def stack(groups, parts):
    X = np.vstack([groups[part]["X"] for part in parts]) if parts else np.zeros((0, 37), np.float32)
    y = np.concatenate([groups[part]["y"] for part in parts]) if parts else np.zeros(0, np.int8)
    return X, y


def pairwise_dataset(groups, parts, per_pair=6, hard_weight=2.0):
    """Разности (pos − neg) внутри пары; hard negatives (IoU 0.3–0.7) с большим весом."""
    diffs, labels, weights = [], [], []
    for part in parts:
        data = groups[part]
        offset = 0
        for entry in data["meta"]:
            n = entry["n"]
            if n:
                X = data["X"][offset:offset + n]
                y = data["y"][offset:offset + n]
                ious = data["ious"][offset:offset + n]
                pos = np.flatnonzero(y == 1)
                neg = np.flatnonzero(y == 0)
                if len(pos) and len(neg):
                    hard = neg[(ious[neg] >= 0.3) & (ious[neg] < 0.7)]
                    easy = neg[~np.isin(neg, hard)]
                    for p in pos:
                        for q in list(hard[:per_pair]) + list(easy[:per_pair]):
                            weight = hard_weight if q in hard else 1.0
                            diffs.append(X[p] - X[q])
                            labels.append(1)
                            weights.append(weight)
                            diffs.append(X[q] - X[p])
                            labels.append(0)
                            weights.append(weight)
            offset += n
    X = np.nan_to_num(np.asarray(diffs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.asarray(labels), np.asarray(weights, dtype=np.float32)


def score_rows(model, X, kind):
    if kind == "pairwise":
        raw = model.decision_function(X) if hasattr(model, "decision_function") else model.predict(X)
        return np.asarray(raw, dtype=np.float64)
    return model.predict_proba(X)[:, 1]


def within_pair_accuracy(scores, data):
    correct = total = 0
    offset = 0
    y = data["y"].astype(int)
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
    return round(correct / max(total, 1), 4)


def val_cell(model, kind, groups, val, tables, risk_key="risk_new"):
    """Метрики договора на нашем val: пороги из сетки в бюджете FPR <= 0.40."""
    data = groups["val"]
    scores = score_rows(model, data["X"], kind)
    index = 0
    by_id = {table["pair_id"]: table for table in tables}
    for entry in data["meta"]:
        table = by_id.get(entry["pair_id"])
        if table is None:
            index += entry["n"]
            continue
        for row in table["rows"]:
            row[risk_key] = float(scores[index])
            index += 1
    rows = [{**L.cell(S.evaluate(val, tables, config, risk_key)), "config": config}
            for config in S.grid()]
    feasible = [row for row in rows if row["fpr"] <= 0.40]
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    best = feasible[0] if feasible else None
    return {
        "auc": round(float(roc_auc_score(data["y"].astype(int), scores)), 4),
        "within_pair_acc": within_pair_accuracy(scores, data),
        "val": {key: best[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")}
        if best else None,
        "config": best["config"] if best else None,
    }


def rank_norm(values):
    """Ранговая нормировка в [0, 1] (устойчиво к масштабу скоров)."""
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks / max(len(order) - 1, 1)


def base_risks(groups, tables, risk_key="risk_cand"):
    """Базовые скоры v11 по кандидатам в порядке строк группы val."""
    data = groups["val"]
    by_id = {table["pair_id"]: table for table in tables}
    values = np.zeros(int(data["X"].shape[0]), dtype=np.float64)
    index = 0
    for entry in data["meta"]:
        table = by_id.get(entry["pair_id"])
        if table is None:
            index += entry["n"]
            continue
        for row in table["rows"]:
            values[index] = row[risk_key]
            index += 1
    return values


def blend_cell(model, kind, groups, val, tables):
    """Смесь штатного риска v11 с моделью: вес подбирается по AUC на val (наш dev-набор)."""
    data = groups["val"]
    model_scores = score_rows(model, data["X"], kind)
    base = base_risks(groups, tables)
    y = data["y"].astype(int)
    base_auc = float(roc_auc_score(y, base))
    model_auc = float(roc_auc_score(y, model_scores))
    if max(model_auc, base_auc) - min(model_auc, base_auc) < 1e-9 or base_auc < 0.5:
        weight = 0.0 if model_auc <= base_auc else 1.0
        chosen_auc = max(model_auc, base_auc)
    else:
        grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
        aucs = [float(roc_auc_score(y, w * rank_norm(model_scores) + (1 - w) * rank_norm(base)))
                for w in grid]
        best = int(np.argmax(aucs))
        weight, chosen_auc = float(grid[best]), aucs[best]
    blended = weight * rank_norm(model_scores) + (1 - weight) * rank_norm(base)
    index = 0
    by_id = {table["pair_id"]: table for table in tables}
    for entry in data["meta"]:
        table = by_id.get(entry["pair_id"])
        if table is None:
            index += entry["n"]
            continue
        for row in table["rows"]:
            row["risk_new"] = float(blended[index])
            index += 1
    rows = [{**L.cell(S.evaluate(val, tables, config, "risk_new")), "config": config}
            for config in S.grid()]
    feasible = [row for row in rows if row["fpr"] <= 0.40]
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    best_row = feasible[0] if feasible else None
    return {"auc": round(chosen_auc, 4), "base_auc_v11": round(base_auc, 4),
            "model_auc": round(model_auc, 4), "weight_model": weight,
            "within_pair_acc": within_pair_accuracy(blended, data),
            "val": {key: best_row[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")}
            if best_row else None,
            "config": best_row["config"] if best_row else None}


def auc_on_v19(model, kind, groups):
    """AUC модели на строках v19 (для вариантов без v19 в обучении — честный OOD-замер)."""
    data = groups["v19"]
    scores = score_rows(model, data["X"], kind)
    return round(float(roc_auc_score(data["y"].astype(int), scores)), 4)


def in_domain_v19(model, kind, groups):
    """In-domain AUC на v19 (2 фолда по документам: обучение на половине, оценка на другой)."""
    data = groups["v19"]
    docs = sorted({L.doc_key(pair) for pair in data["pairs"]})
    np.random.RandomState(17).shuffle(docs)
    fold_of = {doc: index % 2 for index, doc in enumerate(docs)}
    by_id = {pair["pair_id"]: pair for pair in data["pairs"]}
    row_fold = []
    for entry in data["meta"]:
        row_fold.extend([fold_of[L.doc_key(by_id[entry["pair_id"]])]] * entry["n"])
    row_fold = np.asarray(row_fold)
    aucs = []
    for fold in (0, 1):
        train_mask = row_fold != fold
        held_mask = ~train_mask
        if train_mask.sum() == 0 or held_mask.sum() == 0:
            continue
        weights, _ = hard_weights(data)
        m = fit_hgb(data["X"][train_mask], data["y"][train_mask], weights[train_mask],
                    trees=150, leaves=15)
        scores = m.predict_proba(data["X"][held_mask])[:, 1]
        aucs.append(round(float(roc_auc_score(data["y"][held_mask].astype(int), scores)), 4))
    return {"folds": aucs, "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None}


def group_cv_v19(groups, folds=3, seed=17):
    """Group-CV по документам внутри v19: обобщение на новые документы того же генератора."""
    data = groups["v19"]
    docs = sorted({L.doc_key(pair) for pair in data["pairs"]})
    np.random.RandomState(seed).shuffle(docs)
    fold_of = {doc: index % folds for index, doc in enumerate(docs)}
    by_id = {pair["pair_id"]: pair for pair in data["pairs"]}
    row_fold = []
    for entry in data["meta"]:
        row_fold.extend([fold_of[L.doc_key(by_id[entry["pair_id"]])]] * entry["n"])
    row_fold = np.asarray(row_fold)
    weights, _ = hard_weights(data)
    results = []
    for fold in range(folds):
        train_mask = row_fold != fold
        held_mask = ~train_mask
        if train_mask.sum() == 0 or held_mask.sum() == 0:
            continue
        model = fit_hgb(data["X"][train_mask], data["y"][train_mask], weights[train_mask])
        scores = model.predict_proba(data["X"][held_mask])[:, 1]
        results.append({"fold": fold, "rows": int(held_mask.sum()),
                        "positives": int(data["y"][held_mask].sum()),
                        "auc": round(float(roc_auc_score(data["y"][held_mask].astype(int), scores)), 4)})
    return {"folds": results,
            "mean_auc": round(float(np.mean([row["auc"] for row in results])), 4) if results else None}


def oof_real(groups, parts, config, folds=3, seed=17, trees=250, leaves=31):
    """OOF по документам на реальных парах: обучение на прочих документах + прочие наборы."""
    data = groups["real"]
    docs = sorted({L.doc_key(pair) for pair in data["pairs"]})
    np.random.RandomState(seed).shuffle(docs)
    fold_of = {doc: index % folds for index, doc in enumerate(docs)}
    by_id = {pair["pair_id"]: pair for pair in data["pairs"]}
    row_fold = []
    for entry in data["meta"]:
        row_fold.extend([fold_of[L.doc_key(by_id[entry["pair_id"]])]] * entry["n"])
    row_fold = np.asarray(row_fold)
    Xs, ys = stack(groups, parts)
    tables = S.build_tables(data["pairs"], L.weights(), CACHE16 / "real_tables.json")
    rows = []
    for fold in range(folds):
        train_mask = row_fold != fold
        held_mask = ~train_mask
        if train_mask.sum() == 0 or held_mask.sum() == 0:
            continue
        model = fit_hgb(np.vstack([Xs, data["X"][train_mask]]),
                        np.concatenate([ys, data["y"][train_mask]]), trees=trees, leaves=leaves)
        scores = model.predict_proba(data["X"][held_mask])[:, 1]
        held_ids = {pair["pair_id"] for pair in data["pairs"]
                    if fold_of[L.doc_key(pair)] == fold}
        held_pairs = [pair for pair in data["pairs"] if pair["pair_id"] in held_ids]
        held_tables = [table for table in tables if table["pair_id"] in held_ids]
        by_table = {table["pair_id"]: table for table in held_tables}
        index = 0
        for entry in data["meta"]:
            if entry["pair_id"] not in held_ids:
                continue
            table = by_table[entry["pair_id"]]
            for row in table["rows"]:
                row["risk_new"] = float(scores[index])
                index += 1
        cell = L.cell(S.evaluate(held_pairs, held_tables, config, "risk_new"))
        cell.update({"fold": fold, "pairs": len(held_pairs)})
        rows.append(cell)
    mean = {key: round(float(np.mean([row[key] for row in rows])), 4)
            for key in ("span", "token", "pair", "fpr")} if rows else None
    return {"folds": rows, "mean": mean, "baseline_oof_span_v14": BASELINE_OOF_SPAN,
            "pass": bool(mean and mean["span"] >= BASELINE_OOF_SPAN + 0.02)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ranker_train_v20.json")
    parser.add_argument("--ablation", default="artifacts/span_oof_v20.json")
    parser.add_argument("--v19-limit", type=int, default=0, help="0 — все пары")
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--skip-oof", action="store_true")
    args = parser.parse_args(argv)

    L.set_weights(L.weights())
    train, val, mfava = L.load_all()
    v19 = read_jsonl(ROOT / "artifacts/ru_clause_gen_v19.jsonl")
    if args.v19_limit:
        v19 = v19[:args.v19_limit]

    groups = {}
    plan = (("real", train, CACHE20 / "real.npz"),
            ("synth", read_jsonl(ROOT / "artifacts/rushallu_synth_v11.jsonl"), CACHE20 / "synth.npz"),
            ("pseudo", read_jsonl(ROOT / "artifacts/clause_pseudo_v11.jsonl"), CACHE20 / "pseudo.npz"),
            ("v19", v19, CACHE20 / "v19.npz"),
            ("val", val, CACHE20 / "val.npz"),
            ("mfava", mfava, CACHE20 / "mfava.npz"))
    for name, pairs, cache in plan:
        started = time.time()
        groups[name] = build_group(name, pairs, cache)
        print(f"{name}: {groups[name]['X'].shape} pos {int(groups[name]['y'].sum())} "
              f"hard {int(((groups[name]['y'] == 0) & (groups[name]['ious'] >= 0.3) & (groups[name]['ious'] < 0.7)).sum())} "
              f"({time.time() - started:.1f} c)", flush=True)

    tables_val = S.build_tables(val, L.weights(), CACHE16 / "val_tables.json")

    report = {
        "rule": "ранжировщик кандидатов внутри пары: 37 штатных лексических признаков v11 + "
                "декадер v11 (mode-гейт, pair-гейт, local_hgb) сохранён; пороги калибруются на val "
                "в бюджете FPR <= 0.40; held-out не читается",
        "features": {"lexical": 37},
        "data": {name: {"pairs": len(groups[name]["pairs"]),
                        "candidates": int(groups[name]["X"].shape[0]),
                        "positives": int(groups[name]["y"].sum())}
                 for name in groups},
        "hard_negatives": {"definition": "кандидат-негатив с IoU 0.3–0.7 к gold внутри пары",
                           "weight": 2.0},
        "variants": {},
        "reference": {"v11_risk_val": {"span": 0.3621, "token": 0.6247, "pair": 0.5122,
                                       "fpr": 0.1442, "auc": 0.7425}},
    }

    def save():
        (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")

    variants = {}
    specs = (
        ("control_37", ["real", "synth", "pseudo"], False, "hgb"),
        ("v19_only", ["v19"], False, "hgb"),
        ("v19_plus_real", ["real", "synth", "pseudo", "v19"], False, "hgb"),
        ("v19_plus_real_hardneg", ["real", "synth", "pseudo", "v19"], True, "hgb"),
        ("full_pairwise", ["real", "synth", "pseudo", "v19"], True, "pairwise"),
    )
    for label, parts, use_hard, kind in specs:
        started = time.time()
        if kind == "pairwise":
            X, y, weights = pairwise_dataset(groups, parts)
            model = LogisticRegression(max_iter=3000, C=0.5).fit(X, y, sample_weight=weights)
            in_domain = in_domain_v19(model, "pairwise", groups)
        else:
            X, y = stack(groups, parts)
            weights = None
            hard = 0
            if use_hard:
                blocks, hards = [], 0
                for part in parts:
                    weight, count = hard_weights(groups[part])
                    blocks.append(weight)
                    hards += count
                weights = np.concatenate(blocks)
                hard = hards
            model = fit_hgb(X, y, weights, args.trees, args.leaves, args.lr)
            in_domain = in_domain_v19(model, "hgb", groups)
        entry = val_cell(model, kind, groups, val, tables_val)
        entry.update({"train_rows": int(X.shape[0]), "train_positives": int(np.sum(y)),
                      "hard_negatives": int(hard) if kind != "pairwise" else None,
                      "in_domain_v19": in_domain,
                      "auc_on_v19": auc_on_v19(model, kind, groups),
                      "trained_on_v19": "v19" in parts,
                      "fit_seconds": round(time.time() - started, 1)})
        entry["blend_with_v11"] = blend_cell(model, kind, groups, val, tables_val)
        report["variants"][label] = entry
        variants[label] = model
        save()
        print(label, "val_auc", entry["auc"], "within", entry["within_pair_acc"],
              "val", entry["val"], "in_domain", in_domain["mean_auc"],
              "on_v19", entry["auc_on_v19"], "blend", entry["blend_with_v11"]["val"],
              flush=True)

    candidates = []
    for label, entry in report["variants"].items():
        candidates.append((label, entry, (entry["val"] or {"span": -1})["span"]))
        blend = entry.get("blend_with_v11") or {}
        if blend.get("val"):
            candidates.append((f"{label}__blend_v11", entry, blend["val"]["span"]))
    best_label, best_entry, _ = max(candidates, key=lambda item: (item[2], -item[1]["val"]["fpr"]))
    blend_key = "__blend_v11"
    if best_label.endswith(blend_key):
        base_label = best_label[:-len(blend_key)]
        report["best"] = {"variant": best_label, "base_variant": base_label,
                          "mode": "blend_with_v11",
                          **{key: best_entry["blend_with_v11"][key]
                             for key in ("auc", "within_pair_acc", "val", "config")},
                          "in_domain_v19": best_entry["in_domain_v19"],
                          "weight_model": best_entry["blend_with_v11"]["weight_model"]}
    else:
        report["best"] = {"variant": best_label, "mode": "model_only",
                          **{key: best_entry[key]
                             for key in ("auc", "within_pair_acc", "val", "config", "in_domain_v19")}}
    report["group_cv_v19"] = group_cv_v19(groups)
    print("group_cv_v19", report["group_cv_v19"], flush=True)

    base_label = report["best"].get("base_variant", best_label)
    best_parts = dict((label, parts) for label, parts, _, _ in specs)[base_label]
    if not args.skip_oof:
        oof_parts = [part for part in best_parts if part != "real"]
        report["oof_real"] = oof_real(groups, oof_parts,
                                      report["variants"][base_label]["config"] or
                                      {"mode_cut": 0.5, "gate": 0.3, "threshold": 0.4,
                                       "max_spans": 3, "local_threshold": 0.4,
                                       "local_max_spans": 3, "nms_iou": 0.1})
        print("oof_real", report["oof_real"]["mean"], "pass", report["oof_real"]["pass"], flush=True)
    save()

    best_val = report["best"]["val"] or {}
    guard = {
        "span": {"value": best_val.get("span"), "need": 0.50,
                 "pass": bool(best_val.get("span", 0) >= 0.50)},
        "token": {"value": best_val.get("token"), "need": 0.50,
                  "pass": bool(best_val.get("token", 0) >= 0.50)},
        "pair": {"value": best_val.get("pair"), "need": 0.50,
                 "pass": bool(best_val.get("pair", 0) >= 0.50)},
        "fpr": {"value": best_val.get("fpr"), "need": 0.40,
                "pass": bool(best_val.get("fpr", 1) <= 0.40)},
        "oof": {"value": (report.get("oof_real") or {}).get("mean", {}).get("span"),
                "need": round(BASELINE_OOF_SPAN + 0.02, 4),
                "pass": bool((report.get("oof_real") or {}).get("pass"))},
    }
    guard["pass_all"] = all(item["pass"] for item in guard.values())
    report["guard"] = guard
    report["verdict"] = "candidate" if guard["pass_all"] else (
        "reject_by_span" if not guard["span"]["pass"] else "reject_by_guard")
    save()

    ablation = {
        "rule": "ablation по множествам обучения: cross-domain AUC (наш val), in-domain AUC (v19, "
                "2 фолда по документам), Group-CV по документам v19, OOF по документам на реальных парах",
        "reference_v11": report["reference"]["v11_risk_val"],
        "variants": {label: {"train_parts": parts, "hard_negatives": use_hard, "kind": kind,
                             "val_auc": report["variants"][label]["auc"],
                             "within_pair_acc": report["variants"][label]["within_pair_acc"],
                             "in_domain_v19_auc": report["variants"][label]["in_domain_v19"]["mean_auc"],
                             "auc_on_v19": report["variants"][label]["auc_on_v19"],
                             "trained_on_v19": report["variants"][label]["trained_on_v19"],
                             "blend_with_v11": report["variants"][label]["blend_with_v11"],
                             "val": report["variants"][label]["val"],
                             "config": report["variants"][label]["config"]}
                     for label, parts, use_hard, kind in specs},
        "group_cv_v19": report["group_cv_v19"],
        "oof_real": report.get("oof_real"),
        "best_variant": best_label,
        "guard": guard,
        "verdict": report["verdict"],
    }
    (ROOT / args.ablation).write_text(json.dumps(ablation, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print("best", best_label, best_val, report["verdict"], flush=True)


if __name__ == "__main__":
    main()
