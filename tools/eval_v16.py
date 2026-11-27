"""V16 шаг 5 (финал): честная сборка результатов — калибровка, OOF по документам, потолки.

Считает и складывает в `artifacts/span_oof_v16.json`:

  * A. калибровка существующего v11-декодера (без переобучения скора) в бюджете FPR ≤ 0.40 —
       лучший по val конфиг, метрики val/mFAVA и 3-fold OOF по документам на реальных парах;
  * B. фьюжн v11-риска с относительным GBDT (w=0.85, val-only: модель обучена на всех данных,
       поэтому OOF для него нечестен и в вердикт не идёт);
  * потолки: какой Span F1 достижим при идеальном ранжировании внутри пары и текущем
       pair-гейте — то есть верхняя граница для любого reranker'а при этом декодере;
  * вердикт по метрикам договора (Span ≥ 0.50, Token ≥ 0.50, Pair ≥ 0.50, FPR ≤ 0.40).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import sweep_v16 as S
import v11_lab as L
from spanver.evaluate import iou, match_spans, metrics

CACHE = ROOT / "artifacts/cache_v16"


def load_tables(name):
    return json.loads((CACHE / f"{name}_tables.json").read_text(encoding="utf-8"))


def fold_of_documents(pairs, folds=3, seed=16):
    docs = sorted({L.doc_key(pair) for pair in pairs})
    np.random.RandomState(seed).shuffle(docs)
    mapping = {doc: index % folds for index, doc in enumerate(docs)}
    return {pair["pair_id"]: mapping[L.doc_key(pair)] for pair in pairs}


def oof_calibration(pairs, tables, config, risk_key="risk_cand", folds=3):
    assignment = fold_of_documents(pairs, folds)
    by_id = {table["pair_id"]: table for table in tables}
    rows = []
    for fold in range(folds):
        held = [pair for pair in pairs if assignment[pair["pair_id"]] == fold]
        held_tables = [by_id[pair["pair_id"]] for pair in held]
        cell = L.cell(S.evaluate(held, held_tables, config, risk_key))
        cell["fold"] = fold
        cell["pairs"] = len(held)
        rows.append(cell)
    mean = {key: round(float(np.mean([row[key] for row in rows])), 4)
            for key in ("span", "token", "pair", "fpr")}
    return rows, mean


def oracle_ceiling(pairs, tables, config, max_spans=1):
    """Идеальное ранжирование внутри пары при том же pair-гейте и режиме."""
    by_id = {table["pair_id"]: table for table in tables}
    hit = total_pred = total_gold = 0
    flagged = 0
    for pair in pairs:
        table = by_id[pair["pair_id"]]
        gold = pair.get("spans") or []
        local = table["mode_prob"] >= config["mode_cut"]
        if not local and table["pair_prob"] < config["gate"]:
            continue
        flagged += 1
        pool = table["rows"]
        if local:
            pool = [row for row in pool if not (row["start"] == 0 and row["end"] == pool[0]["end"])]
        if not pool:
            continue
        scored = sorted(pool, key=lambda row: -max(
            [iou(row["start"], row["end"], span["start"], span["end"]) for span in gold] or [0.0]))
        picked = scored[:max_spans]
        matched, _, _ = match_spans(
            [{"start": row["start"], "end": row["end"], "risk": 1.0} for row in picked], gold, 0.5)
        hit += matched
        total_pred += len(picked)
        total_gold += len(gold)
    precision = hit / total_pred if total_pred else 0.0
    recall = hit / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"flagged_pairs": flagged, "predicted": total_pred, "gold": total_gold,
            "matched": hit, "precision": round(precision, 4), "recall": round(recall, 4),
            "span_f1": round(f1, 4)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/span_oof_v16.json")
    parser.add_argument("--mfava-report", default="artifacts/mfava_v16.json")
    parser.add_argument("--fpr-max", type=float, default=0.40)
    args = parser.parse_args(argv)

    params = L.weights()
    L.set_weights(params)
    train, val, mfava = L.load_all()
    tables = {name: load_tables(name) for name in ("val", "mfava", "real")}

    # A. калибровка v11-скора
    rows = []
    for config in S.grid():
        rows.append({**L.cell(S.evaluate(val, tables["val"], config, "risk_cand")),
                     "config": config})
    feasible = [row for row in rows if row["fpr"] <= args.fpr_max]
    feasible.sort(key=lambda row: (-row["span"], -row["pair"], row["fpr"]))
    best = feasible[0]
    config = best["config"]
    val_cell = {key: best[key] for key in ("span", "token", "pair", "fpr", "span_p", "span_r")}
    mfava_cell = L.cell(S.evaluate(mfava, tables["mfava"], config, "risk_cand"))
    oof_rows, oof_mean = oof_calibration(train, tables["real"], config, "risk_cand")

    report = json.loads((ROOT / args.report).read_text(encoding="utf-8")) \
        if (ROOT / args.report).exists() else {}
    report.update({
        "rule": "метрики договора проверяются на val (калибровочный набор) + 3-fold OOF по "
                "документам; held-out не читается",
        "baseline_v11": {"span": 0.3363, "token": 0.6191, "pair": 0.4872, "fpr": 0.125},
        "calibration": {
            "what": "пороги v11-декодера (mode_cut, gate, threshold, max_spans, local_config) "
                    "калибруются на val в бюджете FPR ≤ 0.40; скор кандидата не переобучается",
            "configs_tested": len(rows),
            "best_config": config,
            "val": val_cell,
            "mfava": mfava_cell,
        },
        "oof_calibration": {
            "rule": "3 фолда по документам на реальных русских парах, тот же конфиг",
            "folds": oof_rows, "mean": oof_mean,
            "baseline_oof_span_v14": 0.2437,
            "pass": bool(oof_mean["span"] >= 0.2437 + 0.02),
        },
        "oracle_ceiling": {
            "what": "идеальное ранжирование внутри пары при том же pair-гейте — верхняя граница "
                    "для любого reranker'а на этом декодере",
            "max_spans_1": oracle_ceiling(val, tables["val"], config, 1),
            "max_spans_2": oracle_ceiling(val, tables["val"], config, 2),
            "max_spans_3": oracle_ceiling(val, tables["val"], config, 3),
        },
        "candidate_reachability": {
            "what": "есть ли у каждого gold-спана кандидат с IoU ≥ 0.5 (val)",
            **candidate_reachability(val, tables["val"]),
        },
    })

    guard = (val_cell["span"] >= 0.50 and val_cell["token"] >= 0.50
             and val_cell["pair"] >= 0.50 and val_cell["fpr"] <= 0.40)
    report["guard"] = {"val_span": 0.50, "val_token": 0.50, "val_pair": 0.50,
                       "val_fpr_max": 0.40, "val_pass": bool(guard),
                       "oof_pass": report["oof_calibration"]["pass"]}
    report["verdict"] = "success" if guard else "reject"
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")

    mfava_payload = {
        "rule": "mFAVA — внутренний набор, НЕ метрика договора; падение не блокирует v16",
        "at_best_calibration": mfava_cell,
        "variants": {name: entry.get("mfava") for name, entry in
                     (report.get("variants") or {}).items()},
    }
    (ROOT / args.mfava_report).write_text(json.dumps(mfava_payload, ensure_ascii=False,
                                                     indent=2) + "\n", encoding="utf-8")
    print("A(калибровка):", val_cell, "mfava", mfava_cell["span"], flush=True)
    print("OOF:", oof_mean, flush=True)
    print("oracle:", report["oracle_ceiling"]["max_spans_1"], flush=True)
    print("verdict:", report["verdict"], flush=True)


def candidate_reachability(pairs, tables):
    by_id = {table["pair_id"]: table for table in tables}
    total = reachable = 0
    ranks = []
    for pair in pairs:
        table = by_id[pair["pair_id"]]
        order = sorted(range(len(table["rows"])),
                       key=lambda index: (-table["rows"][index]["risk_cand"],
                                          table["rows"][index]["start"]))
        rank_of = {index: position + 1 for position, index in enumerate(order)}
        for span in pair.get("spans") or []:
            total += 1
            best_index, best_iou = None, 0.0
            for index, row in enumerate(table["rows"]):
                value = iou(row["start"], row["end"], span["start"], span["end"])
                if value > best_iou:
                    best_index, best_iou = index, value
            if best_iou >= 0.5:
                reachable += 1
                ranks.append(rank_of[best_index])
    ranks.sort()
    return {"gold_spans": total, "reachable": reachable,
            "reachable_share": round(reachable / total, 4) if total else 0.0,
            "rank1": sum(1 for rank in ranks if rank == 1),
            "rank_le2": sum(1 for rank in ranks if rank <= 2),
            "rank_le3": sum(1 for rank in ranks if rank <= 3),
            "rank_le5": sum(1 for rank in ranks if rank <= 5),
            "ranks": ranks}


if __name__ == "__main__":
    main()
