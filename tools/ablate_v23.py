"""V23: добивка абляций и диагностика потолка recall.

Первый прогон `fpr_sweep_v23.py` оставил в таблицах подменённый risk_local от последнего варианта
риска (`ce+local`), из-за чего абляции `multi_span` и `snap` считались не на том конфиге.
Здесь: восстановить штатный risk_local v11, честно пересчитать абляции на выбранном конфиге и
посчитать диагностику — сколько gold-спанов вообще достижимо при текущем пара-гейте и пуле
кандидатов (это и есть объяснение, почему FPR-бюджет не поднимает Span).

Результат вливается в `artifacts/fpr_sweep_v23.json` (секции `multi_span_ablation`, `snap_ablation`,
`ceiling_diagnostics`) и в `artifacts/span_oof_v23.json` (секция `ablations`).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np

import ce_train_cpu_v21 as C
import fpr_sweep_v23 as V
import sweep_v16 as S
import v11_lab as L
from spanver.evaluate import iou as iou_check


def main():
    report = json.loads((ROOT / "artifacts/fpr_sweep_v23.json").read_text(encoding="utf-8"))
    best = report["best_under_0.40"]
    config = best["config"]
    risk_key = {"v11": "risk_cand", "ce": "risk_new", "blend_0.4": "risk_blend",
                "rank_blend": "risk_rank", "z_blend": "risk_z"}[best["risk"]]

    train_pairs, val_pairs, _ = L.load_all()
    tables = S.build_tables(val_pairs, L.weights(), ROOT / "artifacts/cache_v16/val_tables.json")
    # штатный risk_local v11: пересобираем из чистого кэша признаков (в таблицах он уже подменён)
    fresh = S.build_tables(val_pairs, L.weights(), ROOT / "artifacts/cache_v16/val_tables_fresh.json")
    original_local = {}
    for table in fresh:
        for row in table["rows"]:
            original_local[(table["pair_id"], row["start"], row["end"])] = row["risk_local"]
    by_pair = {table["pair_id"]: table for table in tables}
    for table in tables:
        for row in table["rows"]:
            row["risk_local"] = original_local[(table["pair_id"], row["start"], row["end"])]

    # скоры CE и варианты риска: в кэше таблиц их нет, поэтому считаем заново (как в основном скрипте)
    import torch

    from ce_model_v21 import build_model, load_tokenizer

    torch.set_num_threads(2)
    tokenizer = load_tokenizer()
    model = build_model()
    model.load_state_dict(torch.load(ROOT / "artifacts/ce_model_cpu_v23.pt", map_location="cpu")["state_dict"])
    model.eval()
    tensors, meta = C.pack(val_pairs, tokenizer, 64, 128, max_easy=-1)
    values = C.scores(model, tensors)
    base = []
    for entry in meta:
        row = next(row for row in by_pair[entry["pair_id"]]["rows"]
                   if row["start"] == entry["start"] and row["end"] == entry["end"])
        base.append(row["risk_cand"])
    base = np.asarray(base, dtype=np.float64)
    mapped = V.quantile_map(base, values)
    blend = 0.6 * base + 0.4 * mapped
    C.apply_to_tables(tables, meta, values, "risk_new")
    C.apply_to_tables(tables, meta, blend, "risk_blend")
    C.apply_to_tables(tables, meta, V.rank_within_pairs(meta, blend), "risk_rank")
    C.apply_to_tables(tables, meta, V.zscore_within_pairs(meta, blend), "risk_z")

    answers = {pair["pair_id"]: pair.get("model_output") or "" for pair in val_pairs}

    multi = {}
    for max_spans in (1, 2, 3, 5, 999):
        multi[str(max_spans)] = V.evaluate(val_pairs, tables, dict(config) | {"max_spans": max_spans},
                                           risk_key, answers=answers)
    snap = {}
    for mode in ("none", "expand", "shrink", "nearest"):
        snap[mode] = V.evaluate(val_pairs, tables, config, risk_key, answers=answers, snap_mode=mode)

    # ---- диагностика потолка -------------------------------------------------
    gold_counts, reachable, unreachable = 0, 0, []
    gate_stats = {}
    for pair in val_pairs:
        gold = pair.get("spans") or []
        table = by_pair[pair["pair_id"]]
        if not gold:
            continue
        gold_counts += len(gold)
        rows = table["rows"]
        for truth in gold:
            hit = any(iou_check(row["start"], row["end"], truth["start"], truth["end"]) >= 0.5
                      for row in rows)
            if hit:
                reachable += 1
            else:
                unreachable.append({"pair_id": pair["pair_id"], "start": truth["start"],
                                    "end": truth["end"], "answer_chars": len(pair.get("model_output") or "")})
    for gate in (0.0, 0.2, 0.3, 0.4, 0.5):
        passed = total = 0
        for table in tables:
            if table["mode_prob"] >= config["mode_cut"]:
                continue
            total += 1
            if table["pair_prob"] >= gate:
                passed += 1
        gate_stats[f"{gate:.1f}"] = {"pairs_in_span_mode": total, "pass_gate": passed,
                                     "share": round(passed / total, 4) if total else None}

    ceiling = {
        "gold_spans": gold_counts,
        "gold_spans_with_matching_candidate": reachable,
        "oracle_recall": round(reachable / gold_counts, 4) if gold_counts else None,
        "unreachable_examples": unreachable[:10],
        "gate_pass_share": gate_stats,
        "note": ("FPR-бюджет ограничивает ложные срабатывания на ЧИСТЫХ парах, а Span теряется "
                 "в парах с gold: там лишние спаны сразу становятся FP. Поэтому поднятие бюджета "
                 "не даёт recall, а только снижает precision"),
    }

    report["multi_span_ablation"] = multi
    report["snap_ablation"] = snap
    report["ceiling_diagnostics"] = ceiling
    report["ablation_note"] = ("абляции пересчитаны после восстановления штатного risk_local v11: "
                               "в первом прогоне local-риск был подменён последним вариантом (ce+local)")
    (ROOT / "artifacts/fpr_sweep_v23.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    span_oof = json.loads((ROOT / "artifacts/span_oof_v23.json").read_text(encoding="utf-8"))
    span_oof["ablations"]["multi_span"] = multi
    span_oof["ablations"]["snap_adjuster"] = snap
    span_oof["ceiling_diagnostics"] = ceiling
    span_oof["val"]["chosen"] = best
    (ROOT / "artifacts/span_oof_v23.json").write_text(
        json.dumps(span_oof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("multi-span:", json.dumps({k: v["span"] for k, v in multi.items()}, ensure_ascii=False))
    print("snap:", json.dumps({k: (v["span"], v["token"]) for k, v in snap.items()}, ensure_ascii=False))
    print("ceiling:", json.dumps({k: ceiling[k] for k in
                                  ("gold_spans", "gold_spans_with_matching_candidate", "oracle_recall")},
                                 ensure_ascii=False))
    print("gate:", json.dumps(gate_stats, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
