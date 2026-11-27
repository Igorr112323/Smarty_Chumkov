"""V20 доп-вариант: переобучение самого линейного риска v11 (37 признаков → logit) на v19.

Гипотеза: GBDT на 37 признаках проигрывает ручному линейному риску v11 (AUC 0.7425) и плохо
переносится; линейная модель с теми же признаками может перенестись лучше, потому что не строит
пороговых разрывов. Форма риска и декодер не меняются — только веса (это «ранжировщик на v19»).

Сравниваются: линейный риск, обученный без v19 и с v19; оценка в том же декадере v11 через
квантильное выравнивание шкал (смесь с baseline при w: 0 = baseline, 1 = чистая модель).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import sweep_v16 as S
import v11_lab as L
import train_rank_v20 as T
import blend_v20 as B


def fit_linear(groups, parts, hard_weight=2.0):
    X, y = T.stack(groups, parts)
    weights = np.concatenate([T.hard_weights(groups[part], hard_weight)[0] for part in parts])
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=4000, C=0.3).fit(scaler.transform(X), y,
                                                       sample_weight=weights)
    return scaler, model


def scores_for(groups, part, scaler, model):
    return model.predict_proba(scaler.transform(groups[part]["X"]))[:, 1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/linear_v20.json")
    parser.add_argument("--step", type=float, default=0.2)
    args = parser.parse_args(argv)

    groups = B.load_groups()
    train, val, _ = L.load_all()
    tables = S.build_tables(val, L.weights(), T.CACHE16 / "val_tables.json")
    base = T.base_risks(groups, tables)
    reference = B.best_cell(groups, val, tables, base)
    print("reference", reference["span"], reference["pair"], flush=True)

    report = {"rule": "переобучение линейного риска v11 (37 признаков) на реальных + синтетических "
                      "данных, с v19 и без; форма риска и декодер v11 сохранены",
              "reference_v11": reference, "variants": {}}
    weights_grid = np.round(np.arange(0.0, 1.0001, args.step), 2)
    for label, parts in (("linear_without_v19", ["real", "synth", "pseudo"]),
                         ("linear_with_v19", ["real", "synth", "pseudo", "v19"]),
                         ("linear_v19_only", ["v19"])):
        scaler, model = fit_linear(groups, parts)
        val_scores = scores_for(groups, "val", scaler, model)
        v19_scores = scores_for(groups, "v19", scaler, model)
        entry = {"parts": parts,
                 "val_auc": round(float(roc_auc_score(groups["val"]["y"].astype(int), val_scores)), 4),
                 "auc_on_v19": round(float(roc_auc_score(groups["v19"]["y"].astype(int), v19_scores)), 4),
                 "grid": []}
        mapped = B.quantile_map(base, val_scores)
        for weight in weights_grid:
            values = (1 - weight) * base + weight * mapped
            cell = B.best_cell(groups, val, tables, values)
            if cell is None:
                continue
            entry["grid"].append({"weight_model": round(float(weight), 2),
                                  "auc": round(float(roc_auc_score(groups["val"]["y"].astype(int),
                                                                   values)), 4), **cell})
        entry["best"] = max(entry["grid"], key=lambda row: (row["span"], -row["fpr"]))
        report["variants"][label] = entry
        print(label, "val_auc", entry["val_auc"], "on_v19", entry["auc_on_v19"],
              "best span", entry["best"]["span"], "w", entry["best"]["weight_model"],
              "fpr", entry["best"]["fpr"], flush=True)

    b = report["variants"]
    report["check"] = {
        "hypothesis_holds": bool(max(entry["best"]["span"] for entry in b.values())
                                 > reference["span"]),
        "best": max(b, key=lambda key: (b[key]["best"]["span"])),
    }
    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report["check"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
