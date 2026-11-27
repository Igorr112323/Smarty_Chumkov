"""V17: сборка артефактов отчёта (ранкер-фичи, span_oof, вердикт)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from sklearn.metrics import roc_auc_score
from spanver.evaluate import iou
import v11_lab as L

train, val, mfava = L.load_all()
feat = json.loads((ROOT / "artifacts/ranker_features_v17.json").read_text())
X = np.load(ROOT / "artifacts/cache_v17/val_span.npz")["X"]
cols = list(np.load(ROOT / "artifacts/cache_v17/val_span.npz")["columns"])
tables = json.loads((ROOT / "artifacts/cache_v16/val_tables.json").read_text())
by = {t["pair_id"]: t for t in tables}
labels = []
for pair in val:
    rows = by[pair["pair_id"]]["rows"]; gold = pair.get("spans") or []
    for row in rows:
        labels.append(any(iou(row["start"], row["end"], g["start"], g["end"]) >= 0.5 for g in gold))
labels = np.asarray(labels)
auc = {}
for k, name in enumerate(cols):
    value = roc_auc_score(labels, X[:, k])
    auc[name] = {"auc": round(float(max(value, 1 - value)), 4), "direction": "+" if value >= 0.5 else "-"}
feat["usefulness_val"] = {
    "candidates": int(X.shape[0]), "positives": int(labels.sum()),
    "auc_per_feature": auc,
    "best": max(auc.items(), key=lambda item: item[1]["auc"])[0],
    "conclusion": "новые span-level признаки дают AUC 0.50–0.60 по отдельности; штатный "
                  "лексический риск v11 на тех же кандидатах — 0.7425, поэтому они идут только "
                  "как добавка к 37 признакам, а не как самостоятельный скор",
}
train_report = json.loads((ROOT / "artifacts/ranker_train_v17.json").read_text())
oof = json.loads((ROOT / "artifacts/oof_ranker_v17.json").read_text())
speed = json.loads((ROOT / "artifacts/speed_v17.json").read_text())
v16 = json.loads((ROOT / "artifacts/span_oof_v16.json").read_text())

payload = {
    "rule": "ранжировщик внутри пары: 37 лексических + 14 span-level кросс-внимание + 10 per-token "
            "LM; декодер v11 сохранён; пороги калибруются на val в бюджете FPR ≤ 0.40",
    "reference": {"v11_risk_auc_val": 0.7425, "v11_calibrated_val": v16["calibration"]["val"],
                  "oracle_within_pair_span_f1": v16["oracle_ceiling"]["max_spans_1"]["span_f1"],
                  "candidate_reachability": v16["candidate_reachability"]},
    "variants": {name: {"auc": entry["auc"], "within_pair_acc": entry["within_pair_acc"],
                        "val": entry["val"], "config": entry["config"]}
                 for name, entry in train_report["variants"].items()},
    "widest": {"task_target_auc": 0.88,
               "best_auc": max(entry["auc"] for entry in train_report["variants"].values()),
               "delta_to_target": round(0.88 - max(entry["auc"] for entry in train_report["variants"].values()), 4)},
    "group_cv_sber": train_report["group_cv_sber"],
    "oof_hgb_61": oof,
    "speed": {key: speed["runs"][key] for key in speed["runs"]},
    "guard": {"val_span_min": 0.50, "val_token_min": 0.50, "val_pair_min": 0.50,
              "val_fpr_max": 0.40, "oof_delta": 0.02},
    "verdict": "reject",
    "why": "ни один вариант не превысил AUC 0.7425 штатного риска v11 (лучший 0.6921 у "
           "pairwise-логита, у HGB 0.6451); на новых данных Group-CV даёт AUC 0.981 — модель "
           "выучивает синтетические правки, но перенос на реальные клаузовые галлюцинации "
           "отрицательный (val AUC падает, OOF Span 0.0426)",
}
(ROOT / "artifacts/ranker_features_v17.json").write_text(json.dumps(feat, ensure_ascii=False, indent=2) + "\n")
(ROOT / "artifacts/span_oof_v17.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print("best auc", payload["widest"]["best_auc"], "target", payload["widest"]["task_target_auc"])
print("group cv", payload["group_cv_sber"]["mean_auc"], "oof span", oof["mean"]["span"])
print("speed", {k: (v["seconds"], v["pass"]) for k, v in payload["speed"].items()})
