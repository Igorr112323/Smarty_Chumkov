"""V16: статистика 5 LM-признаков (§4.1) и их полезность для ранжирования кандидатов."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from sklearn.metrics import roc_auc_score
import v11_lab as L

def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--report", default="artifacts/lm_features_v16.json")
    args = parser.parse_args(argv)
    base = json.loads((ROOT / "artifacts/lm_features_v15.json").read_text(encoding="utf-8"))
    out = {"rule": "5 признаков на токен ответа (энтропия внимания, доля внимания на документ, пик "
                   "внимания, косинус к среднему документу, косинус к ближайшему токену документа), "
                   "mean/max по кандидату; модель — rubert-tiny2 (dequant из int8 ONNX), eval без дообучения",
           "columns": base["columns"], "input": base["input"],
           "weights": "models/rubert-tiny2-deq (см. artifacts/weights_v16.json)", "groups": {}}
    for name, entry in base["groups"].items():
        out["groups"][name] = {k: entry.get(k) for k in ("pairs", "candidates", "positives", "seconds", "seconds_per_1k_chars")}
        out["groups"][name]["feature_stats"] = entry.get("feature_stats")
    # полезность на val: AUC каждого признака и AUC штатного риска
    val = np.load(ROOT / "artifacts/cache_v15/val_lm.npz")
    X, y = val["X"], val["y"].astype(int)
    auc = {}
    for index, column in enumerate(base["columns"]):
        value = roc_auc_score(y, X[:, index]) if len(set(y.tolist())) > 1 else None
        auc[column] = round(max(value, 1 - value), 4) if value is not None else None
    tables = json.loads((ROOT / "artifacts/cache_v16/val_tables.json").read_text(encoding="utf-8"))
    risk = np.concatenate([[row["risk_cand"] for row in table["rows"]] for table in tables]) if tables else np.zeros(0)
    out["usefulness_val"] = {
        "candidates": int(X.shape[0]), "positives": int(y.sum()),
        "auc_per_feature": auc,
        "auc_v11_risk": round(float(roc_auc_score(y, risk[:X.shape[0]])), 4) if len(risk) >= X.shape[0] else None,
        "summary": "LM-признаки по отдельности дают AUC 0.5–0.63, штатный риск v11 — заметно выше; "
                   "в v15 добавление всех 10 колонок к 37 признакам меняло AUC на val в пределах "
                   "+0.007 (смесь) / -0.03 (русские пары)",
    }
    (ROOT / args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"auc_per_feature": auc, "auc_v11_risk": out["usefulness_val"]["auc_v11_risk"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
