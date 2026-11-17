import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, token_positions


def counts(rows):
    support = flagged = negatives = negative_flagged = 0
    for row in rows:
        text = row["answer"]
        truth = row["gold_spans"]
        predicted = row["predicted_spans"]
        g = covered_tokens(text, [(span["start"], span["end"]) for span in truth])
        p = covered_tokens(text, [(span["start"], span["end"]) for span in predicted])
        support += len(token_positions(text)) - len(g)
        flagged += len(p - g)
        if not truth:
            negatives += 1
            negative_flagged += bool(predicted)
    return {"negative_tokens": support, "flagged_negative_tokens": flagged, "fpr_token": flagged / support if support else None, "negative_pairs": negatives, "flagged_negative_pairs": negative_flagged, "fpr_pair": negative_flagged / negatives if negatives else None}


def main():
    manifest = json.loads((ROOT / "artifacts/predictions40_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "baseline nested38 recovered, not round40" or len(manifest["outputs"]) != 5:
        raise ValueError("ожидались пять внешних фолдов baseline")
    folds = []
    for index, item in enumerate(manifest["outputs"], 1):
        raw = (ROOT / item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("SHA не совпал")
        rows = [json.loads(line) for line in raw.splitlines() if line]
        if len(rows) != item["pairs"] or any(row["fold"] != index or row["source"] != "nested38_baseline" for row in rows):
            raise ValueError("неверный источник")
        folds.append({"fold": index, **counts(rows)})
    if any(row["fpr_token"] is None for row in folds):
        raise ValueError("нет отрицательных токенов")
    worst = max(folds, key=lambda row: row["fpr_token"])
    points = [{"overlap_percent": threshold, "gap_exclusive_chars": gap, "worst_outer_fpr_token_lower_bound": worst["fpr_token"], "measured_inner_metrics": None, "selected": False} for threshold in (20, 30, 40, 50, 60) for gap in (0, 10, 20, 40)]
    output = {"status": "нижняя граница на сохранённых внешних фолдах; не вложенная сетка", "proof": "Любой B3 только расширяет каждый исходный спан или объединяет пересекающиеся; множество покрытых токенов не уменьшается. Разметка и знаменатель FPR_token фиксированы, поэтому FPR_token для каждой пары и фолда не может уменьшиться. Только после прежнего gate и декодирования, без пересчёта порогов.", "baseline_folds": folds, "worst_outer_fold": worst["fold"], "worst_outer_fpr_token_lower_bound": worst["fpr_token"], "worst_fpr_pair_unchanged": max(row["fpr_pair"] for row in folds), "inner_curve_status": "нет сохранённых пообъектных внутренних предсказаний; 20 вложенных значений и отбор не вычислены", "points": points}
    (ROOT / "artifacts/fpr43_feasibility.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "points"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
