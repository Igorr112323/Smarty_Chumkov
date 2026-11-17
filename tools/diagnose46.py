import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.decode import spans_from_flags
from spanver.evaluate import covered_tokens, token_positions
from tools.evaluate_snap42 import compute


def without_snap(row):
    pieces = row["pieces"]
    if max((piece[2] for piece in pieces), default=0.0) < row["gate_threshold"]:
        return []
    flags = []
    active = False
    for _, _, score in pieces:
        active = score >= (row["extend_threshold"] if active else row["start_threshold"])
        flags.append(active)
    return spans_from_flags(row["answer"], pieces, flags, 2)


def tally(rows):
    negatives = false_positives = negative_pairs = false_pairs = 0
    for row in rows:
        answer = row["answer"]
        gold = row["gold_spans"]
        predicted = row["predicted_spans"]
        gold_tokens = covered_tokens(answer, [(span["start"], span["end"]) for span in gold])
        predicted_tokens = covered_tokens(answer, [(span["start"], span["end"]) for span in predicted])
        negatives += len(token_positions(answer)) - len(gold_tokens)
        false_positives += len(predicted_tokens - gold_tokens)
        if not gold:
            negative_pairs += 1
            false_pairs += bool(predicted)
    return {"negative_tokens": negatives, "false_positive_tokens": false_positives, "negative_pairs": negative_pairs, "false_positive_pairs": false_pairs}


def summarize(folds, counts):
    names = ("token_precision", "token_recall", "token_f1", "exact_f1", "overlap_f1", "pair_f1", "fpr_pair", "fpr_token")
    total = {key: sum(row[key] for row in counts) for key in counts[0]}
    return {"macro_means": {name: round(statistics.mean(row[name] for row in folds), 4) for name in names}, "max_fpr_token": max(row["fpr_token"] for row in folds), "max_fpr_pair": max(row["fpr_pair"] for row in folds), "pooled_counts": total, "pooled_fpr_token": total["false_positive_tokens"] / total["negative_tokens"], "pooled_fpr_pair": total["false_positive_pairs"] / total["negative_pairs"], "folds": folds}


def main():
    manifest = json.loads((ROOT / "artifacts/predictions40_manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "artifacts/nested38_summary.json").read_text(encoding="utf-8"))
    if len(manifest["outputs"]) != 5:
        raise ValueError("нужно пять файлов исходных предсказаний")
    result = {"nested38": [], "without_expand_fixed_thresholds": []}
    counts = {key: [] for key in result}
    ids = set()
    for index, item in enumerate(manifest["outputs"]):
        payload = (ROOT / item["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError("неверный SHA исходных предсказаний")
        rows = [json.loads(line) for line in payload.splitlines() if line]
        if len(rows) != item["pairs"] or any(row["fold"] != index + 1 or row["source"] != "nested38_baseline" for row in rows):
            raise ValueError("неверный источник предсказаний")
        for row in rows:
            if row["pair_id"] in ids:
                raise ValueError("повтор pair_id")
            ids.add(row["pair_id"])
        old = compute(rows)
        reference = expected["folds"][index]["metrics"]
        for key in ("token_precision", "token_recall", "token_f1", "pair_f1", "fpr_pair", "fpr_token", "predicted_spans"):
            if old[key] != reference[key]:
                raise ValueError(f"историческая метрика не совпала: {index + 1} {key}")
        raw = [{**row, "predicted_spans": without_snap(row)} for row in rows]
        result["nested38"].append(old)
        result["without_expand_fixed_thresholds"].append(compute(raw))
        counts["nested38"].append(tally(rows))
        counts["without_expand_fixed_thresholds"].append(tally(raw))
    if len(ids) != 699:
        raise ValueError("ожидались 699 пар train/val")
    report = {"status": "диагностика фиксированных сохранённых внешних прогнозов", "pairs": len(ids), "test301": False, "external_data": False, "new_training": False, "new_inference": False, "threshold_reselection": False, "caveat": "Порог выбран ранее для декодера с клаузным expand; отсутствие expand при этих порогах не заменяет новый внутренний выбор порога. Внешние фолды ранее использовались в экспериментах; оценка адаптивная.", "nested38": summarize(result["nested38"], counts["nested38"]), "without_expand_fixed_thresholds": summarize(result["without_expand_fixed_thresholds"], counts["without_expand_fixed_thresholds"])}
    target = ROOT / "artifacts/diagnosis46.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in result:
        print(name, json.dumps(report[name]["macro_means"], ensure_ascii=False), "pooled_fpr_token", report[name]["pooled_fpr_token"])


if __name__ == "__main__":
    main()
