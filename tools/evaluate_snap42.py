import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, match_spans, metrics, token_positions
from spanver.sentence_snap import snap_sentences

MODES = ("baseline", "B1", "B2", "B3", "B4")


def compute(rows):
    pairs = [{"pair_id": row["pair_id"], "model_output": row["answer"], "spans": row["gold_spans"], "predicted_spans": row["predicted_spans"]} for row in rows]
    result = metrics(pairs, lambda pair: pair["predicted_spans"])
    exact = overlap = pred_count = gold_count = length = negative_tokens = flagged_tokens = pairwise_tp_upper = 0
    gold_with_any_overlap = 0
    for pair in pairs:
        gold = pair["spans"]
        predicted = pair["predicted_spans"]
        exact += match_spans(predicted, gold, 1.0)[0]
        overlap += match_spans(predicted, gold, 1e-15)[0]
        pred_count += len(predicted)
        gold_count += len(gold)
        pairwise_tp_upper += min(len(predicted), len(gold))
        length += sum(span["end"] - span["start"] + 1 for span in predicted)
        gold_with_any_overlap += sum(any(max(span["start"], candidate["start"]) <= min(span["end"], candidate["end"]) for candidate in predicted) for span in gold)
        positions = token_positions(pair["model_output"])
        g = covered_tokens(pair["model_output"], [(span["start"], span["end"]) for span in gold])
        p = covered_tokens(pair["model_output"], [(span["start"], span["end"]) for span in predicted])
        negative_tokens += len(positions) - len(g)
        flagged_tokens += len(p - g)
    total = pred_count + gold_count
    return {"pairs": len(rows), "exact_tp": exact, "overlap_tp": overlap, "exact_f1": 2 * exact / total if total else 0.0, "overlap_f1": 2 * overlap / total if total else 0.0, "iou_f1": result["span_level"]["f1"], "token_precision": result["token_level"]["precision"], "token_recall": result["token_level"]["recall"], "token_f1": result["token_level"]["f1"], "pair_f1": result["pair_level"]["f1"], "fpr_pair": result["fpr"], "fpr_token": round(flagged_tokens / negative_tokens, 4) if negative_tokens else 0.0, "type_accuracy": result["type_accuracy"], "predicted_spans": pred_count, "gold_spans": gold_count, "pairwise_tp_upper": pairwise_tp_upper, "predicted_chars": length, "mean_predicted_length": length / pred_count if pred_count else 0.0, "gold_with_any_overlap": gold_with_any_overlap, "gold_without_any_overlap": gold_count - gold_with_any_overlap}


def transform(row, mode):
    predicted = row["predicted_spans"] if mode == "baseline" else snap_sentences(row["predicted_spans"], row["answer"], mode)
    return {"pair_id": row["pair_id"], "doc_ids": row["doc_ids"], "fold": row["fold"], "answer": row["answer"], "gold_spans": row["gold_spans"], "predicted_spans": predicted, "mode": mode, "source": "nested38_baseline_after_gate"}


def main():
    manifest = json.loads((ROOT / "artifacts/predictions40_manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "artifacts/nested38_summary.json").read_text(encoding="utf-8"))
    if manifest["status"] != "baseline nested38 recovered, not round40" or len(manifest["outputs"]) != 5:
        raise ValueError("ожидались пять фолдов старой baseline")
    results = {mode: [] for mode in MODES}
    hashes = {}
    seen = set()
    for fold, item in enumerate(manifest["outputs"], 1):
        raw = (ROOT / item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("SHA предсказаний не совпал")
        rows = [json.loads(line) for line in raw.splitlines() if line]
        if len(rows) != item["pairs"] or any(row["source"] != "nested38_baseline" or row["fold"] != fold for row in rows):
            raise ValueError("неверное происхождение фолда")
        for row in rows:
            if row["pair_id"] in seen:
                raise ValueError("повтор пары")
            seen.add(row["pair_id"])
        for mode in MODES:
            changed = [transform(row, mode) for row in rows]
            if mode != "baseline":
                target = ROOT / f"artifacts/snap42_fold{fold}_mode{mode}.jsonl"
                target.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in changed), encoding="utf-8")
                hashes[str(target.relative_to(ROOT))] = hashlib.sha256(target.read_bytes()).hexdigest()
                changed = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]
            fold_result = compute(changed)
            if mode == "baseline":
                expected = baseline["folds"][fold - 1]["metrics"]
                for name in ("token_precision", "token_recall", "token_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy", "predicted_spans"):
                    if fold_result[name] != expected[name]:
                        raise ValueError(f"baseline не совпала: fold {fold}, {name}")
                if fold_result["iou_f1"] != expected["span_f1"]:
                    raise ValueError("baseline IoU не совпала")
            results[mode].append(fold_result)
    if len(seen) != 699 or sum(row["gold_spans"] for row in results["baseline"]) != 300:
        raise ValueError("неверная поддержка")
    summaries = {}
    for mode, folds in results.items():
        predicted = sum(row["predicted_spans"] for row in folds)
        summaries[mode] = {"exact_f1": statistics.mean(row["exact_f1"] for row in folds), "overlap_f1": statistics.mean(row["overlap_f1"] for row in folds), "iou_f1": statistics.mean(row["iou_f1"] for row in folds), "predicted_spans": predicted, "gold_spans": sum(row["gold_spans"] for row in folds), "mean_predicted_length": sum(row["predicted_chars"] for row in folds) / predicted if predicted else 0.0, "token_f1": statistics.mean(row["token_f1"] for row in folds), "pair_f1": statistics.mean(row["pair_f1"] for row in folds), "fpr_pair_mean": statistics.mean(row["fpr_pair"] for row in folds), "fpr_pair_max": max(row["fpr_pair"] for row in folds), "fpr_token_mean": statistics.mean(row["fpr_token"] for row in folds), "fpr_token_max": max(row["fpr_token"] for row in folds), "type_accuracy": statistics.mean(row["type_accuracy"] for row in folds), "exact_tp": sum(row["exact_tp"] for row in folds), "overlap_tp": sum(row["overlap_tp"] for row in folds), "gold_with_any_overlap": sum(row["gold_with_any_overlap"] for row in folds), "gold_without_any_overlap": sum(row["gold_without_any_overlap"] for row in folds), "folds": folds}
    original = summaries["baseline"]
    limit_macro = statistics.mean(2 * row["pairwise_tp_upper"] / (row["predicted_spans"] + row["gold_spans"]) for row in results["baseline"])
    upper_tp = sum(row["pairwise_tp_upper"] for row in results["baseline"])
    output = {"status": "готово; после сохранённой baseline, без обучения и инференса", "warning": "Варианты сравниваются на внешних фолдах, уже использованных в прошлых экспериментах; выбор лучшего по ним адаптивен, не независимая оценка.", "protocol": "Спаны после существующих gate и клаузного expand. Границы предложений: SENTENCE_BREAK; крайние пробелы удалены. B1/B2: пересечение со 30%/15% длины предложения включительно. B3: выбранные B1 предложения объединяются со всей цепочкой смежных предложений, если каждый разделитель короче 20 символов; промежуточные предложения могут не иметь предсказания. B4: для каждого выбранного B1 предложения добавить ровно по одному соседу слева и справа, без рекурсии. Пересекающиеся итоговые спаны объединяются. Сопоставление exact/overlap: 1:1 в порядке предсказаний и gold, тип игнорируется; длина и границы включительные. F1 — среднее пяти фолдов; длина — общий средний по всем предсказанным спанам.", "results": summaries, "output_sha256": hashes, "fixed_count_upper_bound": {"note": "теоретический верх при сохранении числа предсказаний в каждой паре и сопоставлении 1:1; не гарантия достижимости снапом", "max_tp": upper_tp, "fold_mean": limit_macro, "pooled": 2 * upper_tp / (original["predicted_spans"] + original["gold_spans"]), "min_unmatched_gold_if_perfect": original["gold_spans"] - upper_tp, "gold_without_any_overlap_now": original["gold_without_any_overlap"]}}
    target = ROOT / "artifacts/snap42_summary.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for mode, row in summaries.items():
        print(mode, ' '.join(f'{name}={row[name]:.4f}' for name in ("exact_f1", "overlap_f1", "iou_f1", "mean_predicted_length", "token_f1", "fpr_token_max")), "pred", row["predicted_spans"], flush=True)
    print("fixed_count_upper_bound", json.dumps(output["fixed_count_upper_bound"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
