import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.decode44 import predict44
from spanver.evaluate import read_jsonl, token_positions
from spanver.sentence_snap import sentence_intervals
from tools.evaluate_snap42 import compute as score_rows
from tools.fpr_bound43 import counts as fpr_counts
from tools.preflight_13 import components, id_digest, our_keys, split_groups
from tools.prepare_ours33 import EXPECTED_IDS
from tools.select44 import base_curve, blend, compensation_curve, prediction_rows

EXPECTED_GOLD_SHA = "0ba55b91943c149a61ffe9db35a91db2a32deda573d72ba8fb6575f51b027f63"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def words(answer, pieces):
    result = []
    for start, end in token_positions(answer):
        scores = [piece[2] for piece in pieces if piece[0] <= end and piece[1] > start]
        result.append({"start": start, "end": end, "probability": max(scores) if scores else None})
    return result


def load_reports(folds):
    reports = {}
    for outer in range(1, 6):
        for inner in range(1, 6):
            if outer == inner:
                continue
            path = ROOT / f"artifacts/train44_fold{outer}_inner{inner}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["status"] != "готово" or report["outer_fold"] != outer or report["inner_fold"] != inner or report["gold_sha256"] != EXPECTED_GOLD_SHA or report["fold_ids_sha256"] != [id_digest(fold) for fold in folds] or report["inner_id_sha256"] != id_digest(folds[inner - 1]) or report["outer_id_sha256"] != id_digest(folds[outer - 1]):
                raise ValueError("неполный прогон или несовпадение разбиения")
            if [seed["seed"] for seed in report["seeds"]] != list(range(len(report["seeds"]))) or len(report["seeds"]) < 2:
                raise ValueError("нужны хотя бы два последовательных сида")
            for info in report["seeds"]:
                if digest(ROOT / info["inner_file"]) != info["inner_initial_sha256"]:
                    raise ValueError("SHA внутреннего файла не совпал")
            reports[outer, inner] = report
    if len({row["git_sha"] for row in reports.values()}) != 1:
        raise ValueError("различаются версии кода заданий")
    return reports


def select_all(folds, reports, seed_count):
    selections = {}
    for outer in range(1, 6):
        inner_sets = []
        inner_numbers = [inner for inner in range(1, 6) if inner != outer]
        for inner in inner_numbers:
            seed_rows = [read_jsonl(ROOT / f"artifacts/inner_predictions_fold{outer}_inner{inner}_seed{seed}.jsonl") for seed in range(seed_count)]
            if any(id_digest(rows) != id_digest(folds[inner - 1]) or len(rows) != len(folds[inner - 1]) for rows in seed_rows):
                raise ValueError("неверные ID внутренних прогнозов")
            truth = {row["pair_id"]: row for row in folds[inner - 1]}
            for seed, rows in enumerate(seed_rows):
                if any(row["fold"] != outer or row["inner_fold"] != inner or row["seed"] != seed or row["answer"] != truth[row["pair_id"]]["model_output"] or row["gold_spans"] != truth[row["pair_id"]]["spans"] or row["doc_ids"] != truth[row["pair_id"]]["doc_ids"] for row in rows):
                    raise ValueError("неверные внутренние данные")
            inner_sets.append(blend(seed_rows))
        base, base_grid = base_curve(inner_sets)
        selection_path = ROOT / f"artifacts/selection44_fold{outer}.json"
        if base is None:
            save_json(selection_path, {"status": "STOP: нет допустимого базового порога", "inner_folds": inner_numbers, "base_grid": base_grid})
            raise ValueError("нет базового порога с обоими внутренними FPR <= 0,38")
        for index, (pairs, cache) in enumerate(inner_sets):
            check = score_rows(prediction_rows(pairs, cache, base["start"], base["extend"], False))
            reference = base["inner"][index]
            if check["token_f1"] != reference["token"]["f1"] or check["iou_f1"] != reference["span"]["f1"] or check["pair_f1"] != reference["pair"]["f1"] or check["fpr_pair"] != reference["fpr_pair"] or check["fpr_token"] != reference["fpr_token"]:
                raise ValueError("быстрый и точный evaluator не совпали")
        compensation, curve = compensation_curve(inner_sets, base["start"], base["extend"])
        selection = {"status": "готово" if compensation else "STOP: нет допустимой компенсации снапа", "outer_fold": outer, "seed_count": seed_count, "inner_folds": inner_numbers, "base": base, "base_grid": base_grid, "compensation": compensation, "compensation_grid": curve, "rule": "только четыре внутренних фолда: базовый порог по среднему token F1 при обоих FPR <= 0.38 в каждом; затем d по среднему exact F1 при обоих FPR <= 0.38 в каждом; extend фиксирован; нейтральность относительно базовых FP показана отдельно"}
        save_json(selection_path, selection)
        if compensation is None:
            raise ValueError("нет допустимого d снапа на внутренних фолдах")
        selections[outer] = selection
    return selections


def finalize_inner(reports, selections, seed_count):
    for outer in range(1, 6):
        choice = selections[outer]["compensation"]
        for inner in range(1, 6):
            if inner == outer:
                continue
            for seed in range(seed_count):
                path = ROOT / f"artifacts/inner_predictions_fold{outer}_inner{inner}_seed{seed}.jsonl"
                rows = read_jsonl(path)
                for row in rows:
                    value = {"answer": row["answer"], "pieces": row["pieces"]}
                    row["predicted_spans"] = predict44(value, choice["start"], choice["extend"], True)
                    row["prediction_status"] = "selected_on_inner_ensemble"
                    row["start_threshold"] = choice["start"]
                    row["extend_threshold"] = choice["extend"]
                    row["snap_delta"] = choice["delta"]
                save_rows(path, rows)
                reports[outer, inner]["seeds"][seed]["inner_selected_sha256"] = digest(path)
            save_json(ROOT / f"artifacts/train44_fold{outer}_inner{inner}.json", reports[outer, inner])


def outer_evaluation(folds, selections, seed_count):
    outcomes = []
    for outer in range(1, 6):
        choice = selections[outer]["compensation"]
        gold = {row["pair_id"]: row for row in folds[outer - 1]}
        seed_rows = []
        for seed in range(seed_count):
            source = ROOT / f"artifacts/outer44_raw_fold{outer}_seed{seed}.jsonl"
            raw = read_jsonl(source)
            if len(raw) != len(gold) or {row["pair_id"] for row in raw} != set(gold) or any(row["fold"] != outer or row["seed"] != seed for row in raw):
                raise ValueError("неполные внешние вероятности")
            rows = []
            for item in raw:
                truth = gold[item["pair_id"]]
                if item["answer"] != truth["model_output"] or item["doc_ids"] != truth["doc_ids"]:
                    raise ValueError("несовпадение внешних ID или текста")
                value = {"answer": item["answer"], "pieces": item["pieces"]}
                row = {**item, "pair_prob": max((piece[2] for piece in item["pieces"]), default=0.0), "pair_prob_definition": "max_piece_probability_gate_score_not_separate_pair_head", "word_probabilities": words(item["answer"], item["pieces"]), "gold_spans": truth["spans"], "predicted_spans": predict44(value, choice["start"], choice["extend"], True), "sentence_intervals": sentence_intervals(item["answer"]), "start_threshold": choice["start"], "extend_threshold": choice["extend"], "snap_delta": choice["delta"], "selection": "only_inner"}
                rows.append(row)
            target = ROOT / f"artifacts/outer_predictions_fold{outer}_seed{seed}.jsonl"
            save_rows(target, rows)
            seed_rows.append(read_jsonl(target))
        pairs, cache = blend(seed_rows)
        ensemble = prediction_rows(pairs, cache, choice["start"], choice["extend"], True)
        metric = score_rows(ensemble)
        exact_fpr = fpr_counts(ensemble)
        outcomes.append({"fold": outer, "start": choice["start"], "extend": choice["extend"], "delta": choice["delta"], "metrics": metric, "fpr_support": exact_fpr})
    return outcomes


def main():
    path = ROOT / "artifacts/ours33.jsonl"
    data = read_jsonl(path)
    if digest(path) != EXPECTED_GOLD_SHA or len(data) != 699 or id_digest(data) != EXPECTED_IDS:
        raise ValueError("неверный набор train/val")
    folds = split_groups(components(data, our_keys), 5, 0)
    reports = load_reports(folds)
    seed_count = 5 if all(len(row["seeds"]) == 5 for row in reports.values()) else 2
    selections = select_all(folds, reports, seed_count)
    finalize_inner(reports, selections, seed_count)
    outcomes = outer_evaluation(folds, selections, seed_count)
    metrics = [row["metrics"] for row in outcomes]
    names = ("token_precision", "token_recall", "token_f1", "exact_f1", "overlap_f1", "iou_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")
    summary = {name: {"mean": statistics.mean(row[name] for row in metrics), "std": statistics.pstdev(row[name] for row in metrics), "max": max(row[name] for row in metrics)} for name in names}
    histories = [seed for report in reports.values() for seed in report["seeds"][:seed_count]]
    output = {"status": "готово", "outer_evaluations": 1, "seeds_used": list(range(seed_count)), "seed_fallback": seed_count == 2, "fallback_rule": "если хотя бы одно из 20 заданий не завершило 5 сидов, все ансамбли используют только 0 и 1", "gold_sha256": EXPECTED_GOLD_SHA, "git_sha": next(iter(reports.values()))["git_sha"], "protocol": "5 внешних, на каждый 4 внутренних и 3 обучающих фолда; EMA loss и пороги только на внутренних; старты и d из внутреннего ансамбля; внешняя оценка после всех 5 выборов", "warning": "внешние фолды использовались в прежних экспериментах и не являются независимым test", "gate": 0.45, "inner_limit": 0.38, "max_epochs": 15, "patience": 3, "frozen_embedding_epochs": 5, "embedding_lr_multiplier": 0.1, "ema_decay": 0.999, "warmup_fraction": 0.1, "epochs_ran": [row["epochs_ran"] for row in histories], "best_epochs": [row["best_epoch"] for row in histories], "seed_seconds": [row["seconds"] for row in histories], "sum_job_seconds": sum(report["elapsed_s"] for report in reports.values()), "max_job_seconds": max(report["elapsed_s"] for report in reports.values()), "folds": outcomes, "summary": summary, "target_met_on_outer_folds": summary["token_f1"]["mean"] >= 0.50 and summary["fpr_pair"]["max"] <= 0.40 and summary["fpr_token"]["max"] <= 0.40}
    save_json(ROOT / "artifacts/summary44.json", output)
    print(json.dumps({"status": output["status"], "seeds_used": output["seeds_used"], "summary": summary, "target_met_on_outer_folds": output["target_met_on_outer_folds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
