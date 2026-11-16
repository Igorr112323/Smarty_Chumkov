import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")
MODES = ("seed0_coarse", "seed3_coarse", "seed5_coarse", "seed5_fine")


def aggregate(rows):
    result = {name: {"mean": round(statistics.mean(row[name] for row in rows), 4), "std": round(statistics.pstdev(row[name] for row in rows), 4)} for name in METRICS}
    result["folds_fpr_token_over_040"] = sum(row["fpr_token"] > 0.4 for row in rows)
    result["folds_fpr_pair_over_040"] = sum(row["fpr_pair"] > 0.4 for row in rows)
    return result


def fmt(value):
    return f"{value['mean']:.4f} ± {value['std']:.4f}".replace(".", ",")


def main():
    paths = [ROOT / f"artifacts/nested37_fold_{number}.json" for number in range(1, 6)]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.exists()]
    summary = {"status": "неполный прогон", "folds_expected": 5, "folds_available": [row["fold"] for row in rows], "folds_complete": [row["fold"] for row in rows if row.get("status") == "готово"], "adaptive_warning": "Пять внешних фолдов повторно используются после изучения результатов промта 36; новые варианты выбраны адаптивно. Это не независимая подтверждающая оценка.", "pair_features": "не добавлены: отдельной обучаемой pair-gate головки нет; её введение изменило бы архитектуру", "weights": "Вычислены sha256 временных файлов state_dict; файлы удалены и недоступны для развёртывания.", "test301": "не оценивался", "external_data": False}
    if len(rows) == 5 and all(row.get("status") == "готово" for row in rows):
        if [row["fold"] for row in rows] != list(range(1, 6)) or any(row["fold_ids_sha256"] != rows[0]["fold_ids_sha256"] or row["git_sha"] != rows[0]["git_sha"] for row in rows):
            raise ValueError("несовместимые фолды или версии кода")
        if any([seed["seed"] for seed in row["seeds"]] != [0, 1, 2, 3, 4] for row in rows):
            raise ValueError("неполный набор сидов")
        modes = {mode: aggregate([row["results"][mode] for row in rows]) for mode in MODES}
        summary.update({"status": "готово", "git_sha": rows[0]["git_sha"], "fold_ids_sha256": rows[0]["fold_ids_sha256"], "prior_metric_reproduced_folds": [row["fold"] for row in rows if row["prior_metric_reproduction"]], "modes": modes, "thresholds": [{"fold": row["fold"], "coarse": row["thresholds"]["coarse"], "fine": row["thresholds"]["fine"], "fine_extend": row["thresholds"]["fine_extend"]} for row in rows], "checkpoint_hashes": [{"fold": row["fold"], "seeds": [{"seed": seed["seed"], **seed["checkpoint"]} for seed in row["seeds"]]} for row in rows], "compute_seconds_sum": {"seed0_coarse": round(sum(row["seeds"][0]["seconds"] for row in rows), 3), "seed3_coarse": round(sum(seed["seconds"] for row in rows for seed in row["seeds"][:3]), 3), "seed5_coarse": round(sum(seed["seconds"] for row in rows for seed in row["seeds"]), 3), "seed5_fine": round(sum(row["elapsed_s"] for row in rows), 3)}, "nominal_target_by_mean": {mode: data["token_f1"]["mean"] >= 0.5 and data["fpr_token"]["mean"] <= 0.4 and data["fpr_pair"]["mean"] <= 0.4 for mode, data in modes.items()}, "interpretation": "Достижение порогов по средним не гарантирует FPR <= 0.40 на каждом внешнем фолде и не заменяет независимого подтверждения; веса не сохранены."})
    target = ROOT / "artifacts/nested37_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if summary["status"] == "готово":
        old = json.loads((ROOT / "artifacts/nested36_summary.json").read_text(encoding="utf-8"))
        labels = (("seed0_coarse", "seed 0, грубая сетка"), ("seed3_coarse", "3 сида, грубая сетка"), ("seed5_coarse", "5 сидов, грубая сетка"), ("seed5_fine", "5 сидов, мелкая сетка"))
        columns = ("token_precision", "token_recall", "token_f1", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")
        lines = ["# Промт 37: повторная вложенная CV и ансамбль", "", "Внешние фолды промта 36 использованы повторно после изучения прежних результатов. Это адаптивное исследование, а не независимое подтверждение достигнутого качества.", "", "При первичном запуске один синтетический тест мелкой сетки упал: в его фикстуре не оказалось допустимой точки. `tee` скрыл ненулевой код pytest, поэтому обучение ошибочно продолжилось. Тест исправлен без ослабления условия и добавлен `pipefail`; повторная независимая проверка тестов запускается отдельно. Это не замалчивается как успешная проверка исходного запуска.", "", "| Вариант | token P | token R | token F1 | span F1 | pair F1 | FPR_pair | FPR_token | точность типа | Время, сумма сек |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for mode, label in labels:
            data = summary["modes"][mode]
            lines.append("| " + label + " | " + " | ".join(fmt(data[key]) for key in columns) + f" | {summary['compute_seconds_sum'][mode]:.1f} |")
        lines.extend(["", "Значения — среднее ± population std по пяти фолдам. Время — сумма измеренного времени заданий, не wall-clock Actions; все пять сидов обучались заново, так как прежние веса и вероятности не сохранялись.", "", f"Ранее 3 сида: token F1 {fmt(old['seed_bagging_3']['token_f1'])}; число фолдов, воспроизведённых по округлённым метрикам и грубому порогу: {len(summary['prior_metric_reproduced_folds'])}/5. Совпадение метрик не доказывает побайтовой идентичности весов.", "", "Порог выбран на внутреннем фолде модели seed 0: максимум token F1 при FPR_pair и FPR_token ≤ 0,40; мелкая сетка ±0,05 с шагом 0,01, T_extend=T_start/2. Внешний фолд лишь оценивается. Ограничение FPR на внутреннем фолде не гарантирует его на внешнем."])
        for mode, label in labels:
            data = summary["modes"][mode]
            lines.append(f"- {label}: внешний FPR_token > 0,40 в {data['folds_fpr_token_over_040']}/5 фолдах; FPR_pair > 0,40 в {data['folds_fpr_pair_over_040']}/5 фолдах.")
        lines.extend(["", "Отдельные pair-признаки не добавлялись: имеющийся gate — порог максимума токенных вероятностей, обучаемой pair-gate головки нет; её добавление изменило бы архитектуру. Временные state_dict удалены, их SHA-256 есть в JSON, но deployable checkpoint отсутствует; конфигурация не заморожена.", "", "База 0,5138 получена с внешними негативами; новые числа без них, прямое сравнение ограничено. Test 301 не оценивался; внешние данные в новом прогоне не использовались."])
        (ROOT / "artifacts/nested37_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["status"] == "готово" else 2


if __name__ == "__main__":
    raise SystemExit(main())
