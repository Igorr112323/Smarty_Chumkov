import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def triple(value):
    return "/".join(f"{value[key]:.4f}" for key in ("precision", "recall", "f1"))


def metric_row(left, right, hysteresis):
    start = f"{left['start']:.2f}"
    threshold = f"{start}/{left['extend']:.3f}" if hysteresis else start
    def values(row):
        return [triple(row["token"]), triple(row["span"]), f"{row['pair']['f1']:.4f}", f"{row['fpr_pair']:.4f}/{row['fpr_token']:.4f}"]
    return "| " + " | ".join([threshold] + values(left) + values(right)) + " |"


def main():
    reports = {name: json.loads((ROOT / f"artifacts/threshold30_{name}.json").read_text(encoding="utf-8")) for name in ("bce", "focal")}
    verification = json.loads((ROOT / "artifacts/threshold30_verification.json").read_text(encoding="utf-8"))
    timings = json.loads((ROOT / "artifacts/threshold30_timings.json").read_text(encoding="utf-8"))
    if any(report["status"] != "готово" for report in reports.values()) or not all(check["matches"] for check in verification["comparisons"]):
        raise ValueError("отчёты или проверка не завершены")
    lines = ["# Промт 30: пороги на отложенных RAGTruth-ru", "", "Обе модели оценены на одних 1458 отложенных парах. Инференс: полные fp32 состояния, top-4 чанка документа, token threshold, snap expand, минимальная длина спана 2 знака. При переборе pair gate отключён; train и test 301 не использовались. Обучения не было.", "", "Порог выбран и измерен на одном и том же holdout: лучшие значения оптимистичны и не являются независимой итоговой оценкой. Они не являются оценкой по нашим 699 парам.", "", "## Квантили вероятности субтокена", "", "| квантиль | BCE gold + | BCE gold − | focal gold + | focal gold − |", "|---|---:|---:|---:|---:|"]
    for quantile in ("1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "100%"):
        values = [reports[model]["histogram"][label]["quantiles"][quantile] for model, label in (("bce", "positive"), ("bce", "negative"), ("focal", "positive"), ("focal", "negative"))]
        lines.append("| " + quantile + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", f"Субтокенов: 20 075 положительных и 225 188 отрицательных; ROC AUC BCE {reports['bce']['histogram']['subword_roc_auc']:.4f}, focal {reports['focal']['histogram']['subword_roc_auc']:.4f}. Субтокен положителен при пересечении с gold-спаном; для пересечений чанков используется максимум. Распределения сдвинуты, но существенно перекрываются.", "", "Метрики token и FPR_token рассчитаны по словам, отделённым пробелами, после snap, а не по субтокенам квантилей. Span P/R/F1 — совпадение по IoU ≥ 0,5. Pair F1 повторяет evaluator: положительная пара считается верной при наличии совпавшего спана, а не только любого предсказания. FPR_pair вычислен на 287 парах без gold-спанов; FPR_token — на всех отрицательных словах.", "", "## Лучшие точки", "", "| модель и режим | T_start / T_extend | token F1 | span P/R/F1 | pair F1 | FPR_pair | FPR_token |", "|---|---|---:|---|---:|---:|---:|"]
    for model in ("bce", "focal"):
        for mode in ("simple", "hysteresis"):
            for category, suffix in (("без ограничения", "best_unrestricted"), ("оба FPR ≤ 0,40", "best_fpr_limited")):
                row = reports[model][f"{mode}_{suffix}"]
                if row is None:
                    lines.append(f"| {model}, {mode}, {category} | нет допустимого порога | — | — | — | — | — |")
                else:
                    lines.append(f"| {model}, {mode}, {category} | {row['start']:.2f} / {row['extend']:.3f} | {row['token']['f1']:.4f} | {triple(row['span'])} | {row['pair']['f1']:.4f} | {row['fpr_pair']:.4f} | {row['fpr_token']:.4f} |")
    for mode, title in (("simple", "Один порог"), ("hysteresis", "Гистерезис: T_extend = T_start / 2")):
        lines += ["", f"## {title}", "", "В ячейках токена и спана: precision / recall / F1. В последней ячейке каждой модели: FPR_pair / FPR_token.", "", "| порог | BCE token P/R/F1 | BCE span P/R/F1 | BCE pair F1 | BCE FPR_pair/token | focal token P/R/F1 | focal span P/R/F1 | focal pair F1 | focal FPR_pair/token |", "|---|---|---|---:|---|---|---|---:|---|"]
        left, right = reports["bce"][mode], reports["focal"][mode]
        if len(left) != 19 or len(right) != 19:
            raise ValueError("неполная сетка")
        for lrow, rrow in zip(left, right):
            if lrow["start"] != rrow["start"]:
                raise ValueError("сетки не совпадают")
            lines.append(metric_row(lrow, rrow, mode == "hysteresis"))
    lines += ["", "Лучший span F1 ниже 0,10 во всех режимах и для обоих чекпоинтов, диапазон 0,25–0,40 не достигнут. Прирост гистерезиса относительно лучшего одиночного порога: BCE +0,0172 без ограничения и +0,0015 при ограничениях; focal +0,0144 и +0,0001 соответственно (разности точных значений, округлены). У focal одиночные пороги 0,55 и 0,60 имеют одинаковый F1 при округлении до четырёх знаков, но точный F1 выше при 0,60. Лучшие точки выбираются по неокруглённому F1, ограничения — по точным числителям и знаменателям.", "", "## Прежняя оценка и проверка", "", "В промте 29 использовались token threshold 0,25, отдельный pair gate 0,45, snap expand и без гистерезиса (`tools/train_scheme13_a1.py`). С текущим кодом точка для BCE воспроизведена: token F1 0,1558, span F1 0,0276, FPR_pair 1,0000, FPR_token 0,9824. Порог 0,25 использовался в прежней CV на наших данных; для RAGTruth этот декодер задавался фиксированно, а не подбирался. Ранее гистерезис подбирали на наших train/val 699 парах и CV-чекпоинтах, отличных от этих двух (`tools/sweep_hyst.py`); он не применялся в оценке промта 29.", "", "FPR_pair 1,0000 означает, что помечены все 287 негативных пар; само по себе это не доказывает, что все вероятности или все токены выше порога.", "", f"Быстрый расчёт сверён с исходной реализацией на {verification['pairs']} случайных парах, {len(verification['comparisons'])} сочетаний: все совпали. Исходные прогоны BCE и focal с инференсом: {timings['bce_initial_seconds']:.3f} и {timings['focal_initial_seconds']:.3f} с; повторный расчёт по кэшу: {reports['bce']['seconds']:.3f} и {reports['focal']['seconds']:.3f} с.", "", "Подбор по тому же holdout не даёт честной оценки переносимости выбранного порога. Дополнительные эпохи, наша выборка для обучения и test 301 не запускались."]
    target = ROOT / "artifacts/threshold30_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
