import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def load(name):
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def number(value):
    return f"{value:.4f}".replace(".", ",")


def row(label, summary, epochs, seconds):
    names = ("token_precision", "token_recall", "token_f1", "fpr_token", "fpr_pair", "exact_f1", "overlap_f1", "pair_f1")
    data = [number(summary[name]["mean"]) for name in names]
    data.insert(4, number(summary["fpr_token"]["max"]))
    data.insert(6, number(summary["fpr_pair"]["max"]))
    return "| " + " | ".join([label] + data + [epochs, seconds]) + " |"


def main():
    original = load("nested38_summary.json")
    reproduction = load("final45_reproduction_summary.json")
    baseline_spans = load("predictions40_baseline_analysis.json")
    info = load("ablation46_run_info.json") or {}
    statuses = {}
    for variant in ("control", "v1", "v2", "v3"):
        value = load(f"ablation46_{variant}_summary.json")
        if value and value.get("status") == "готово":
            statuses[variant] = value
    original_summary = {name: {"mean": original["summary"][name]["mean"], "max": original["summary"][name]["mean"]} for name in ("token_precision", "token_recall", "token_f1", "fpr_token", "fpr_pair", "pair_f1")}
    original_summary["fpr_token"]["max"] = original["max_outer_fpr_token"]
    original_summary["fpr_pair"]["max"] = original["max_outer_fpr_pair"]
    original_summary["exact_f1"] = {"mean": baseline_spans["mean_exact_f1"], "max": 0}
    original_summary["overlap_f1"] = {"mean": baseline_spans["mean_any_overlap_f1"], "max": 0}
    header = "| Конфигурация | Token P | Token R | Token F1 | FPR_token средний | FPR_token максимум | FPR_pair средний | FPR_pair максимум | Span exact | Span any-overlap | Pair F1 | Эпохи min / median / max | Время Actions |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    table = [row("Исторический контроль, 5 сидов", original_summary, "3 / 3 / 3", "см. nested38")]
    labels = {"control": "Контроль, 2 сида", "v1": "В1: 15 эпох + early stopping", "v2": "В2: В1 + freeze/LR", "v3": "В3: В2 + EMA"}
    for variant, label in labels.items():
        value = statuses.get(variant)
        if value:
            epochs = value["epochs_ran"]
            run = info.get(variant, {})
            seconds = f"{run['wall_seconds']} с" if "wall_seconds" in run else "не зафиксировано"
            table.append(row(label, value["summary"], f"{min(epochs)} / {statistics.median(epochs):g} / {max(epochs)}", seconds))
        else:
            table.append("| " + label + " | " + " | ".join(["—"] * 12) + " |")
    reproduced = bool(reproduction and reproduction.get("all_metrics_match_original") and reproduction.get("macro_token_f1") == 0.4506)
    lines = ["# Абляция 46: контроль и поэтапные изменения", "", f"Пятисидовый исторический контроль 0,4506 воспроизведён: **{'да' if reproduced else 'не подтверждено'}** (`artifacts/final45_reproduction_summary.json`). Это не результат двухсидового контроля.", "", "Клаузный `expand` оставлен по согласованию, предложенческий снап B1–B4 и его компенсация отключены. Три эпохи для контроля; В1 — до 15 эпох, patience 3 по внутренней ошибке и восстановление лучшей эпохи; В2 — дополнительно заморозка эмбеддингов в первые 5 эпох и множитель их LR 0,1; В3 — дополнительно EMA 0,999. Планировщик learning rate рассчитан на 15 эпох с warmup 10%, поэтому изменение длительности также растягивает его расписание относительно трёхэпохового контроля. Энкодер и остальные параметры схемы не менялись. Выбор порога: четыре внутренних фолда, оба FPR ≤0,40 в каждом; внешний фолд не участвует в подборе порога.", "", "Бюджет: два сида 0 и 1 для всех новых конфигураций, включая отдельный контроль, чтобы не смешивать изменение числа сидов с абляцией. Оценка по прошлому запуску: три последовательных пятисидовых раунда по 1 ч 50 мин 22 с уже превышают 180 минут, без учёта контроля и доставки весов. Поэтому 5-сидовая строка служит проверкой воспроизводимости, но разности В1/В2/В3 считаются относительно двухсидовых предшественников.", "", "Метрики — средние пяти внешних фолдов, кроме столбцов максимум. Exact и any-overlap — 1:1 сопоставление спанов; FPR_pair и FPR_token показываются отдельно. Время Actions — настенная длительность запуска варианта, включая тесты и доставку, а не сумма параллельных заданий.", "", header, *table, "", "## Разности", ""]
    for name, previous in (("v1", "control"), ("v2", "v1"), ("v3", "v2")):
        if name in statuses and previous in statuses:
            delta = statuses[name]["summary"]["token_f1"]["mean"] - statuses[previous]["summary"]["token_f1"]["mean"]
            change = "плюс" if delta > 0 else "минус" if delta < 0 else "без изменения"
            lines.append(f"- {name.upper()} относительно {previous.upper()}: {change}, Δ token F1 {delta:+.4f}.")
        else:
            lines.append(f"- {name.upper()}: нет завершённой пары результатов для сравнения.")
    for name in ("v2", "v3"):
        value = statuses.get(name)
        if value:
            primary = [load(f"ablation46_{name}_outer{fold}_inner{fold % 5 + 1}.json") for fold in range(1, 6)]
            if all(primary):
                seeds = [seed for item in primary for seed in item["seeds"]]
                stopped = sum(epoch <= 5 for epoch in value["epochs_ran"])
                best_frozen = sum(seed["best_epoch"] <= 5 for seed in seeds)
                lines.append(f"- {name.upper()}: {stopped} из {len(value['epochs_ran'])} обучений остановились не позже эпохи 5; у {best_frozen} из {len(seeds)} основных checkpoint лучшая эпоха пришлась на заморозку. Пониженный LR после разморозки не повлиял на такие checkpoint.")
    lines += ["", "## Объединённые out-of-fold показатели на 699 парах", ""]
    for name, label in labels.items():
        pooled = load(f"ablation46_{name}_pooled.json")
        if pooled:
            counts = pooled["counts"]
            lines.append(f"- {label}: token F1 {number(pooled['pooled_token_f1'])}; FPR_token {counts['false_positive_tokens']}/{counts['negative_tokens']} = {number(pooled['pooled_fpr_token'])}; FPR_pair {counts['false_positive_pairs']}/{counts['negative_pairs']} = {number(pooled['pooled_fpr_pair'])}.")
    lines += ["", "## Ограничения", "", "Эти внешние фолды ранее участвовали в экспериментах, а решение не запускать В3 при F1 ≥0,50 использует наблюдаемый внешний результат: оценка адаптивная, независимого подтверждения нет. Пятикратное CV на 699 парах — не единая отложенная выборка 1200 пар. Поэтому ни средний FPR, ни F1 этой CV сами по себе не подтверждают выполнение пункта договора. Максимальный FPR приведён как диагностика, а не как дополнительно утверждённый пункт договора.", "", "Веса основных моделей каждого завершённого варианта: `weights/ablation46/<вариант>/fold<N>/`, два seed на фолд. Каждая модель отдельно восстановлена из частей и её вероятности проверены; веса внутренних моделей для выбора порога не сохраняются. Использование — `weights/ablation46/README.md`. Test 301, extra и внешние данные не использовались. Сравнение результатов не доказывает глобальной границы достижимого качества или причины различий за пределами этих конфигураций."]
    target = ARTIFACTS / "ablation46_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
