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


def triple(value):
    return " / ".join(number(value[name]) for name in ("precision", "recall", "f1"))


def main():
    names = ("v2", "v4", "v5")
    results = {name: load(f"{'ablation46' if name == 'v2' else 'ablation47'}_{name}_summary.json") for name in names}
    pooled = {name: load(f"{'ablation46' if name == 'v2' else 'ablation47'}_{name}_pooled.json") for name in names}
    levels = {name: load(f"ablation47_{name}_levels.json") for name in names}
    runs = load("ablation47_run_info.json") or {}
    available = [name for name in names if results[name] and results[name]["status"] == "готово" and pooled[name] and levels[name]]
    if "v2" not in available:
        raise ValueError("нет рассчитанных метрик исходного В2")
    best = max(available, key=lambda name: results[name]["summary"]["token_f1"]["mean"])
    best_f1 = results[best]["summary"]["token_f1"]["mean"]
    best_pooled = pooled[best]
    lines = ["# Промпт 47: В2 и пять сидов", "", "Результаты — адаптивная вложенная CV на 699 train/val парах, уже использованных в экспериментах; это не независимая отложенная часть 1200 пар. Пороги выбирались только по четырём внутренним фолдам, оба FPR ≤0,40 на каждом; по внешним меткам пороги не выбирались. Сохранён клаузный `expand`, предложенческий снап, компенсация и EMA отключены. Замена двух сидов на пять не гарантирует прирост +0,0238: это наблюдение для другого контроля, а не причинная оценка В2.", "", "## Было / стало: три уровня", "", "Macro — среднее по пяти внешним фолдам; span P/R/F1 соответствует IoU≥0,5. Сводные оценки объединяют out-of-fold предсказания всех 699 пар и не являются оценкой одной модели.", "", "| Показатель | В2, 2 сида | " + best.upper() + ", " + ("2 сида" if best == "v2" else "5 сидов") + " |", "|---|---:|---:|"]
    old, new = levels["v2"]["macro"], levels[best]["macro"]
    for name, label in (("token", "Token P / R / F1"), ("span", "Span IoU≥0,5 P / R / F1"), ("pair", "Pair P / R / F1")):
        lines.append(f"| {label} | {triple(old[name])} | {triple(new[name])} |")
    for metric, label in (("exact_f1", "Span exact F1"), ("overlap_f1", "Span any-overlap F1"), ("fpr_pair", "FPR_pair средний / максимум"), ("fpr_token", "FPR_token средний / максимум")):
        left, right = results["v2"]["summary"][metric], results[best]["summary"][metric]
        a = number(left["mean"]); b = number(right["mean"])
        if metric.startswith("fpr_"):
            a += " / " + number(left["max"]); b += " / " + number(right["max"])
        lines.append(f"| {label} | {a} | {b} |")
    lines += [f"| Сводный token F1 | {number(pooled['v2']['pooled_token_f1'])} | {number(best_pooled['pooled_token_f1'])} |", f"| Сводный FPR_token | {number(pooled['v2']['pooled_fpr_token'])} | {number(best_pooled['pooled_fpr_token'])} |", "", "## Все измеренные варианты", "", "| Вариант | Сиды | Token F1 macro | Token F1 сводный | FPR_token сводный / средний / максимум | Эпохи min / медиана / max | Время Actions |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in available:
        summary = results[name]; fprs = summary["summary"]["fpr_token"]; epochs = summary["epochs_ran"]
        duration = runs.get(name, {}).get("wall_seconds")
        if name == "v2":
            duration = (load("ablation46_run_info.json") or {}).get("v2", {}).get("wall_seconds")
        elapsed = f"{duration // 60}:{duration % 60:02}" if duration is not None else "—"
        lines.append(f"| {name.upper()} | {len(summary['seeds'])} | {number(summary['summary']['token_f1']['mean'])} | {number(pooled[name]['pooled_token_f1'])} | {number(pooled[name]['pooled_fpr_token'])} / {number(fprs['mean'])} / {number(fprs['max'])} | {min(epochs)} / {statistics.median(epochs):g} / {max(epochs)} | {elapsed} |")
    if "v4" in available:
        difference = results["v4"]["summary"]["token_f1"]["mean"] - results["v2"]["summary"]["token_f1"]["mean"]
        signed = f"{difference:+.4f}".replace(".", ",")
        lines += ["", f"В4 относительно В2: Δ macro token F1 **{signed}**. Это наблюдаемая разница двух и пяти сидов, не гарантия переноса на другую выборку."]
        audit = results["v4"].get("paired_v2_audit", {})
        lines += ["", f"Проверка первых двух сидов В4 относительно В2: совпадение числа эпох {audit.get('epochs_match', 0)}/{audit.get('count', 0)}, лучшей эпохи {audit.get('best_epoch_match', 0)}/{audit.get('count', 0)}; максимум абсолютного расхождения train/inner loss {audit.get('max_loss_abs_diff', 0):.8g}. Версии torch и transformers совпали соответственно {audit.get('torch_match', 0)}/{audit.get('count', 0)} и {audit.get('transformers_match', 0)}/{audit.get('count', 0)}. Loss расходится, побайтовая идентичность моделей и чисто причинное объяснение разницы не доказаны."]
    lines += ["", "## F1 / FPR: наблюдаемые точки", "", "Ось X — сводный FPR_token, Y — сводный token F1 на 699 парах. Это разные конфигурации, не измеренная кривая и не доказанная граница достижимого.", "", "```text", "token F1 (сводный)"]
    points = [("A, 5 сидов", 0.28329909010859994, 0.4512)] + [(name.upper(), pooled[name]["pooled_fpr_token"], pooled[name]["pooled_token_f1"]) for name in available]
    for label, fpr, f1 in sorted(points, key=lambda item: item[2], reverse=True):
        index = max(0, min(36, round((fpr - 0.2) / 0.3 * 36)))
        lines.append(f"{number(f1)} | {' ' * index}● {label} ({number(fpr)}; {number(f1)})")
    lines += ["       +------------------------------------- FPR_token", "        0,20                    0,40       0,50", "```"]
    decision = load("ablation47_decision.json")
    if decision and "v4" in available:
        lines += ["", "## Правило остановки", "", f"В4: macro token F1 {number(decision['macro_token_f1_from_rounded_fold_metrics'])}, сводный token F1 {number(decision['pooled_token_f1'])}, сводный FPR_token {number(decision['pooled_fpr_token'])}. По macro-метрике 0,50 преодолён, поэтому В5 не запускался; по сводному F1 значение 0,50 не достигнуто даже на этих 699 парах. Это различие нельзя скрывать округлением или считать подтверждением договора."]
    lines += ["", "## Ограничения и веса", "", "Сводные F1/FPR рассчитаны по объединённым out-of-fold прогнозам; средний FPR и максимальный по фолдам отдельно показаны выше. Договорная оценка на единой отложенной по документам части 1200 пар не проводилась, выполнение договора не подтверждено. Выбор вариантов и правило остановки адаптивны по уже исследованным внешним фолдам. Test 301, extra и внешние данные не использовались; архитектура encoder не менялась.", "", "В каждом завершённом пятисидовом варианте сохранены 25 основных checkpoint — 5 сидов каждого из 5 внешних ансамблей; восстановление каждого и равенство его вероятностей проверены в Actions. Это не веса всех 100 внутренних обучений; сетки выбора, истории и предсказания сохранены в `artifacts/ablation47_*`. Инструкции по инференсу и происхождение сохранённой модели — в манифестах. Пять моделей одного фолда не подтверждают средние показатели всех пяти фолдов."]
    target = ARTIFACTS / "ablation47_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
