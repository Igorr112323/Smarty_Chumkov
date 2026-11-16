import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def number(value):
    return f"{value:.4f}"


def spread(values):
    return f"{min(values)} / {statistics.median(values):g} / {max(values)}"


def main():
    summary = json.loads((ROOT / "artifacts/summary44.json").read_text(encoding="utf-8"))
    if summary["status"] != "готово" or summary["outer_evaluations"] != 1:
        raise ValueError("нет завершённой внешней оценки")
    original = (ROOT / "artifacts/final_summary.md").read_text(encoding="utf-8")
    definitions = original.split("## Определения метрик для согласования\n", 1)[1].split("## Было / стало\n", 1)[0]
    m = summary["summary"]
    def metric(name):
        return number(m[name]["mean"])
    def maximum(name):
        return number(m[name]["max"])
    choices = [json.loads((ROOT / f"artifacts/selection44_fold{fold}.json").read_text(encoding="utf-8")) for fold in range(1, 6)]
    neutral = sum(row["fpr_neutral"] for choice in choices for row in choice["compensation"]["inner"])
    selections = " / ".join(f"{number(choice['base']['start'])}+{number(choice['compensation']['delta'])}" for choice in choices)
    status = "условие выполнено на наблюдаемых внешних фолдах" if summary["target_met_on_outer_folds"] else "условие не выполнено на наблюдаемых внешних фолдах"
    document = f"""# Итоговая справка: финальный раунд 44

**{status}.** Полученная оценка адаптивная: эти внешние фолды ранее использовались для выбора конфигураций; независимого подтверждения нет, test 301 повторно не оценивался.

## Определения метрик для согласования
{definitions}
## Было / стало

| Показатель | Было: nested38, 5 сидов, 3 эпохи | Стало: раунд 44 |
|---|---:|---:|
| Token P / R / F1 | 0,4517 / 0,4529 / 0,4506 | {metric('token_precision')} / {metric('token_recall')} / **{metric('token_f1')}** |
| Span exact / any-overlap / IoU≥0,5 F1 | 0,1398 / 0,3715 / 0,2229 | {metric('exact_f1')} / {metric('overlap_f1')} / {metric('iou_f1')} |
| Pair F1 | 0,2829 | {metric('pair_f1')} |
| FPR_pair средний / максимум | 0,1276 / 0,1957 | {metric('fpr_pair')} / {maximum('fpr_pair')} |
| FPR_token средний / максимум | 0,2810 / 0,4036 | {metric('fpr_token')} / **{maximum('fpr_token')}** |
| Точность типа | 0,9359 | {metric('type_accuracy')} |

Пять внешних фолдов; на каждый четыре внутренних и три обучающих. Сидов в каждом ансамбле: **{len(summary['seeds_used'])}** ({', '.join(map(str, summary['seeds_used']))}); сокращение из-за бюджета: **{'да' if summary['seed_fallback'] else 'нет'}**. Эпох фактически выполнено на модель, min / медиана / max: **{spread(summary['epochs_ran'])}**; лучшие эпохи EMA: **{spread(summary['best_epochs'])}**. Сумма времени 20 обучающих заданий: **{summary['sum_job_seconds']:.1f} с**; длительность самого долгого задания: **{summary['max_job_seconds']:.1f} с**. Это не общее настенное время запуска.

Сетка исходного порога start 0,10–0,95 с шагом 0,01; выбран максимум среднего внутреннего token F1 при FPR_pair ≤0,38 и FPR_token ≤0,38 **в каждом** из четырёх внутренних фолдов. Затем применён B3 и start повышен на d из 0 / 0,02 / 0,04 / 0,06; extend не менялся. Из d с обоими FPR ≤0,38 на каждом внутреннем выбран максимум среднего exact F1. Базовый start + d по внешним фолдам: **{selections}**. Отбор сделан по внутренним файлам; внешние вероятности вычислялись без внешних gold, их оценка сделана однократно после пяти выборов. Проверка фактической FPR-нейтральности относительно базового декодера: {neutral} из 20 внутренних оценок не увеличили число FP-токенов; компенсация сама по себе не гарантирует нейтральности.

FPR ограничивался на внутренних фолдах, что не гарантирует ограничение на внешних. Подробные сетки и поддержки: `artifacts/selection44_fold*.json`; пофолдовые результаты и истории: `artifacts/summary44.json`, `artifacts/train44_fold*_inner*.json`. Пообъектные данные: `inner_predictions_fold*_inner*_seed*.jsonl`, `outer_predictions_fold*_seed*.jsonl`. Все результаты вычислены после gate и снапа, тип спанов и границы сохранены.

## Ограничения и происхождение

При **старой фиксированной детекции** 142 предсказания на 300 gold, не более 83 TP при сохранении числа предсказаний в каждой паре, теоретический потолок exact F1 0,3760 по среднему фолдов; 183 gold не пересекались ни с одним прогнозом. Это ограничение не относится к новой модели, которая может предсказать другое число спанов. В раунде 44 одновременно изменены обучение, калибровка и постобработка, поэтому отдельно установить причину изменения качества нельзя. Результаты прежнего B3 были адаптивно выбраны на внешних фолдах; нельзя говорить, что внешние данные никогда не использовались для выбора конфигураций.

Test 301 ранее израсходован на другой конфигурации и повторно не оценивался; с CV его не сравнивают. База **0,5138** получена с 2994 внешними RuFacts-негативами и не является парным сравнением с этой CV. Extra 200 и внешние данные в раунде 44 не подключались. Исходная схема, encoder и gate не изменялись; три эпохи старого nested38 уже включали warmup 10%. **Новые чекпоинты не сохранялись:** сохранены предсказания и отчёты, но развернуть или повторно выполнить инференс нового ансамбля по этим артефактам нельзя. Старые frozen-веса не переписывались.
"""
    (ROOT / "artifacts/final_summary.md").write_text(document, encoding="utf-8")
    status_path = ROOT / "STATUS.md"
    text = status_path.read_text(encoding="utf-8")
    heading = "## Промпт 44: финальный раунд\n\n"
    paragraph = f"{status}. Token F1 {metric('token_f1')}, exact {metric('exact_f1')}, overlap {metric('overlap_f1')}, максимальный FPR_pair {maximum('fpr_pair')}, FPR_token {maximum('fpr_token')}; сидов {len(summary['seeds_used'])}. Отбор порогов только по четырём внутренним фолдам, внешние данные использованы однократно для оценки после выбора. Эти фолды ранее участвовали в экспериментах, независимого подтверждения нет. Полный протокол и предсказания — `artifacts/final_summary.md`, `artifacts/summary44.json`, `artifacts/selection44_fold*.json`. Test 301 и extra не использовались.\n\n"
    marker = "Дата обновления: 2026-11-14.\n\n"
    if marker not in text:
        raise ValueError("не найдена дата STATUS")
    status_path.write_text(text.replace(marker, marker + heading + paragraph, 1), encoding="utf-8")
    print(document, flush=True)


if __name__ == "__main__":
    main()
