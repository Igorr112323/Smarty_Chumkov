import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fmt(value):
    return f"{value:.4f}".replace(".", ",")


def main():
    artifacts = ROOT / "artifacts"
    summary = json.loads((artifacts / "nested38_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((artifacts / "nested38_test_audit.json").read_text(encoding="utf-8"))
    verification = json.loads((artifacts / "frozen38_verification.json").read_text(encoding="utf-8"))
    if summary["status"] != "готово" or audit["status"] != "готово" or verification["status"] != "готово":
        raise ValueError("не все этапы завершены")
    values = summary["summary"]
    success = values["token_f1"]["mean"] >= 0.50 and summary["max_outer_fpr_token"] <= 0.40 and summary["max_outer_fpr_pair"] <= 0.40
    lines = ["# Промт 38: ограничение по худшему внутреннему фолду", "", "Порог каждого внешнего фолда выбирался только на четырёх остальных внутренних: максимум среднего token F1 при обоих FPR ≤ 0,40 в каждом внутреннем. Внешний фолд не участвовал в этом выборе. Сетка start 0,10…0,95 шаг 0,01; extend=start/2; пять сидов, три эпохи.", "", "Оценка получена на фолдах, использованных для выбора конфигурации; независимого подтверждения нет, test 301 израсходован ранее на другой конфигурации. Внутренние ограничения не гарантируют внешний FPR.", "", "| Оценка | token F1 | смысл |", "|---|---:|---|", "| 3 сида, крупная сетка | 0,4862 | предыдущий протокол, до решения о пяти сидах |", "| 5 сидов, мелкая сетка | 0,5226 | те же внешние фолды после выбора конфигурации |", f"| 5 сидов, худший внутренний фолд | {fmt(values['token_f1']['mean'])} | новая адаптивная оценка |", "", "| внешний фолд | start / extend | token F1 | span F1 | pair F1 | FPR_pair | FPR_token | max FPR_pair внутренних | max FPR_token внутренних |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["folds"]:
        t, m = row["threshold"], row["metrics"]
        lines.append(f"| {row['outer_fold']} | {fmt(t['start'])} / {fmt(t['extend'])} | {fmt(m['token_f1'])} | {fmt(m['span_f1'])} | {fmt(m['pair_f1'])} | {fmt(m['fpr_pair'])} | {fmt(m['fpr_token'])} | {fmt(t['max_inner_fpr_pair'])} | {fmt(t['max_inner_fpr_token'])} |")
    lines.extend(["", "| метрика | mean ± population std |", "|---|---:|"])
    for key in ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy"):
        lines.append(f"| {key} | {fmt(values[key]['mean'])} ± {fmt(values[key]['std'])} |")
    lines.extend(["", f"Максимум внешнего FPR_pair: {fmt(summary['max_outer_fpr_pair'])}, FPR_token: {fmt(summary['max_outer_fpr_token'])}; нарушений: {summary['outer_folds_fpr_pair_over_040']} и {summary['outer_folds_fpr_token_over_040']} соответственно. Условие на наблюдаемых фолдах: {'выполнено' if success else 'не выполнено'}; перенос на новые документы не подтверждён.", "", ("## Ограничения результата" if success else "## Почему целевое значение не достигнуто"), "", f"Token F1: {fmt(values['token_f1']['mean'])}; максимальный внешний FPR_token: {fmt(summary['max_outer_fpr_token'])}. Наши 699 train+val включают 239 пар со спанами и 300 спанов. Малый объём разметки может влиять на качество, но это не доказанная причина. Утверждение о 185 размеченных парах для этого набора не подтверждено; прежние сравнения внешних наборов не парные.", "", "Тесты выполнялись без tee на исходных версиях. В промте 36 все пять сохранённых логов содержат 19 passed и 0 failed. В промте 37 во всех пяти заданиях скрыто одно и то же падение синтетического теста: 22 passed, 1 failed; исправленная версия 23 passed. Полные повторные результаты `artifacts/nested38_test_audit.json`. Test 301 в повторных проверках не использовался.", "", "Пять восстановленных checkpoint сохранены для заранее заданного внешнего фолда 1, а не для всех пяти CV-ансамблей. SHA-256 и части — `weights/frozen/`; после восстановления метрики фолда 1 совпали. Это одна конкретная модель с порогом первого фолда, а не доказательство, что её FPR или F1 равны средним CV.", "", "База 0,5138 получена с внешними негативами; новые числа без них, прямое сравнение ограничено. Наши train+val: 699 пар и 300 размеченных фрагментов; причинное объяснение качества размером разметки не доказано. Прежний отдельный RAGTruth focal: span F1 0,2288 против 0,2982 в другой базе; сравнение непарное. Rubert-base закрыт по скорости 122,82–908,03 мс/1000 символов против 30."])
    (artifacts / "nested38_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    title = "Целевая метрика: условие по наблюдаемым фолдам выполнено" if success else "Целевая метрика: условие по наблюдаемым фолдам не выполнено"
    section = [f"## {title}", "", "Оценка получена на фолдах, использованных для выбора конфигурации; независимого подтверждения нет, test 301 израсходован ранее на другой конфигурации.", "", "| уровень | F1 mean ± std |", "|---|---:|"]
    for key, label in (("token_f1", "токены"), ("pair_f1", "ответы"), ("span_f1", "фрагменты")):
        section.append(f"| {label} | {fmt(values[key]['mean'])} ± {fmt(values[key]['std'])} |")
    section.extend(["", f"Средний FPR_token {fmt(values['fpr_token']['mean'])}, максимум по внешним фолдам {fmt(summary['max_outer_fpr_token'])}; средний FPR_pair {fmt(values['fpr_pair']['mean'])}, максимум {fmt(summary['max_outer_fpr_pair'])}. Текст договора в репозитории не найден: по сообщению куратора п. 5 перечисляет три уровня, п. 6 уровень не уточняет.", "", "Сравнение предшествующих внешних наборов не парное; размер разметки — возможный фактор, не доказанная причина. Наши train+val содержат 699 пар и 300 размеченных фрагментов. Rubert-base закрыт по скорости. Подробности: `artifacts/nested38_report.md`. Веса `weights/frozen/` относятся к выбранному фолду 1 и восстановлены с проверкой, но метрики пяти CV-ансамблей не являются оценкой одной сохранённой модели.", "", "Тесты повторены без tee по исходным версиям; логи в `artifacts/nested38_test_audit.json`, единственное скрытое падение относилось к синтетическому тесту промта 37.", ""])
    status = ROOT / "STATUS.md"
    text = status.read_text(encoding="utf-8")
    marker = "Дата обновления: 2026-11-14.\n"
    if marker not in text:
        raise ValueError("не найдено место для секции статуса")
    status.write_text(text.replace(marker, marker + "\n" + "\n".join(section), 1), encoding="utf-8")
    print("Отчёт и STATUS обновлены.")


if __name__ == "__main__":
    main()
