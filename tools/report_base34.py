import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    diagnostic = json.loads((ROOT / "artifacts/base34_diagnostic.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / "artifacts/actions33_cv_report_base3.json").read_text(encoding="utf-8"))
    recalculated = json.loads((ROOT / "artifacts/base34_recalculated.json").read_text(encoding="utf-8"))
    speed = json.loads((ROOT / "artifacts/actions33_speed_report_base.json").read_text(encoding="utf-8"))
    fold = diagnostic["folds"][0]
    held = fold["holdout"]
    free = held["best_unrestricted_diagnostic_only"]
    lines = ["# Дефекты прогона rubert-base", "", "Исходный прогон: два фолда; по условию куратора результат не засчитывается как валидный результат схемы 4. Диагностика не обучала модель и не использовала test 301. Исходный артефакт с checkpoint использовался только для чтения в Actions; новый артефакт содержит только JSON.", "", "## 1. Нулевой первый фолд", "", f"Checkpoint fold 1 загружен строго: {fold['checkpoint_loaded_strictly']}; SHA-256 `{fold['checkpoint_sha256']}`. На отложенном фолде получены вероятности для {held['cached_pairs']} из {held['pairs']} пар. Через gate 0,45 проходят {held['pairs_passing_gate']} пар, но ни в одной нет субтокена с вероятностью ≥ {held['chosen_start']:.2f}; при start {held['chosen_start']:.2f} и extend {held['chosen_extend']:.2f} выдано {held['predicted_spans']} спанов в {held['predicted_pairs']} парах. Исходные метрики воспроизведены.", "", f"В исходной сетке {held['grid_count']} сочетаний, из них {held['eligible_count']} удовлетворяют обоим FPR ≤ 0,40; среди них нет ни одного с ненулевым token F1. Поэтому выбор пустого предсказания — следствие ограничений и правила выбора, а не ошибка загрузки. Без ограничений лучшая по token F1 диагностическая точка {free['start']:.2f}/{free['extend']:.2f}: token F1 {free['token']['f1']:.4f}, span F1 {free['span']['f1']:.4f}, FPR_pair {free['fpr_pair']:.4f}, FPR_token {free['fpr_token']:.4f}. Она нарушает лимит FPR_token и не считается исправленным результатом.", "", "В тренере добавлен подсчёт предсказанных пар и спанов; при нуле спанов фолд теперь помечается STOP и общая CV-сводка не выдаётся как успешная. Причину невозможности получить ненулевую допустимую точку это не устраняет; выдавать пригодную оценку без изменения ограничений нельзя.", "", "### 20 случайных пар первого фолда, seed 34", "", "Для каждой пары приведены минимум, медиана и максимум вероятностей субтокенов ответа. Все 20 имеют 0 субтокенов ≥ 0,55 и 0 предсказанных спанов.", "", "| pair_id | gold-спаны | субтокенов | минимум | медиана | максимум |", "|---|---:|---:|---:|---:|---:|"]
    for row in held["sample_pairs"]:
        lines.append(f"| {row['pair_id']} | {row['gold_spans']} | {row['subwords']} | {row['min']:.4f} | {row['median']:.4f} | {row['max']:.4f} |")
    lines += ["", "По всем субтокенам этих 20 пар:", "", "| gold-класс | число | 1% | 5% | 10% | 25% | медиана | 75% | 90% | 95% | 99% | максимум |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, title in (("sample_gold_positive_subwords", "+"), ("sample_gold_negative_subwords", "−")):
        item = held[key]
        values = [item["quantiles"][str(value)] for value in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)] + [item["max"]]
        lines.append(f"| {title} | {item['count']} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "## 2. Проверка mean ± std", "", "В исходном JSON 0,3137 — средняя token precision, а не token F1. Обе сводки вычислялись по двум фолдам со стандартным отклонением генеральной совокупности (ddof=0). При значениях 0 и x среднее равно x/2, и такое std тоже равно x/2; это не признак выборки из одного фолда. Независимый пересчёт совпал с исходным JSON после округления для всех метрик.", "", "| метрика | фолд 1 | фолд 2 | mean точно | std ddof=0 точно | std ddof=1 точно |", "|---|---:|---:|---:|---:|---:|"]
    for key in ("token_precision", "token_f1", "span_f1", "fpr_pair", "fpr_token"):
        item = recalculated["metrics"][key]
        lines.append(f"| {key} | {item['fold_values'][0]:.4f} | {item['fold_values'][1]:.4f} | {item['mean_exact']:.5f} | {item['std_population_exact']:.5f} | {item['std_sample_exact']:.5f} |")
    lines += ["", "Эти цифры являются проверкой арифметики исходного невалидного прогона, а не новой оценкой схемы 4. Для наглядности подсчёт сводки вынесен в отдельную функцию и покрыт тестом.", "", "## 3. Энкодер и learning rate", "", "Оба checkpoint загружены строго и сравнены с теми же исходными весами, подтверждёнными SHA-256. Группы optimizer восстановлены из исходного кода; runtime-снимок optimizer не сохранялся.", "", "| фолд | всего параметров в группах | энкодер и embeddings | голова | изменённых элементов в 12 слоях относительно старта | изменённых тензоров слоёв |", "|---:|---:|---:|---:|---:|---:|"]
    for row in diagnostic["folds"]:
        layers = row["parameters"]["encoder_layers"]
        lines.append(f"| {row['fold']} | {row['optimizer_total_parameters_reconstructed']} | {row['optimizer_encoder_parameters_reconstructed']} | {row['optimizer_head_parameters_reconstructed']} | {layers['changed_elements']} / {layers['parameters']} | {layers['changed_tensors']} / {layers['tensor_count']} |")
    lines += ["", "В коде все параметры этих групп имеют requires_grad и передаются AdamW; веса 12 слоёв изменились относительно исходного checkpoint. Версия о замороженном энкодере данными не подтверждается; повторное обучение фолда 2 не запускалось. Состояние конца первой эпохи не сохранялось, поэтому сравнение именно эпохи 1 с эпохой 3 не проверено. Изменение весов показывает обновление относительно старта, но не заменяет отсутствующий промежуточный снимок.", "", "Фактические LR в лог не записывались. По коду планировщика восстановлены LR, применявшиеся на шагах обоих фолдов (22 шага на эпоху, всего 66, warmup 6):", "", "| эпоха | encoder первый → последний | head первый → последний |", "|---:|---:|---:|"]
    for row in fold["learning_rates"]["epochs"]:
        lines.append(f"| {row['epoch']} | {row['encoder_lr_first']:.10f} → {row['encoder_lr_last']:.10f} | {row['head_lr_first']:.10f} → {row['head_lr_last']:.10f} |")
    lines += ["", "У второго фолда исходный span F1 0,1087 при FPR_pair 0,0091 и FPR_token 0,0539. Он ниже прежней базы 0,2982, но это один фолд, с другим составом обучения и подбором на том же отложенном фолде. Это не доказывает, что размер модели не влияет на качество.", "", "## 4. Скорость", "", "| знаков документа | мс/1000 знаков модели | мс/пару модели |", "|---:|---:|---:|"]
    for row in speed["rows"]:
        lines.append(f"| {row['length']} | {row['model_ms_per_1000_doc']:.2f} | {row['model_ms_per_pair']:.2f} |")
    lines += ["", "Во всех четырёх замерах превышена норма 30 мс/1000 знаков документа. На длине 20000 значение 2456,42 мс/пару; для этой нагрузки документы были искусственно удлинены повторением реальных текстов только в бенчмарке. ONNX и int8 не проверялись; ускорение x2–3 — предположение, не измерение.", "", "Дополнительно: user-данные прежнего рабочего дерева до синхронизации ветки сохранены в git stash. Диагностический запуск: https://github.com/Igorr112323/Smarty_Chumkov/actions/runs/43884555732 . Подробный JSON: `artifacts/base34_diagnostic.json`; пересчёт: `artifacts/base34_recalculated.json`."]
    (ROOT / "artifacts/base34_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
