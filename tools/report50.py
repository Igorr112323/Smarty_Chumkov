import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifacts'


def load(name):
    path = ART / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    base, variant = load('analysis50_v5.json'), load('analysis50_v1.json')
    audit, converted = load('mfava50_audit.json'), load('mfava50_conversion.json')
    decision = load('decision50.json')
    runs = load('run50.json') or {}
    old_runs = load('ablation47_run_info.json')
    columns = [('В5', base), ('Вариант 1 — TAPT', variant), ('Вариант 2', None), ('Вариант 3', None)]
    lines = ['# Промпт 50: mFAVA-ru и адаптация', '', '## Источник и ограничение разметки', '', f"Файл `Dataset/mFAVA_ru/test.json` из коммита `e1ec868a681f7686c630dd68e06b7117a94f9d42`: {audit['bytes']} байт, SHA-256 `{audit['sha256']}` — совпал. {audit['total_rows']} записей, из них {audit['russian_rows']} русских. Датасет взят из репозитория, Hugging Face для данных не использовался.", '', f"Полные теги находятся в `gold_annotations`; в `generated_text` обнаружено {sum(audit['generated_text_complete_tags'].values())} полных распознаваемых тегов, хотя имеются повреждённые фрагменты. Здесь {audit['tag_open_close_counts']['open:contradictory']} открывающих тегов `contradictory`, а также опечатки и нарушения вложенности; исходная статистика 793 спана не воспроизводится простым чтением всех типов. Во всех {audit['rows_with_any_complete_tag']} строках есть хотя бы один тег. Это число — строки, не число верифицированных, согласованных спанов.", '', f"Из 267 строк только {audit['unique_reference_answer_pairs']} уникальных пар reference/answer; {audit['duplicate_pair_groups']} групп повторов и во всех {audit['duplicate_groups_disagreeing_gold']} аннотации различаются. После удаления тегов gold точно совпадает с generated_text только в {audit['tag_stripped_gold_vs_generated']['exact']} строках; в {audit['tag_stripped_gold_vs_generated']['nonwhitespace_disagreement']} различия затрагивают не только пробелы. Расхождения и опечатки нельзя автоматически превращать в проверенные границы.", '', f"Консервативная предварительная конвертация: {converted['converted_rows']} строк, {converted['converted_spans']} спанов, не допущена к обучению; ещё {converted['rejected_rows']} строк отклонено из-за неоднозначности. Файл `artifacts/mfava50_converted_partial.jsonl`, диагностика — `artifacts/mfava50_conversion.json`. Разметки повторов не объединялись и типы не отображались на проектную схему. В TAPT gold и silver не использовались.", '', f"Точные нормализованные совпадения с 699 CV-парами: ссылок {audit['normalized_exact_reference_overlap']}, ответов {audit['normalized_exact_answer_overlap']}; проверка выбранных 120-символьных фрагментов: {audit['sampled_120_char_reference_snippet_overlap_pairs']} пар. Частичные и семантические пересечения полностью не исключены.", '', '## Результаты на 699 CV-парах', '', 'Числа вычислены по целочисленным счётчикам без предварительного округления. Macro — среднее точных показателей фолдов; сводный показатель — объединение out-of-fold прогнозов. Выбор по этой CV адаптивен, это не независимый договорный тест. Обе span-метрики отдельно, без подмены token F1.', '', '| Показатель | В5 | Вариант 1: TAPT | Вариант 2 | Вариант 3 |', '|---|---:|---:|---:|---:|']
    rows = [('Token P сводный','pooled','token_precision'),('Token R сводный','pooled','token_recall'),('Token F1 сводный','pooled','token_f1'),('Token P macro','macro','token_precision'),('Token R macro','macro','token_recall'),('Token F1 macro','macro','token_f1'),('Span exact F1 сводный','pooled','span_exact_f1'),('Span exact F1 macro','macro','span_exact_f1'),('Span overlap F1 сводный','pooled','span_overlap_f1'),('Span overlap F1 macro','macro','span_overlap_f1'),('Pair F1 сводный','pooled','pair_f1'),('Pair F1 macro','macro','pair_f1'),('FPR_token сводный','pooled','fpr_token'),('FPR_token средний','macro','fpr_token'),('FPR_pair сводный','pooled','fpr_pair'),('FPR_pair средний','macro','fpr_pair'),('Точность типа сводная','pooled','type_accuracy'),('Точность типа macro','macro','type_accuracy')]
    for label, scope, field in rows:
        values = [str(data[scope][field]) if data else 'не запускался' for _,data in columns]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines.append('| FPR_token максимум фолда | ' + ' | '.join(str(data['max_fpr_token']) if data else 'не запускался' for _,data in columns) + ' |')
    def epochs(data):
        if not data: return 'не запускался'
        values=data['epochs_ran']
        return f"{min(values)} / {statistics.median(values)} / {max(values)}"
    lines.append('| Эпохи min / медиана / max | ' + ' | '.join(epochs(data) for _,data in columns) + ' |')
    lines.append('| Время Actions, секунд | ' + ' | '.join([str(old_runs['v5']['wall_seconds']), str(runs.get('wall_seconds', 'в процессе')), 'не запускался', 'не запускался']) + ' |')
    lines += ['', 'Число 0,1633 из запроса — macro exact F1 прежнего двухсидового В2, а не В5. База В5 показана выше; прежний потолок 0,3760 относился только к фиксированным числам предсказаний в каждой паре и не является потолком новой модели.', '', '## Принятие и правило остановки', '']
    if decision:
        state = 'принят' if decision['accepted'] else 'отклонён'
        lines.append(f"Вариант 1: **{state}**. Строгий рост сводного F1: {decision['f1_improved']}; оба лимита FPR: {decision['fpr_limits_passed']}. Порог максимального FPR по фолдам применён как явно заданное в запросе значение 0,5550, без округления результатов.")
        if decision['harm']:
            lines.append('Вариант 1 навредил по заданному правилу: варианты 2 и 3 не запускались; остаётся В5.')
        else:
            lines.append('Для варианта 2 требуется согласовать небезопасные границы, дубли и соответствие типов ручной разметки; он пока не запускался. Вариант 3 без двух положительных результатов запрещён.')
    else:
        lines.append('Вариант 1 ещё не завершён; решения о вариантах 2–3 нет.')
    lines += ['', '## Протокол', '', 'Вариант 1: отдельный label-free MLM-TAPT для каждого из пяти сидов, одна эпоха по всем 267 references и generated_text, блоки до 256 токенов, маскирование 15%, AdamW lr 1e-5, batch 4. Повторяющиеся записи не дедуплицировались. Затем вложенная CV по протоколу В5: до 15 эпох, patience 3, пять сидов, embeddings постоянно заморожены на supervised этапе; порог выбран только по внутренним фолдам при FPR_token и FPR_pair ≤0,40. Клаузный expand сохранён; предложенческий снап и компенсация отключены.', '', 'Test 301 не оценивался, silver_annotations не использовались; другие новые внешние наборы не подключались. Вход V1 — внешний mFAVA без меток, а исходные 699 — прежний train/val. Новых независимых оценок нет; метрики, основанные на предварительно изученных фолдах, не подтверждают договор.', '', 'Аудит источника: `artifacts/mfava50_audit.json`; точные данные: `artifacts/analysis50_v5.json`, `artifacts/analysis50_v1.json`; решение: `artifacts/decision50.json`.']
    (ART / 'mfava50_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
