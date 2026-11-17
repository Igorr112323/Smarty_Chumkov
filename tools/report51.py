import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
ART=ROOT/'artifacts'


def read(name):
    path=ART/name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    base=read('analysis50_v5.json')
    one=read('analysis51_step1.json')
    two=read('analysis51_step12.json')
    ten=read('analysis51_ten.json')
    nine=read('analysis51_nine.json')
    speed=read('speed51_preflight.json')
    package=read('package51_baseline.json')
    runs=read('run51.json') or {}
    columns=[base,one,two,ten,nine]
    lines=['# Промпт 51: ансамбль, калибровка и 10 фолдов','', 'Только сохранённые результаты; пропуск не означает нулевую метрику. Все F1 и FPR пересчитаны по точным счётчикам без округления. CV адаптивна и не является независимым договорным тестом.', '', '| Метрика | В5 | Шаг 1 | Шаг 1+2 | Шаг 3 | Шаг 4 |', '|---|---:|---:|---:|---:|---:|']
    rows=[('token P сводный','pooled','token_precision'),('token R сводный','pooled','token_recall'),('token F1 сводный','pooled','token_f1'),('token P macro','macro','token_precision'),('token R macro','macro','token_recall'),('token F1 macro','macro','token_f1'),('span exact сводный','pooled','span_exact_f1'),('span exact macro','macro','span_exact_f1'),('span overlap сводный','pooled','span_overlap_f1'),('span overlap macro','macro','span_overlap_f1'),('pair F1 сводный','pooled','pair_f1'),('pair F1 macro','macro','pair_f1'),('FPR_token сводный','pooled','fpr_token'),('FPR_token средний','macro','fpr_token'),('FPR_pair сводный','pooled','fpr_pair'),('FPR_pair средний','macro','fpr_pair'),('точность типа сводная','pooled','type_accuracy'),('точность типа macro','macro','type_accuracy')]
    for label,scope,key in rows:
        vals=[str(x[scope][key]) if x else 'нет результата' for x in columns]
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines.append('| FPR_token максимум | '+' | '.join(str(x['max_fpr_token']) if x else 'нет результата' for x in columns)+' |')
    def epochs(x):
        if not x:return 'нет результата'
        e=x['epochs_ran']; return f'{min(e)} / {statistics.median(e)} / {max(e)}'
    lines.append('| эпохи min / медиана / max | '+' | '.join(epochs(x) for x in columns)+' |')
    if speed:
        v5=str(speed['results']['v5_only']['median_ms_per_1000_document_chars'])+' (предв.)'
        dual=str(speed['results']['two_architecture']['median_ms_per_1000_document_chars'])+' (предв.)'
    else:v5=dual='нет замера'
    lines.append('| скорость, мс/1000 знаков | '+v5+' | '+dual+' | не измерена | не измерена | не измерена |')
    lines.append('| размер пакета, байт | '+(str(package['bytes'])+' (старый архив)' if package else 'не измерен')+' | не измерен | не измерен | не измерен | не измерен |')
    old=read('ablation47_run_info.json')
    lines.append('| время Actions, секунд | '+str(old['v5']['wall_seconds'])+' | '+str(runs.get('dual',{}).get('wall_seconds','нет результата'))+' | то же обучение + калибровка | '+str(runs.get('ten',{}).get('wall_seconds','нет результата'))+' | '+str(runs.get('nine',{}).get('wall_seconds','не запускался'))+' |')
    lines += ['', '## Статус и ограничения', '', 'В5 — исторический baseline, не принимается автоматически: полноценная независимая договорная оценка отсутствует.']
    for name,data in [('Шаг 1',one),('Шаг 1+2',two),('Шаг 3',ten),('Шаг 4',nine)]:
        if not data:
            lines.append(f'- {name}: нет результата; принять или отклонить по качеству нельзя.')
            continue
        gain=data['pooled']['token_f1']>base['pooled']['token_f1']
        fpr=data['pooled']['fpr_token']<=0.4 and data['max_fpr_token']<=0.555
        lines.append(f"- {name}: прирост сводного F1 = {gain}, лимиты FPR = {fpr}; без подтверждённых скорости и размера развертываемого пакета приёмка не установлена.")
    if speed:
        lines += ['',f"Предварительный CPU-замер: 5 документов по 1000 знаков, 4 потока, batch 4, медиана трёх прогонов. В5: {v5}; две архитектуры: {dual}. Вторая архитектура ещё не была обучена, декодирование и усреднение не включены; это не итоговый замер полученной модели и не измерение на 2 млн знаков. На этом стенде оба forward-замера выше 30, что является неблагоприятным сигналом, но не доказательством скорости иной реализации."]
    if package:
        lines.append(f"Существующий архив кода занимает {package['bytes']} байт, веса отдельно; он не содержит загрузчика двухархитектурного ансамбля, поэтому размер реального развёртываемого пакета не установлен.")
    lines += ['', 'Шаг 3: 10 групп по doc_id, 4 внутренних фолда (смещения +1…+4 по циклу) для каждого внешнего, 8 фолдов обучения, по 5 сидов. Это меняет не только разбиение, но и объём обучения относительно В5; падение не будет скрыто или заменено результатом В5.', 'Шаги 1–2: по 5 сидов tiny2 и rubert-tiny; вероятности второй модели проецируются по пересечению символов на субтокены tiny2. Не покрытые второй моделью символы получают ноль. Изотоническая калибровка обучена только на четырёх внутренних фолдах; отбор порога на тех же данных может быть оптимистичен. Клаузный expand сохранён, предложенческий снап и компенсация не использованы.', 'Шаг 4 разрешён только после результатов всех предыдущих шагов и только если каждый их F1 меньше 0,51. mFAVA, RuFacts, RAGTruth и дополнительные наборы RusHallu не подключались; test 301 не оценивался. Исходные 699 train/val сохранены без расширения.', '', 'Точные данные: `artifacts/analysis51_*.json`; предварительный замер — `artifacts/speed51_preflight.json`.']
    (ART/'report51.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
