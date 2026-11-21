"""Свод только завершённых 10 фолдов 64; без второго замера test 301."""
import json
import statistics
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.run64 import load_data
from tools.metrics64 import measure
from tools.metrics59 import baseline_all_positive, bootstrap_ci


def main():
    from tools.nested_cv35 import predict
    pairs, folds, split = load_data()
    reports = []
    for outer in range(1, 11):
        path = ROOT / f'artifacts/p64_fold{outer:02}.json'
        if not path.exists():
            raise ValueError(f'нет отчёта фолда {outer}')
        r = json.loads(path.read_text(encoding='utf-8'))
        if r.get('status') != 'готово' or r.get('gold_sha256') != split['gold_sha256'] or r.get('fold_ids_sha256') != split['fold_ids_sha256'] or r.get('outer_fold') != outer or r.get('inner_fold') != outer % 10 + 1 or [s['seed'] for s in r.get('seeds', [])] != list(range(5)):
            raise ValueError(f'неполный/несовместимый отчёт фолда {outer}')
        if set(r['outer_predictions']) != {p['pair_id'] for p in folds[outer-1]}:
            raise ValueError(f'ошибка состава отложенных пар фолда {outer}')
        reports.append(r)
    if len({r['git_sha'] for r in reports}) != 1:
        raise ValueError('Разные коммиты обучения')
    all_pred = {}; per_fold = []
    for r, held in zip(reports, folds):
        threshold = r['threshold_point']['start']
        pred = {p['pair_id']: predict({'answer': p['model_output'], 'pieces': r['outer_predictions'][p['pair_id']]}, threshold) for p in held}
        m = measure(held, pred)
        if m != {k: v for k, v in r['metrics'].items() if k != 'threshold'}:
            raise ValueError(f"Метрики фолда {r['outer_fold']} не воспроизводятся")
        per_fold.append({'fold': r['outer_fold'], 'threshold': threshold, 'train_pairs': r['train_pairs'],
                         'inner_pairs': r['inner_pairs'], **m})
        all_pred.update(pred)
    pooled = measure(pairs, all_pred)
    baseline = baseline_all_positive(pairs)
    fprs = [r['fpr_token'] for r in per_fold]
    f1s = [r['token']['f1'] for r in per_fold]
    worst = max(per_fold, key=lambda r: r['fpr_token'])
    speed = reports[0].get('speed')
    previous_path = ROOT / 'artifacts/question59.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))['A_without_question'] if previous_path.exists() else None
    result = {'status': 'готово', 'gold_sha256': split['gold_sha256'], 'fold_ids_sha256': split['fold_ids_sha256'],
              'git_sha': reports[0]['git_sha'], 'pooled': pooled, 'baseline_all_positive': baseline,
              'delta_vs_baseline': pooled['token']['f1'] - baseline['f1'],
              'folds_over_fpr': sum(x > .4 for x in fprs),
              'max_fold': {'fold': worst['fold'], 'fpr_token': worst['fpr_token']},
              'std': {'token_f1': statistics.pstdev(f1s), 'fpr_token': statistics.pstdev(fprs)},
              'per_fold': per_fold, 'ci_95_pair_bootstrap': bootstrap_ci(pairs, all_pred, rounds=10000, seed=64),
              'speed': speed,
              'contract': {'3.1_f1_ge_0.50': pooled['token']['f1'] >= .50,
                           '3.2_fpr_le_0.40': pooled['fpr_token'] <= .40,
                           '3.7_speed_le_30': speed['ms_per_1000'] <= 30 if speed else None},
              'previous_699': {'source': 'artifacts/question59.json / A_without_question', 'metrics': previous, 'span_f1_soft': None, 'note': 'свод предыдущей линии; другое разбиение и правило объединения внутренних фолдов; pair F1 может иметь иное определение, сравнение не причинное'},
              'old_test': 'входит в 1000 как обычные пары, отдельного замера нет',
              'attribution': 'эффекты объёма и чистоты одновременно изменены: причинное разложение без контрольных прогонов не идентифицируется',
              'train_size_note': 'сохранён внутренний фолд схемы 52; фактические train_pairs указаны по фолдам'}
    path = ROOT / 'artifacts/summary64.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pooled': pooled, 'folds_over_fpr': result['folds_over_fpr'], 'max_fold': result['max_fold'], 'std': result['std'], 'contract': result['contract']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
