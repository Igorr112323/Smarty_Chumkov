"""Свод согласованного 5-seed ансамбля; запуск в GitHub Actions.

Порог ансамбля выбирается по усреднённым вероятностям внутреннего фолда.
При отсутствии любого сида или порога приёмочный свод не строится.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import match_spans, read_jsonl
from tools.metrics59 import spans_of
from tools.train_base33 import grouped_folds
from tools.train63 import decide, pick_threshold, score, support_of
from tools.nested38 import file_digest
from tools.summarize63 import bootstrap

SEEDS = range(5)


def average_candidates(reports, field):
    first = reports[0][field]
    if any(set(r[field]) != set(first) for r in reports):
        raise ValueError(f'{field}: разные наборы пар')
    merged = {}
    for pair_id in first:
        tables = []
        for report in reports:
            rows = report[field][pair_id]
            table = {(r['start'], r['end']): r['risk'] for r in rows}
            if len(table) != len(rows):
                raise ValueError(f'{field}/{pair_id}: повтор границ')
            tables.append(table)
        if any(set(table) != set(tables[0]) for table in tables):
            raise ValueError(f'{field}/{pair_id}: разные кандидаты')
        merged[pair_id] = [dict(start=start, end=end,
                                risk=sum(table[(start, end)] for table in tables) / len(tables))
                           for start, end in tables[0]]
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='seg63c')
    parser.add_argument('--out', default='artifacts/summary63c.json')
    parser.add_argument('--rounds', type=int, default=2000)
    args = parser.parse_args()
    pairs = read_jsonl(ROOT / 'artifacts/ours33.jsonl')
    folds = grouped_folds(pairs, 10)
    reports = {}
    for path in sorted((ROOT / 'artifacts').glob(f'{args.prefix}_outer*_inner*_seed*.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        if (report.get('status') in ('готово', 'готово: нет допустимого порога')
                and report.get('epochs') == 20 and report.get('unfreeze_after') == 3
                and report.get('encoder_lr') == 1e-5 and 'inner_candidates' in report
                and 'outer_candidates' in report):
            reports[(report['outer_fold'], report['inner_fold'], report['seed'])] = report
    missing = []
    for outer in range(1, 11):
        inner = outer % 10 + 1
        for seed in SEEDS:
            if (outer, inner, seed) not in reports:
                missing.append([outer, inner, seed])
    out = ROOT / args.out
    if missing:
        out.write_text(json.dumps({'status': 'неполный прогон', 'missing': missing}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('Нет обязательных отчётов:', len(missing))
        return 1
    selected_reports = [reports[(outer, outer % 10 + 1, seed)] for outer in range(1, 11) for seed in SEEDS]
    if len({r['git_sha'] for r in selected_reports}) != 1:
        raise ValueError('Отчёты разных версий кода нельзя объединять')
    if {r['gold_sha256'] for r in selected_reports} != {file_digest(ROOT / 'artifacts/ours33.jsonl')}:
        raise ValueError('Отчёты не соответствуют входному набору')
    predicted, per_fold = {}, []
    for outer in range(1, 11):
        inner = outer % 10 + 1
        selected = [reports[(outer, inner, seed)] for seed in SEEDS]
        for key in ('gold_sha256', 'train_id_sha256', 'inner_id_sha256', 'outer_id_sha256'):
            if len({r[key] for r in selected}) != 1:
                raise ValueError(f'фолд {outer}: несовпадение {key}')
        inner_avg = average_candidates(selected, 'inner_candidates')
        outer_avg = average_candidates(selected, 'outer_candidates')
        tuning, held = folds[inner - 1], folds[outer - 1]
        if set(inner_avg) != {p['pair_id'] for p in tuning} or set(outer_avg) != {p['pair_id'] for p in held}:
            raise ValueError(f'фолд {outer}: неверный набор пар')
        choice = pick_threshold(tuning, inner_avg)
        if choice is None:
            raise ValueError(f'фолд {outer}: нет допустимого порога ансамбля на внутреннем фолде')
        threshold, tune_point = choice
        point = score(held, outer_avg, threshold)
        predicted.update({pair_id: decide(rows, threshold) for pair_id, rows in outer_avg.items()})
        per_fold.append({'fold': outer, 'seeds': 5, 'threshold': threshold,
                         'inner_threshold_point': tune_point, **point})
    token = support_of(pairs, predicted)
    count = sum(len(spans) for spans in predicted.values())
    gold = sum(len(spans_of(pair)) for pair in pairs)
    def span_f1(bound):
        tp = sum(match_spans(predicted[p['pair_id']], spans_of(p), bound)[0] for p in pairs)
        return 2 * tp / (count + gold) if count + gold else 0
    ci = bootstrap(pairs, predicted, rounds=args.rounds)
    fprs = [p['fpr_token'] for p in per_fold]
    soft, exact = span_f1(.5), span_f1(1.)
    checks = {'3.1_token_f1': token['f1'] >= .60, '3.2_token_precision': token['precision'] >= .55,
              '3.3_fpr_token': token['fpr_token'] <= .30, '3.4_span_f1_soft': soft >= .35,
              '3.5_ci_low': ci['low'] > .5001, '3.6_folds_over_fpr': all(x <= .40 for x in fprs),
              '3.7_ms_per_1000': None, '3.8_package_mb': None}
    summary = {'status': 'готово', 'pairs': len(pairs), 'seeds': list(SEEDS), 'per_fold': per_fold,
               'pooled': {'token_f1': token['f1'], 'token_precision': token['precision'],
                          'token_recall': token['recall'], 'fpr_token': token['fpr_token'],
                          'fpr_pair': token['fpr_pair'], 'span_f1_soft': soft, 'span_f1_exact': exact,
                          'predicted_spans': count, 'gold_spans': gold},
               'folds_over_fpr': sum(x > .40 for x in fprs),
               'max_fold': max(per_fold, key=lambda x: x['fpr_token'])['fold'],
               'fpr_range': [min(fprs), max(fprs)],
               'std': {'token_f1': statistics.pstdev(p['token']['f1'] for p in per_fold),
                       'fpr_token': statistics.pstdev(fprs)},
               'ci': ci, 'baseline': .500132, 'delta_vs_baseline': token['f1'] - .500132,
               'checks': checks, 'all_criteria_met': False,
               '3_9_continue': (not all(checks[k] for k in ('3.1_token_f1','3.2_token_precision','3.3_fpr_token'))
                                and token['f1'] > .508888 and token['precision'] > .442003 and token['fpr_token'] < .378691)}
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary['pooled'], ensure_ascii=False)); print('Проверки:', checks)
    return 0


if __name__ == '__main__':
    sys.exit(main())
