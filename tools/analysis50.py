import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import match_spans
from spanver.snap import snapped
from tools.analysis48 import pair_counts, measures, totals
from tools.diagnose46 import without_snap


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def calc(rows):
    result = measures(totals(rows))
    neg = sum(not row['gold_spans'] for row in rows)
    fp = sum(not row['gold_spans'] and bool(row['predicted_spans']) for row in rows)
    correct = matches = 0
    for row in rows:
        count, _, good = match_spans(row['predicted_spans'], row['gold_spans'])
        matches += count
        correct += good
    result['fpr_pair'] = fp / neg if neg else 0.0
    result['type_accuracy'] = correct / matches if matches else 0.0
    result['negative_pairs'] = neg
    result['false_positive_pairs'] = fp
    result['matched_spans'] = matches
    result['type_correct'] = correct
    return result


def analyze(variant):
    art = ROOT / 'artifacts'
    prefix = 'ablation47' if variant == 'v5' else 'ablation50'
    summary = read(art / f'{prefix}_{variant}_summary.json')
    if summary['status'] != 'готово' or len(summary['folds']) != 5:
        raise ValueError('вариант неполный')
    sources = read(art / 'predictions40_manifest.json')['outputs']
    folds = []
    for i, src in enumerate(sources):
        payload = (ROOT / src['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != src['sha256']:
            raise ValueError('SHA фолда не совпал')
        rows = [json.loads(line) for line in payload.splitlines() if line]
        outer = i + 1
        rec = read(art / f'{prefix}_{variant}_outer{outer}_inner{outer%5+1}.json')
        if set(rec['outer_predictions']) != {row['pair_id'] for row in rows}:
            raise ValueError('ID фолда не совпали')
        threshold = summary['folds'][i]['threshold']
        for row in rows:
            row.update({'pieces': rec['outer_predictions'][row['pair_id']], 'gate_threshold': 0.45, 'start_threshold': threshold['start'], 'extend_threshold': threshold['extend']})
            row['predicted_spans'] = snapped(without_snap(row), row['answer'], 'expand')
        got = calc(rows)
        expected = summary['folds'][i]['metrics']
        for ours, theirs in [('token_f1','token_f1'),('fpr_token','fpr_token'),('fpr_pair','fpr_pair'),('pair_f1','pair_f1'),('type_accuracy','type_accuracy')]:
            if round(got[ours], 4) != expected[theirs]:
                raise ValueError(f'метрики не совпали: {outer} {ours}')
        for ours, theirs in [('span_exact_f1', 'exact_f1'), ('span_overlap_f1', 'overlap_f1')]:
            if abs(got[ours] - expected[theirs]) > 1e-12:
                raise ValueError(f'не совпала {outer} {ours}')
        folds.append(rows)
    if len({r['pair_id'] for f in folds for r in f}) != 699:
        raise ValueError('ожидалось 699 пар')
    fm = [calc(f) for f in folds]
    names = [k for k in fm[0] if k not in ('negative_pairs','false_positive_pairs','matched_spans','type_correct')]
    macro = {key: statistics.mean(f[key] for f in fm) for key in names}
    pooled = calc(sum(folds, []))
    result = {'variant': variant, 'macro': macro, 'pooled': pooled, 'folds': fm, 'max_fpr_token': max(f['fpr_token'] for f in fm), 'epochs_ran': summary['epochs_ran'], 'best_epochs': summary['best_epochs'], 'test301_evaluated': False, 'silver_used': False, 'scope': 'адаптивная CV на 699 train/val парах, не независимая оценка'}
    return result


def decision(result, base):
    f1 = result['pooled']['token_f1']
    fpr = result['pooled']['fpr_token']
    maximum = result['max_fpr_token']
    baseline_f1 = base['pooled']['token_f1']
    allowed = fpr <= 0.4 and maximum <= 0.555
    accepted = f1 > baseline_f1 and allowed
    # Any exact F1 decrease or either FPR limit broken means harm and stop.
    harm = f1 < baseline_f1 or not allowed
    return {'baseline_pooled_f1': baseline_f1, 'v1_pooled_f1': f1, 'v1_pooled_fpr': fpr, 'v1_max_fold_fpr': maximum, 'f1_improved': f1 > baseline_f1, 'fpr_limits_passed': allowed, 'accepted': accepted, 'harm': harm, 'allow_v2': not harm, 'allow_v3': False, 'rule_note': 'вред по FPR трактуется как нарушение заданных абсолютных лимитов; снижение F1 без допуска на округление'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=('v5','v1'))
    args = parser.parse_args()
    result = analyze(args.variant)
    (ROOT / f'artifacts/analysis50_{args.variant}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.variant == 'v1':
        base = read(ROOT / 'artifacts/analysis50_v5.json')
        status = decision(result, base)
        (ROOT / 'artifacts/decision50.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(status, ensure_ascii=False))
    else:
        print(json.dumps(result['pooled'], ensure_ascii=False))


if __name__ == '__main__':
    main()
