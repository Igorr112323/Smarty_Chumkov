import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from tools.analysis50 import calc
from tools.diagnose46 import without_snap
from tools.preflight_13 import id_digest
from tools.train_base33 import grouped_folds


def main():
    art = ROOT / 'artifacts'
    summary = json.loads((art / 'ablation51_ten_summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'готово' or len(summary['folds']) != 10:
        raise ValueError('неполный результат 10-fold')
    pairs = read_jsonl(art / 'ours33.jsonl')
    folds = grouped_folds(pairs, 10)
    all_rows, measured = [], []
    for index, held in enumerate(folds):
        outer = index + 1
        rec = json.loads((art / f'ablation51_ten_outer{outer}_inner{outer%10+1}.json').read_text(encoding='utf-8'))
        if rec['outer_id_sha256'] != id_digest(held) or set(rec['outer_predictions']) != {r['pair_id'] for r in held}:
            raise ValueError('не совпадает фолд')
        threshold = summary['folds'][index]['threshold']
        rows=[]
        for row in held:
            entry = {'pair_id':row['pair_id'], 'answer':row['model_output'], 'gold_spans':row['spans'], 'pieces':rec['outer_predictions'][row['pair_id']], 'gate_threshold':0.45, 'start_threshold':threshold['start'], 'extend_threshold':threshold['extend']}
            entry['predicted_spans'] = snapped(without_snap(entry), entry['answer'], 'expand')
            rows.append(entry)
        measured.append(calc(rows))
        all_rows.extend(rows)
    if len(all_rows) != 699 or len({r['pair_id'] for r in all_rows}) != 699:
        raise ValueError('неверное число CV-пар')
    names = [k for k in measured[0] if k not in ('negative_pairs','false_positive_pairs','matched_spans','type_correct')]
    output = {'variant':'ten', 'scope':'10 внешних фолдов; четыре заранее заданных внутренних; адаптивная CV 699 train/val, не независимый тест', 'fold_ids_sha256':[id_digest(f) for f in folds], 'fold_sizes':[len(f) for f in folds], 'macro':{k:statistics.mean(m[k] for m in measured) for k in names}, 'pooled':calc(all_rows), 'folds':measured, 'max_fpr_token':max(m['fpr_token'] for m in measured), 'epochs_ran':summary['epochs_ran'], 'best_epochs':summary['best_epochs'], 'test301_evaluated':False, 'external_data_used':False, 'notes':'по сравнению с 5-фолдовой В5 обучение в каждом внутреннем задании содержит 8, а не 3 фолда; 4 внутренние проверки смещены +1,+2,+3,+4 по циклу и фиксированы заранее'}
    (art / 'analysis51_ten.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'macro':output['macro'], 'pooled':output['pooled'], 'max_fpr_token':output['max_fpr_token']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
