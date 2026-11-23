"""Свод только сохранённых прогнозов 66 в Actions, без повторного обучения.

Исходные отчёты не переписываются. Единственный завершённый на этапе cleanup
отчёт допускается лишь после строгой проверки всей сохранённой полезной работы.
"""
import hashlib
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from spanver.evaluate import read_jsonl
from tools.preflight_13 import id_digest
from tools.segments63 import candidates_with_unions
from tools.summarize66 import GOLD_SHA, summarize
from tools.torchless57 import stub_if_absent
stub_if_absent()
from tools.train_base33 import grouped_folds

EXPECTED_COMMIT='bc298fa4f3c6fcbafe7511685e50b96c786e2db4'
EXPECTED_FAILURE="AttributeError: 'str' object has no attribute 'cleanup'"


def failure_is_after_cleanup(report):
    err=report.get('error','')
    return (report.get('outer_fold')==1 and report.get('inner_fold')==2
            and report.get('seed')==0 and report.get('status')=='падение'
            and report.get('stage')=='predictions'
            and 'tools/train66.py", line 106, in main' in err
            and 'tmp.cleanup()' in err and EXPECTED_FAILURE in err
            and isinstance(report.get('weights_package_bytes'),int)
            and report['weights_package_bytes']>0)


def validate_report(report,outer,seed,folds):
    inner=outer%10+1
    if (report.get('outer_fold'),report.get('inner_fold'),report.get('seed'))!=(outer,inner,seed):raise ValueError('неверный идентификатор отчёта')
    if report.get('git_sha')!=EXPECTED_COMMIT or report.get('gold_sha256')!=GOLD_SHA or report.get('test301') is not False:raise ValueError('неверные происхождение или датасет')
    if report.get('epochs')!=20 or report.get('unfreeze_after')!=3:raise ValueError('неверный протокол эпох')
    losses=[h for h in report.get('history',[]) if 'loss' in h]
    events=[h for h in report.get('history',[]) if h.get('unfroze')]
    if [h.get('epoch') for h in losses]!=list(range(1,21)) or any(not math.isfinite(h['loss']) for h in losses) or any(h.get('diverged') for h in report.get('history',[])):
        raise ValueError('не 20 валидных эпох')
    if len(events)!=1 or events[0].get('epoch')!=4 or events[0].get('encoder_lr')!=1e-5:raise ValueError('разморозка не после 3 эпох')
    feat=report.get('feature_history') or {}; fh=feat.get('history') or []
    if not fh or feat.get('epochs_ran')!=len(fh) or not 1<=feat.get('best_epoch',0)<=len(fh):raise ValueError('неполная история teacher')
    if not re.fullmatch(r'[0-9a-f]{64}',report.get('feature_model_sha256','')):raise ValueError('нет SHA teacher')
    training=[p for i,bucket in enumerate(folds) if i not in (outer-1,inner-1) for p in bucket]
    if any(report.get(k)!=id_digest(rows) for k,rows in [('train_id_sha256',training),('inner_id_sha256',folds[inner-1]),('outer_id_sha256',folds[outer-1])]):raise ValueError('сплит не совпал')
    for key,rows in [('inner_candidates',folds[inner-1]),('outer_candidates',folds[outer-1])]:
        saved=report.get(key)
        if not isinstance(saved,dict) or set(saved)!={p['pair_id'] for p in rows}:raise ValueError(f'неполный {key}')
        for pair in rows:
            vals=saved[pair['pair_id']]
            actual=[(v['start'],v['end']) for v in vals]
            expected=candidates_with_unions(pair.get('model_output') or '')
            if actual!=expected:raise ValueError(f'{key}: изменилась нарезка {pair["pair_id"]}')
            if any(not math.isfinite(v['risk']) or not 0<=v['risk']<=1 for v in vals):raise ValueError('невалидный risk')
    subtokens=report.get('outer_subtokens')
    if not isinstance(subtokens,dict) or set(subtokens)!={p['pair_id'] for p in folds[outer-1]}:raise ValueError('неполный сигнал')
    if any(not math.isfinite(v[2]) or not 0<=v[2]<=1 for values in subtokens.values() for v in values):raise ValueError('невалидная вероятность субтокена')
    if seed==0:
        speed=report.get('speed_pairs') or []
        by_id={p['pair_id']:p for p in folds[outer-1]}
        if len(speed)!=len(by_id) or {v['pair_id'] for v in speed}!=set(by_id):raise ValueError('скорость не покрывает фолд')
        for item in speed:
            n=len(by_id[item['pair_id']]['document_text'])
            if item['chars']!=n or n<=0 or item['ms']<=0 or not math.isfinite(item['ms']) or not math.isclose(item['ms_per_1000'],item['ms']*1000/n,rel_tol=1e-12):raise ValueError('невалидный замер скорости')
    if report.get('status')!='готово' and not failure_is_after_cleanup(report):raise ValueError('неизвестная ошибка отчёта; не использовать')
    if report.get('status')=='готово' and report.get('stage')!='завершено':raise ValueError('готовый отчёт не завершён')


def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('финальный свод только в GitHub Actions')
    path=ROOT/'artifacts/recovery66_summary.json'
    try:
        gold=ROOT/'artifacts/gold66_trainval.jsonl'
        if hashlib.sha256(gold.read_bytes()).hexdigest()!=GOLD_SHA:raise ValueError('неверный SHA gold')
        pairs=read_jsonl(gold);folds=grouped_folds(pairs,10)
        if len(pairs)!=699 or sum(len(p['spans']) for p in pairs)!=300:raise ValueError('неверное число пар или спанов')
        splits=json.loads((ROOT/'spanver/splits.json').read_text())
        if {p['pair_id'] for p in pairs}&set(splits['test']):raise ValueError('обнаружены тестовые ID')
        reports={}; hashes={}; recoveries=[]
        for outer in range(1,11):
            for seed in range(5):
                source=ROOT/f'artifacts/iter66_outer{outer}_seed{seed}.json'
                blob=source.read_bytes(); report=json.loads(blob)
                hashes[f'{outer}/{seed}']=hashlib.sha256(blob).hexdigest()
                validate_report(report,outer,seed,folds)
                if report['status']!='готово':
                    recoveries.append({'outer':outer,'seed':seed,'raw_sha256':hashes[f'{outer}/{seed}'],
                                       'original_status':report['status'],'stage':report['stage'],
                                       'exception':EXPECTED_FAILURE,'action':'только in-memory статус для свода; исходный отчёт не изменён'})
                    report=dict(report,status='готово')
                reports[outer,seed]=report
        if len(recoveries)!=1:raise ValueError('ожидался ровно один установленный сбой cleanup')
        refs={i:json.loads((ROOT/f'artifacts/q59_a_outer{i}_inner{i%10+1}.json').read_text())['outer_predictions'] for i in range(1,11)}
        result=summarize(pairs,reports,folds,refs)
        result['recovery']={'source_run':45026585356,'training_repeated':False,'reports_checked':50,
                            'raw_report_sha256':hashes,'errors_recovered':recoveries,
                            'validation':'границы всех кандидатов, сплиты, SHA и истории, пять сидов, субтокены и скорость seed 0'}
    except Exception:
        result={'status':'ошибка свода','error':traceback.format_exc(),'source_run':45026585356,
                'training_repeated':False,'test301':False}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('folds',)},ensure_ascii=False,indent=2))
    return 0 if result['status']=='готово' else 1

if __name__=='__main__':raise SystemExit(main())
