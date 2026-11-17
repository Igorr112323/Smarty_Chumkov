import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from spanver.evaluate import read_jsonl
from tools.analysis50 import calc
from tools.blend51 import calibrate,fit_isotonic
from tools.collect_nested38 import select_worst
from tools.nested38 import GRID,file_digest,target_inner
from tools.nested_cv35 import GATE,predict
from tools.preflight_13 import id_digest
from tools.sweep_ragtruth30 import prepare,compute
from tools.train_base33 import grouped_folds
from tools.train_scheme13_full import save_json


def main():
    art=ROOT/'artifacts'
    source=art/'ours33.jsonl'
    ref=json.loads((art/'nested38_summary.json').read_text(encoding='utf-8'))
    pairs=read_jsonl(source)
    folds=grouped_folds(pairs,5)
    if file_digest(source)!=ref['gold_train_val_sha256'] or [id_digest(f) for f in folds]!=ref['fold_ids_sha256']:
        raise ValueError('не совпадают исходные данные')
    records={}
    for outer in range(1,6):
        for inner in range(1,6):
            if outer==inner:continue
            rec=json.loads((art/f'ablation51_dual_outer{outer}_inner{inner}.json').read_text(encoding='utf-8'))
            if rec['status']!='готово' or rec['gold_sha256']!=ref['gold_train_val_sha256'] or rec['fold_ids_sha256']!=ref['fold_ids_sha256'] or [(s['architecture'],s['seed']) for s in rec['seeds']] != [(a,i) for a in ('tiny2','tiny') for i in range(5)]:
                raise ValueError('неполная задача')
            if rec['inner_id_sha256']!=id_digest(folds[inner-1]) or rec['outer_id_sha256']!=id_digest(folds[outer-1]):
                raise ValueError('не совпадают ID')
            records[outer,inner]=rec
    if len({r['git_sha'] for r in records.values()})!=1 or len({json.dumps(r['second_source'],sort_keys=True) for r in records.values()})!=1:
        raise ValueError('разные код или версия второго энкодера')
    results={'step1':[],'step12':[]}
    pooled_rows={'step1':[],'step12':[]}
    all_hist=[s for r in records.values() for s in r['seeds']]
    for outer in range(1,6):
        inner_indices=[x for x in range(1,6) if x!=outer]
        records_inner=[records[outer,i] for i in inner_indices]
        if any(set(r['inner_predictions'])!={x['pair_id'] for x in folds[i-1]} for i,r in zip(inner_indices,records_inner)):
            raise ValueError('нет внутренних вероятностей')
        points=[]
        for inner,rec in zip(inner_indices,records_inner):
            for row in folds[inner-1]:
                for left,right,score in rec['inner_predictions'][row['pair_id']]:
                    label=int(any(max(left,s['start'])<min(right,s['end']+1) for s in row['spans']))
                    points.append((score,label))
        blocks=fit_isotonic(points)
        save_json(art/f'calibration51_outer{outer}.json',{'outer_fold':outer,'blocks':blocks,'samples':len(points),'fitted_on':'четыре внутренних фолда; внутренние метки используются для обучения и отбора порога, внешние метки не используются','note':'оценка выбора порога на тех же внутренних данных оптимистична'})
        grids=[]
        for inner,rec in zip(inner_indices,records_inner):
            tuning=folds[inner-1]
            cache={r['pair_id']:{'answer':r['model_output'],'pieces':[[a,b,calibrate(p,blocks)] for a,b,p in rec['inner_predictions'][r['pair_id']]]} for r in tuning}
            prepared=prepare(tuning,cache)
            grids.append([compute(prepared,t,t/2,GATE) for t in GRID])
        choices={'step1':select_worst([r['inner_grid'] for r in records_inner]),'step12':select_worst(grids)}
        if any(choice is None for choice in choices.values()):
            raise ValueError(f'нет допустимого внутреннего порога фолда {outer}')
        primary=records[outer,target_inner(outer-1)+1]
        held=folds[outer-1]
        if not primary['primary'] or set(primary['outer_predictions'])!={r['pair_id'] for r in held} or any(not s['checkpoint'] or not s['checkpoint']['restore_predictions_equal'] for s in primary['seeds']):
            raise ValueError('нет проверенных основных моделей или вероятностей')
        for name in ('step1','step12'):
            rows=[]
            for row in held:
                pieces=primary['outer_predictions'][row['pair_id']]
                if name=='step12': pieces=[[a,b,calibrate(p,blocks)] for a,b,p in pieces]
                pred=predict({'answer':row['model_output'],'pieces':pieces},choices[name]['start'])
                rows.append({'pair_id':row['pair_id'],'answer':row['model_output'],'gold_spans':row['spans'],'predicted_spans':pred})
            results[name].append({'fold':outer,'threshold':choices[name],'metrics':calc(rows)})
            pooled_rows[name].extend(rows)
        manifest={'status':'готово','variant':'dual','outer_fold':outer,'inner_fold':primary['inner_fold'],'git_sha':primary['git_sha'],'gold_train_val_sha256':primary['gold_sha256'],'train_id_sha256':primary['train_id_sha256'],'outer_id_sha256':primary['outer_id_sha256'],'model':'rubert-tiny2 + rubert-tiny, по пять сидов','second_source':primary['second_source'],'gate':GATE,'thresholds':choices,'calibration':f'artifacts/calibration51_outer{outer}.json','clause_expand':True,'sentence_snap':False,'checkpoints':[s['checkpoint'] for s in primary['seeds']],'restore_verified':True}
        save_json(ROOT/'weights/ablation51/dual'/f'fold{outer}'/'manifest.json',manifest)
    for name in ('step1','step12'):
        rows=results[name]
        names=[k for k in rows[0]['metrics'] if k not in ('negative_pairs','false_positive_pairs','matched_spans','type_correct')]
        result={'status':'готово','variant':name,'scope':'адаптивная nested CV 699 train/val, не независимый тест', 'folds':rows, 'macro':{k:statistics.mean(r['metrics'][k] for r in rows) for k in names},'pooled':calc(pooled_rows[name]),'max_fpr_token':max(r['metrics']['fpr_token'] for r in rows),'epochs_ran':[s['epochs_ran'] for s in all_hist],'best_epochs':[s['best_epoch'] for s in all_hist],'speed_ms_per_1000':None,'package_bytes':None,'test301_evaluated':False,'external_data':False,'note':'изотоническая калибровка и отбор порога для step12 обучаются/оцениваются на тех же четырёх внутренних фолдах; внешние метки исключены'}
        save_json(art/f'analysis51_{name}.json',result)
        print(name,json.dumps({'pooled':result['pooled'],'macro':result['macro'],'max_fpr_token':result['max_fpr_token']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
