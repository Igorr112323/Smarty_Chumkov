"""Диагностика всех фолдов 66 без искусственного порога и без test 301.

Запускается только в Actions; использует неизменённые сохранённые отчёты и
предопределённую сетку. Для фолда без допустимого порога outer метрику не
считает. Никакого pooled на неполном наборе и подбора по outer.
"""
import hashlib
import json
import os
import statistics
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from spanver.evaluate import match_spans,read_jsonl
from tools.metrics59 import spans_of
from tools.recover66_summary import EXPECTED_COMMIT,validate_report
from tools.summarize66 import GOLD_SHA,GRID,average,decide,support
from tools.preflight65_signal import spearman
from tools.torchless57 import stub_if_absent
stub_if_absent()
from tools.train_base33 import grouped_folds


def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('диагностика финального прогона только в GitHub Actions')
    path=ROOT/'artifacts/diagnostic66.json';path.parent.mkdir(exist_ok=True)
    try:
        gold=ROOT/'artifacts/gold66_trainval.jsonl'
        if hashlib.sha256(gold.read_bytes()).hexdigest()!=GOLD_SHA:raise ValueError('неверный gold SHA')
        pairs=read_jsonl(gold);folds=grouped_folds(pairs,10)
        if len(pairs)!=699:raise ValueError('не 699 пар')
        split=json.loads((ROOT/'spanver/splits.json').read_text())
        if {p['pair_id'] for p in pairs}&set(split['test']):raise ValueError('в gold есть ID test')
        results=[];global_x=[];global_y=[];speed=[];sha={};recoveries=[]
        refs={i:json.loads((ROOT/f'artifacts/q59_a_outer{i}_inner{i%10+1}.json').read_text())['outer_predictions'] for i in range(1,11)}
        model_bytes=None
        for outer in range(1,11):
            inner=outer%10+1;reports=[]
            for seed in range(5):
                p=ROOT/f'artifacts/iter66_outer{outer}_seed{seed}.json';blob=p.read_bytes();r=json.loads(blob)
                validate_report(r,outer,seed,folds);sha[f'{outer}/{seed}']=hashlib.sha256(blob).hexdigest()
                if r['status']!='готово':recoveries.append({'outer':outer,'seed':seed,'status':r['status'],'stage':r['stage'],'raw_sha256':sha[f'{outer}/{seed}'],'reason':'исключение tmp.cleanup() после завершения расчётов'})
                if outer==1 and seed==0:model_bytes=r['weights_package_bytes']
                reports.append(r)
            inavg=average(reports,'inner_candidates');outavg=average(reports,'outer_candidates')
            grid=[]
            for threshold in GRID:
                predicted={k:decide(v,threshold) for k,v in inavg.items()};metrics=support(folds[inner-1],predicted)
                grid.append({'threshold':threshold,'token_f1':metrics['f1'],'fpr_token':metrics['fpr_token']})
            eligible=[g for g in grid if g['fpr_token']<=.4]
            item={'fold':outer,'inner_fold':inner,'inner_pairs':len(folds[inner-1]),'outer_pairs':len(folds[outer-1]),
                  'inner_grid':grid,'eligible_count':len(eligible),
                  'min_inner_fpr':min(g['fpr_token'] for g in grid),
                  'min_fpr_thresholds':[g['threshold'] for g in grid if g['fpr_token']==min(q['fpr_token'] for q in grid)]}
            if eligible:
                best=max(eligible,key=lambda r:(r['token_f1'],-r['threshold']));threshold=best['threshold']
                chosen={k:decide(v,threshold) for k,v in outavg.items()}
                outer_stats=support(folds[outer-1],chosen);count=sum(map(len,chosen.values()));gold_count=sum(len(spans_of(p)) for p in folds[outer-1])
                for bound,label in [(.5,'soft_span_f1'),(1.,'exact_span_f1')]:
                    hits=sum(match_spans(chosen[p['pair_id']],spans_of(p),bound)[0] for p in folds[outer-1])
                    outer_stats[label]=2*hits/(count+gold_count) if count+gold_count else 0.0
                item.update(status='порог найден',threshold=threshold,inner_f1=best['token_f1'],inner_fpr=best['fpr_token'],
                            outer=outer_stats,gap=.4-outer_stats['fpr_token'])
            else:item.update(status='нет допустимого порога',threshold=None,outer=None)
            foldx=[];foldy=[]
            for pair in folds[outer-1]:
                pid=pair['pair_id']; ref=refs[outer][pid];allpieces=[r['outer_subtokens'][pid] for r in reports]
                offsets=[tuple(v[:2]) for v in ref]
                if any([tuple(v[:2]) for v in a]!=offsets for a in allpieces):raise ValueError(f'несовпадение субтокенов {pid}')
                foldx.extend(v[2] for v in ref);foldy.extend(sum(a[j][2] for a in allpieces)/5 for j in range(len(ref)))
            item['rho']=spearman(foldx,foldy);global_x.extend(foldx);global_y.extend(foldy)
            speed.extend(reports[0]['speed_pairs']);results.append(item)
        if len(recoveries)!=1:raise ValueError('неожиданное число восстановлений')
        incomplete=[v for v in results if v['threshold'] is None]
        if not incomplete:raise ValueError('для всех фолдов пороги найдены: следует использовать полный свод')
        observed=[v for v in results if v['outer'] is not None]
        rho=spearman(global_x,global_y);median=statistics.median(v['ms_per_1000'] for v in speed)
        fpr_over=sum(v['outer']['fpr_token']>.4 for v in observed)
        result={'status':'частичный результат: нет допустимого inner порога','source_run':45026585356,
                'code_sha':EXPECTED_COMMIT,'gold_sha256':GOLD_SHA,'pairs_total':len(pairs),
                'trained_reports':50,'raw_report_sha256':sha,'recovery':recoveries,'folds':results,
                'no_eligible_folds':[v['fold'] for v in incomplete],
                'folds_with_valid_threshold':len(observed),'observed_outer_fpr_over_040':fpr_over,
                'folds_fpr_criterion':'провален' if fpr_over>2 else 'не определён: есть фолды без порога',
                'pooled_token_f1':None,'delta_vs_baseline':None,'paired_bootstrap_ci_difference':None,
                'cannot_evaluate':'по зафиксированной сетке не найден допустимый порог для всех 10 фолдов; порог нельзя сдвигать вручную или подбирать по outer',
                'baseline_reference':0.500132,'decisive_criterion_status':'не определён: полный отрыв не вычислен',
                'rho_spearman_vs_A':rho,'rho_meaning':'контроль согласованности признака текущей модели; не происхождение и не блокер',
                'reference_subtokens':len(global_x),'speed_median_ms_per_1000_document':median,'speed_pairs':len(speed),
                'speed_criterion_pass':median<=30,'weights_package_bytes':model_bytes,
                'package_definition':'gzip state_dict teacher+segment, без кода и tokenizer; нижняя граница пакета в этом формате',
                'package_criterion_pass':False if model_bytes>600000 else None,
                'training_repeated':False,'test301':False,
                'checkpoint_A':'checkpoint прогона A в артефактах не сохранён, воспроизведён не был'}
    except Exception:
        result={'status':'ошибка диагностики','error':traceback.format_exc(),'source_run':45026585356,'training_repeated':False,'test301':False}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('folds','raw_report_sha256')},ensure_ascii=False,indent=2))
    return 0 if result['status'].startswith('частичный результат') else 1

if __name__=='__main__':raise SystemExit(main())
