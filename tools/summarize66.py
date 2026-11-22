"""Итерация 65/66: ансамбль и парный кластерный bootstrap без test 301."""
import json
import math
import random
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from spanver.evaluate import covered_tokens,match_spans,read_jsonl,token_positions
from tools.metrics59 import baseline_all_positive,prf,spans_of
from tools.preflight_13 import components,our_keys,id_digest
from tools.preflight65_signal import spearman

GOLD_SHA='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff'
GRID=[round(.05*i,2) for i in range(1,20)]


def decide(rows,t):
    good=[r for r in rows if r['risk']>=t];good.sort(key=lambda x:-x['risk'])
    return [{'start':r['start'],'end':r['end']} for r in good[:3]]


def counts(pair,spans):
    text=pair['model_output'] or ''
    gold=covered_tokens(text,[(s['start'],s['end']) for s in spans_of(pair)])
    pred=covered_tokens(text,[(s['start'],s['end']) for s in spans])
    return (len(gold&pred),len(pred-gold),len(gold-pred),len(token_positions(text))-len(gold))


def support(pairs,chosen):
    c=[counts(p,chosen.get(p['pair_id'],[])) for p in pairs]
    tp,fp,fn,neg=(sum(x[i] for x in c) for i in range(4));out=prf(tp,fp,fn)
    out['fpr_token']=fp/neg if neg else 0.0
    nneg=[p for p in pairs if not spans_of(p)]
    out['fpr_pair']=sum(bool(chosen.get(p['pair_id'])) for p in nneg)/len(nneg) if nneg else 0.0
    return out


def average(reports,key):
    first=reports[0][key]; out={}
    if any(set(r[key])!=set(first) for r in reports):raise ValueError('наборы пар разных сидов не совпадают: '+key)
    for pid in first:
        tables=[]
        for report in reports:
            rows=report[key][pid]; table={(r['start'],r['end']):r['risk'] for r in rows}
            if len(table)!=len(rows):raise ValueError('повтор кандидатов')
            tables.append(table)
        if any(set(t)!=set(tables[0]) for t in tables):raise ValueError('границы кандидатов расходятся')
        out[pid]=[{'start':a,'end':b,'risk':sum(t[(a,b)] for t in tables)/len(tables)} for a,b in tables[0]]
    return out


def paired_ci(pairs,chosen,rounds=10000,seed=65):
    groups=components(pairs,our_keys); stats=[]
    for group in groups:
        row=[0,0,0,0,0,0]
        for p in group:
            a=counts(p,chosen.get(p['pair_id'],[]));text=p['model_output'] or ''
            gold=covered_tokens(text,[(s['start'],s['end']) for s in spans_of(p)])
            all_tokens=covered_tokens(text,token_positions(text))
            b=(len(gold&all_tokens),len(all_tokens-gold),len(gold-all_tokens))
            row=[v+w for v,w in zip(row,(*a[:3],*b))]
        stats.append(row)
    rng=random.Random(seed); values=[]
    for _ in range(rounds):
        sums=[0]*6
        for _ in stats:
            row=stats[rng.randrange(len(stats))]
            for j in range(6):sums[j]+=row[j]
        values.append(prf(*sums[:3])['f1']-prf(*sums[3:])['f1'])
    values.sort()
    return {'low':values[int(.025*rounds)],'high':values[int(.975*rounds)],'rounds':rounds,'seed':seed,'cluster_count':len(stats),'method':'paired cluster bootstrap by document components, 2.5/97.5 percentiles'}


def summarize(pairs,reports,folds,refs):
    selected={};per=[];x=[];y=[];speeds=[]
    for outer in range(1,11):
        inner=outer%10+1;rr=[reports[(outer,s)] for s in range(5)]
        for r in rr:
            if r['status']!='готово' or r['gold_sha256']!=GOLD_SHA or r['epochs']!=20 or r['unfreeze_after']!=3:raise ValueError('некорректный отчёт')
            training=[p for i,fold in enumerate(folds) if i not in (outer-1,inner-1) for p in fold]
            if r['train_id_sha256']!=id_digest(training) or r['inner_id_sha256']!=id_digest(folds[inner-1]) or r['outer_id_sha256']!=id_digest(folds[outer-1]):raise ValueError('неправильные фолды')
        inner_avg=average(rr,'inner_candidates');outer_avg=average(rr,'outer_candidates')
        if set(inner_avg)!={p['pair_id'] for p in folds[inner-1]} or set(outer_avg)!={p['pair_id'] for p in folds[outer-1]}:raise ValueError('пары фолда не совпали')
        grid=[]
        for t in GRID:
            pred={k:decide(v,t) for k,v in inner_avg.items()};point=support(folds[inner-1],pred)
            grid.append({'threshold':t,'f1':point['f1'],'fpr':point['fpr_token']})
        allowed=[v for v in grid if v['fpr']<=.4]
        if not allowed:raise ValueError(f'нет допустимого порога fold {outer}; вся inner grid: {grid}')
        best=max(allowed,key=lambda v:(v['f1'],-v['threshold']));threshold=best['threshold']
        chosen={k:decide(v,threshold) for k,v in outer_avg.items()};selected.update(chosen)
        point=support(folds[outer-1],chosen)
        per.append({'fold':outer,'threshold':threshold,'inner_grid':grid,'outer':point,'gap':.4-point['fpr_token'],
                    'feature_model_sha256':[r['feature_model_sha256'] for r in rr]})
        ref=refs[outer]; foldx=[];foldy=[]
        for pair in folds[outer-1]:
            pid=pair['pair_id'];a=ref[pid]; allpieces=[r['outer_subtokens'][pid] for r in rr]
            offsets=[(p[0],p[1]) for p in a]
            if any([(p[0],p[1]) for p in pp]!=offsets for pp in allpieces):raise ValueError('разошлись offsets контрольных субтокенов')
            foldx.extend(p[2] for p in a);foldy.extend(sum(pp[i][2] for pp in allpieces)/5 for i in range(len(a)))
        x.extend(foldx);y.extend(foldy);per[-1]['rho']=spearman(foldx,foldy)
        times=rr[0]['speed_pairs']
        if {q['pair_id'] for q in times}!={p['pair_id'] for p in folds[outer-1]}:raise ValueError('скорость не по всем парам')
        speeds.extend(times)
    if set(selected)!={p['pair_id'] for p in pairs}:raise ValueError('OOF покрытие неполное')
    token=support(pairs,selected); baseline=baseline_all_positive(pairs)['f1']
    if abs(baseline-.500132)>.0000005:raise ValueError('базлайн не воспроизведён')
    total=sum(map(len,selected.values())); gold=sum(len(spans_of(p)) for p in pairs)
    soft=sum(match_spans(selected[p['pair_id']],spans_of(p),.5)[0] for p in pairs)
    exact=sum(match_spans(selected[p['pair_id']],spans_of(p),1.)[0] for p in pairs)
    ci=paired_ci(pairs,selected); fprs=[r['outer']['fpr_token'] for r in per];rho=spearman(x,y)
    size=reports[(1,0)]['weights_package_bytes']; speed=statistics.median(s['ms_per_1000'] for s in speeds)
    softf=2*soft/(total+gold) if total+gold else 0;exactf=2*exact/(total+gold) if total+gold else 0
    delta=token['f1']-baseline
    checks={'2.1_f1':token['f1']>=.55,'2.2_delta':delta>=.03,'2.3_fpr':token['fpr_token']<=.40,
            '2.4_precision':token['precision']>=.45,'2.5_soft_span':softf>=.35,
            '2.6_folds':sum(z>.4 for z in fprs)<=2,'2.7_ci_reported':True,'2.8_speed':speed<=30,
            '2.9_package':False if size>600000 else None}
    return {'status':'готово','pairs':len(pairs),'gold_sha256':GOLD_SHA,
            'feature_source':'FeatureModel обученная внутри текущего фолда и сида; не A',
            'checkpoint_A':'checkpoint прогона A в артефактах не сохранён, воспроизведён не был',
            'rho_spearman_vs_A':rho,'rho_meaning':'контроль согласованности признака, не доказательство происхождения и не блокер',
            'subtokens':len(x),'pooled':{'token':token,'soft_span_f1':softf,'exact_span_f1':exactf,'predicted_spans':total,'gold_spans':gold},
            'baseline':baseline,'delta_vs_baseline':delta,'ci_difference':ci,'folds_over_fpr':sum(z>.4 for z in fprs),
            'max_fold':max(per,key=lambda r:r['outer']['fpr_token'])['fold'],'fpr_range':[min(fprs),max(fprs)],
            'std':{'fpr_token':statistics.pstdev(fprs),'f1':statistics.pstdev(r['outer']['f1'] for r in per)},'folds':per,
            'speed_median_ms_per_1000':speed,'speed_pairs':len(speeds),
            'speed_definition':'полный односидовый teacher+сегмент, время на пару; нормирование на число знаков документа',
            'weights_package_bytes_lower_bound':size,'package_note':'gzip полных состояний teacher и segment без tokenizer и кода — нижняя граница пакета в данном формате; если она в лимите, нужна проверка полного пакета',
            'checks':checks,'advance_to_test':bool(delta>=.03 and ci['low']>0),'line63_closed_negative':delta<.03,
            'test301':False}


def main():
    from tools.torchless57 import stub_if_absent
    stub_if_absent()
    from tools.train_base33 import grouped_folds
    from tools.nested38 import file_digest
    path=ROOT/'artifacts/summary66.json'
    try:
        gold=ROOT/'artifacts/gold66_trainval.jsonl'
        if file_digest(gold)!=GOLD_SHA:raise ValueError('неверный SHA данных')
        pairs=read_jsonl(gold);folds=grouped_folds(pairs,10)
        reports={};missing=[]
        for outer in range(1,11):
            for seed in range(5):
                p=ROOT/f'artifacts/iter66_outer{outer}_seed{seed}.json'
                if not p.exists():missing.append([outer,seed]);continue
                reports[outer,seed]=json.loads(p.read_text())
        if missing:raise ValueError('нет обязательных отчётов: '+str(missing))
        if len({r['git_sha'] for r in reports.values()})!=1:raise ValueError('версии кода отличаются')
        refs={i:json.loads((ROOT/f'artifacts/q59_a_outer{i}_inner{i%10+1}.json').read_text())['outer_predictions'] for i in range(1,11)}
        result=summarize(pairs,reports,folds,refs)
    except Exception:
        import traceback
        result={'status':'ошибка свода','error':traceback.format_exc(),'test301':False}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('folds',)},ensure_ascii=False,indent=2))
    return 0 if result['status']=='готово' else 1
if __name__=='__main__':raise SystemExit(main())
