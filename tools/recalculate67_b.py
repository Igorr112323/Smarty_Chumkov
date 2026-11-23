"""Пересчёт B по 40 сохранённым отчётам; без обучения и test 301."""
import hashlib
import json
import os
import random
import statistics
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from tools.torchless57 import stub_if_absent
stub_if_absent()
from spanver.evaluate import covered_tokens, match_spans, read_jsonl, token_positions
from tools.metrics59 import spans_of, prf, score_pairs
from tools.preflight_13 import id_digest
from tools.rule57 import merge, metrics
from tools.summarize59 import counts_for, inner_shift, pick_threshold, predict_fold
from tools.train_base33 import grouped_folds

GOLD_SHA='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff'


def grid_points(grids):
    starts=[(x['start'],x['extend']) for x in grids[0]]
    if not starts or any([(x['start'],x['extend']) for x in g]!=starts for g in grids):
        raise ValueError('несогласованные сетки inner')
    result=[]
    for idx, (start,extend) in enumerate(starts):
        rows=[g[idx] for g in grids]
        if any(not r['support']['negative_tokens'] for r in rows):
            raise ValueError('нет отрицательных токенов inner')
        val=metrics(merge([r['support'] for r in rows]),0,0)
        result.append({'start':start,'extend':extend,'fpr_token':val['fpr_token'],'token_f1':val['token_f1']})
    return result


def paired_ci(pairs,preds,rounds=10000,seed=67):
    rows=[]
    for p in pairs:
        answer=p['model_output'];gold=covered_tokens(answer,[(s['start'],s['end']) for s in spans_of(p)])
        predicted=covered_tokens(answer,[(s['start'],s['end']) for s in preds[p['pair_id']]])
        all_tokens=covered_tokens(answer,token_positions(answer))
        rows.append((len(gold&predicted),len(predicted-gold),len(gold-predicted),len(gold&all_tokens),len(all_tokens-gold),len(gold-all_tokens)))
    rng=random.Random(seed); n=len(rows);diff=[]
    for _ in range(rounds):
        sums=[0]*6
        for i in (rng.randrange(n) for _ in range(n)):
            row=rows[i]
            for j in range(6):sums[j]+=row[j]
        diff.append(prf(*sums[:3])['f1']-prf(*sums[3:])['f1'])
    diff.sort()
    return {'low':diff[int(.025*rounds)],'high':diff[int(.975*rounds)],'rounds':rounds,'seed':seed,'method':'парный bootstrap по парам; разность token F1 модели и all-positive пересчитывается на каждой выборке'}


def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('только в Actions')
    path=ROOT/'artifacts/recalculated67_b.json';path.parent.mkdir(exist_ok=True)
    output={'status':'ошибка','test301':False,'training_repeated':False,'gate':'пользователь разрешил пересчёт после 140/140; стадия 1 имеет отдельный провал калибровки'}
    try:
        source=ROOT/'artifacts/gold66_trainval.jsonl'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=GOLD_SHA:raise ValueError('SHA train/val')
        pairs=read_jsonl(source)
        if len(pairs)!=699 or len({p['pair_id'] for p in pairs})!=699:raise ValueError('ожидались 699 уникальных train/val')
        by_id={p['pair_id']:p for p in pairs};folds=grouped_folds(pairs,10);foldsha=[id_digest(f) for f in folds]
        reports={};filehash={}
        for outer in range(1,11):
            for inner in inner_shift(outer):
                file=ROOT/f'artifacts/q59_b_outer{outer}_inner{inner}.json'; blob=file.read_bytes();r=json.loads(blob)
                if (r.get('status')!='готово' or r.get('variant')!='q59-b' or r.get('with_question') is not True
                    or r.get('test301') is not False or r.get('gold_sha256')!=GOLD_SHA or r.get('fold_ids_sha256')!=foldsha
                    or r.get('outer_fold')!=outer or r.get('inner_fold')!=inner or r.get('outer_id_sha256')!=foldsha[outer-1]
                    or r.get('inner_id_sha256')!=foldsha[inner-1]):
                    raise ValueError(f'неверная метаинформация отчёта {outer}/{inner}')
                reports[outer,inner]=r;filehash[f'{outer}/{inner}']=hashlib.sha256(blob).hexdigest()
        rows=[];choices={};missing=[]
        for outer in range(1,11):
            grids=[reports[outer,i]['inner_grid'] for i in inner_shift(outer)]
            points=grid_points(grids);choice=pick_threshold(grids)
            lowest=min(points,key=lambda x:(x['fpr_token'],-x['token_f1'],x['start']))
            eligible=sum(p['fpr_token']<=0.4 for p in points)
            item={'fold':outer,'inner_folds':inner_shift(outer),'min_inner_fpr':lowest['fpr_token'],
                  'min_fpr_threshold':lowest['start'],'min_fpr_token_f1':lowest['token_f1'],
                  'eligible_points':eligible,'selected':choice}
            if choice is None:missing.append(outer)
            else:choices[outer]=choice['start']
            rows.append(item)
        output.update(pairs=699,gold_sha256=GOLD_SHA,report_count=len(reports),report_sha256=filehash,
                      selection_rule='по четырём inner: максимум pooled inner token F1 при pooled inner token FPR <= 0,40; при равенстве меньший start',
                      outer_source_rule='предопределён inner=outer%10+1 как в tools/summarize59.py; ошибочный флаг primary не используется; ID outer проверены',
                      folds=rows,folds_without_eligible=missing)
        if missing:
            output.update(status='отказ: нет допустимого inner порога',pooled=None,
                          reason='нет ни одной допустимой точки на inner для всех внешних фолдов; порог вручную и по outer не выбирался')
        else:
            predictions={}; per=[]
            for outer in range(1,11):
                r=reports[outer,outer%10+1]
                expected={p['pair_id'] for p in folds[outer-1]}
                if set(r.get('outer_predictions',{}))!=expected:raise ValueError(f'отсутствуют outer predictions {outer}')
                pred=predict_fold(r,by_id,choices[outer]);predictions.update(pred)
                m=counts_for(folds[outer-1],pred,None)
                per.append({'fold':outer,'pairs':len(expected),'threshold':choices[outer],'outer_source_inner':outer%10+1,
                            'stored_primary_flag':r.get('primary'),'token':m})
            if set(predictions)!=set(by_id):raise ValueError('outer пары не покрывают весь train/val')
            all_positive={p['pair_id']:[{'start':a,'end':b} for a,b in token_positions(p['model_output'])] for p in pairs}
            token=counts_for(pairs,predictions,None);base=counts_for(pairs,all_positive,None)
            soft_hits=sum(match_spans(predictions[p['pair_id']],spans_of(p),.5)[0] for p in pairs)
            exact_hits=sum(match_spans(predictions[p['pair_id']],spans_of(p),1.)[0] for p in pairs)
            pred_n=sum(len(v) for v in predictions.values());gold_n=sum(len(spans_of(p)) for p in pairs)
            soft=2*soft_hits/(pred_n+gold_n) if pred_n+gold_n else 0
            exact=2*exact_hits/(pred_n+gold_n) if pred_n+gold_n else 0
            fprs=[f['token']['fpr_token'] for f in per];f1s=[f['token']['f1'] for f in per]; worst=max(per,key=lambda f:f['token']['fpr_token'])
            ci=paired_ci(pairs,predictions)
            speed=None
            checks={'inner_fpr_all':True,'token_f1_ge_0_5165':token['f1']>=.5165,
                    'pooled_fpr_le_0_4':token['fpr_token']<=.4,'paired_ci_low_gt_0':ci['low']>0,
                    'outer_folds_over_fpr_zero':sum(x>.4 for x in fprs)==0,
                    'speed_le_30':None}
            output.update(status='полный расчёт B',fold_metrics=per,
                          pooled={'token':token,'baseline':base,'delta_f1':token['f1']-base['f1'],
                                  'soft_span_f1_iou_0_5':soft,'exact_span_f1':exact,'pair_f1_exact_hit':score_pairs(pairs,predictions)['pair_f1'],
                                  'predicted_spans':pred_n,'gold_spans':gold_n,'paired_delta_ci95':ci,
                                  'folds_over_fpr':sum(x>.4 for x in fprs),'max_fold':{'fold':worst['fold'],'fpr_token':worst['token']['fpr_token']},
                                  'fold_std':{'token_f1':statistics.pstdev(f1s),'fpr_token':statistics.pstdev(fprs)}},
                          speed_with_question_ms_per_1000=speed,criteria=checks,
                          accepted=False,reason='не все численные критерии выполнены' if any(v is False for v in checks.values()) else 'скорость B с вопросом не измерена')
    except Exception:
        output.update(status='ошибка',error=traceback.format_exc())
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return 0 if output['status']!='ошибка' else 1
if __name__=='__main__':raise SystemExit(main())
