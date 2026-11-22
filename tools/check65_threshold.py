"""Проверить сетку порогов 63c по сохранённым кандидатам; без модели и обучения."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import json, collections, statistics
from tools.torchless57 import stub_if_absent
stub_if_absent()
from tools.train_base33 import grouped_folds
from tools.metrics59 import prf
from spanver.evaluate import covered_tokens, token_positions, read_jsonl
from pathlib import Path
pairs=read_jsonl(ROOT / 'artifacts/ours33.jsonl'); folds=grouped_folds(pairs,10)
sumr=json.loads((ROOT / 'artifacts/summary63c.json').read_text())
print('fold chosen expected F1 FPR inner FPR outer gap')
out=[]
for outer in range(1,11):
    inner=outer%10+1; rs=[json.loads((ROOT / f'artifacts/seg63c_outer{outer}_inner{inner}_seed{s}.json').read_text()) for s in range(5)]; merged={}
    for r in rs:
        for pid,rows in r['inner_candidates'].items():
            store=merged.setdefault(pid,{})
            for row in rows: store.setdefault((row['start'],row['end']),[]).append(row['risk'])
    if set(merged)!={p['pair_id'] for p in folds[inner-1]}: raise ValueError('pair set mismatch')
    avg={pid:[(a,b,sum(v)/len(v)) for (a,b),v in store.items()] for pid,store in merged.items()}
    if any(len(v)!=5 for store in merged.values() for v in store.values()): raise ValueError('incomplete seed')
    grid=[]
    for i in range(1,20):
        threshold=round(.05*i,2);tp=fp=fn=neg=0
        for p in folds[inner-1]:
            pred=sorted((r for r in avg[p['pair_id']] if r[2]>=threshold),key=lambda r:-r[2])[:3]
            ans=p['model_output'] or ''; gs=covered_tokens(ans,[(s['start'],s['end']) for s in p['spans']]); ps=covered_tokens(ans,[(s[0],s[1]) for s in pred]);tp+=len(gs&ps);fp+=len(ps-gs);fn+=len(gs-ps);neg+=len(token_positions(ans))-len(gs)
        m=prf(tp,fp,fn);m.update(threshold=threshold,fpr=fp/neg if neg else 0);grid.append(m)
    eligible=[x for x in grid if x['fpr']<=.4];best=max(eligible,key=lambda p:(p['f1'],-p['threshold']))
    row=sumr['per_fold'][outer-1]; match=best['threshold']==row['threshold'] and abs(best['f1']-row['inner_threshold_point']['f1'])<1e-12 and abs(best['fpr']-row['inner_threshold_point']['fpr_token'])<1e-12
    out.append({'fold':outer,'threshold_recorded':row['threshold'],'threshold_recomputed':best['threshold'],'inner_f1':best['f1'],'inner_fpr':best['fpr'],'outer_fpr':row['fpr_token'],'gap':.4-row['fpr_token'],'rule_passed':match,'eligible_points':len(eligible),'grid':grid})
    print(outer,row['threshold'],best['threshold'],round(best['f1'],4),round(best['fpr'],4),round(row['fpr_token'],4),round(.4-row['fpr_token'],4),'PASS' if match else 'FAIL')
(ROOT / 'artifacts/threshold65_check.json').write_text(json.dumps({'source':'summary63c, 50 saved candidate reports; no inference or training','grid':'.05 through .95 step .05','rule':'max token F1 with FPR <= .40; ties prefer smaller threshold','folds':out,'all_pass':all(r['rule_passed'] for r in out),'outer_fpr_mean':statistics.mean(r['outer_fpr'] for r in out),'outer_fpr_median':statistics.median(r['outer_fpr'] for r in out)},ensure_ascii=False,indent=2)+'\n')
if not all(r['rule_passed'] for r in out):
    raise SystemExit('гейт 1.3: сохранённый порог не соответствует внутреннему максимуму')
