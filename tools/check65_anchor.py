"""Воспроизвести A по сохранённым прогнозам и прежнему методу, без модели/test."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from tools.torchless57 import stub_if_absent
stub_if_absent()
from spanver.evaluate import read_jsonl
from tools.summarize59 import load_reports,score_run
pairs=read_jsonl(ROOT/'artifacts/ours56_docs.jsonl')
reports=load_reports('q59_a')
if len(pairs)!=699 or len(reports)!=40: raise ValueError('нет 699 пар и 40 исходных отчётов A')
result=score_run(reports,pairs,with_bootstrap=False)
if result['status']!='готово': raise ValueError('неполная оценка A')
m=result['full'];actual={'token_f1':m['token']['f1'],'fpr_token':m['token']['fpr_token'],
                         'span_f1_exact':m['exact_span_f1'],'pair_f1':m['pair_f1']}
expected={'token_f1':.508888,'fpr_token':.378691,'span_f1_exact':.234086,'pair_f1':.303191}
checks={k:abs(actual[k]-expected[k])<=.0000005 for k in actual}
out={'status':'готово','pairs':len(pairs),'reports':len(reports),'expected_rounded_6':expected,
     'actual':actual,'checks':checks,'all_pass':all(checks.values()),
     'thresholds':result['thresholds'],'folds_over_fpr':result['per_fold']['folds_over_fpr'],
     'max_fold':result['per_fold']['max_fold'],'std':result['per_fold']['std'],
     'test301':False,'training':False,'model_inference':False,
     'note':'повторная обработка сохранённых вероятностей A через существующий summarize59.score_run'}
(ROOT/'artifacts/anchor65_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
if not out['all_pass']: raise SystemExit('якорь A не воспроизведён')
