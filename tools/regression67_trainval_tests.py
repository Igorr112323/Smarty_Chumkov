"""Run entire pytest in Actions; no test301 checkout; isolate train/val fixture."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('only Actions')
    if (ROOT/'Dataset').exists():raise ValueError('Dataset is forbidden')
    source=ROOT/'artifacts/gold66_trainval.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff'
    shutil.copyfile(source,ROOT/'artifacts/gold.jsonl')
    xml=Path('/tmp/regression67_trainval.xml'); log=Path('/tmp/regression67_trainval.log')
    env=dict(os.environ,PYTHONPATH=str(ROOT),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with log.open('w') as stream:
        p=subprocess.run([sys.executable,'-m','pytest','-q','-p','tools.pytest67_trainval','tests',f'--junitxml={xml}'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    cases=[]
    for case in ET.parse(xml).iter('testcase'):
        event=next((x for x in case if x.tag in ('error','failure','skipped')),None)
        cases.append({'id':f"{case.get('classname')}::{case.get('name')}",'status':'passed' if event is None else event.tag,'message':'' if event is None else (event.get('message') or (event.text or '')[:1000])})
    result={'status':'готово','test301':False,'gold_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'fixture_scope':'Только test_question_transferred_for_all_pairs: our_rows заменён 699 train/val. Это проверяет наличие вопросов, а не чтение и передачу из Dataset CSV. Остальные тесты без подмен.',
            'note':'test_split64: два теста помечены skip по согласованной политике; сами проверки не изменялись.',
            'exit_code':p.returncode,'total':len(cases),'passed':sum(x['status']=='passed' for x in cases),'failed':sum(x['status']=='failure' for x in cases),'errors':sum(x['status']=='error' for x in cases),'skipped':sum(x['status']=='skipped' for x in cases),
            'unsuccessful_cases':[x for x in cases if x['status']!='passed'],'question_case':[x for x in cases if x['id'].endswith('::test_question_transferred_for_all_pairs')],
            'log_tail':log.read_text(errors='replace')[-18000:]}
    # Диагностика той же train/val смеси; не изменяет калибровку или допуск.
    from spanver.module import verify, CALIBRATION_POINTS
    rows=[json.loads(x) for x in (ROOT/'artifacts/calib_mixture.jsonl').read_text(encoding='utf-8').splitlines()]
    calibration=[]
    for row in rows:
        score=verify(row['text'],row['document'])
        calibration.append({'share':row['share'],'measured':score['ai_share'],
                            'raw':score['meta']['ai_share_raw'],
                            'abs_error':abs(score['ai_share']-row['share'])})
    result['calibration']={'points':CALIBRATION_POINTS,'sample':'смеси из 699 train/val; прежние константы неизменны',
                           'measurements':calibration,'threshold':0.05}
    out=ROOT/'artifacts/regression67_fixed_tests.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='log_tail'},ensure_ascii=False,indent=2))
    return p.returncode
if __name__=='__main__':raise SystemExit(main())
