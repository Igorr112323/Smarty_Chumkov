"""Весь pytest в Actions на текущем и исходном коде 52; test 301 отсутствует."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent


def run_suite(folder,name):
    result={'name':name,'cwd':str(folder),'status':'ошибка запуска'}
    try:
        fixture=ROOT/'artifacts/gold66_trainval.jsonl'
        if hashlib.sha256(fixture.read_bytes()).hexdigest()!='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff':raise ValueError('train/val SHA')
        if (folder/'Dataset').exists():raise ValueError('Dataset недопустим: есть риск test 301')
        (folder/'artifacts').mkdir(exist_ok=True)
        # Этот путь нужен тестам калибровки; содержимое — строго 699 train/val.
        shutil.copyfile(fixture,folder/'artifacts/gold.jsonl')
        xml=Path(f'/tmp/pytest67_{name}.xml');log=Path(f'/tmp/pytest67_{name}.log')
        env=dict(os.environ,PYTHONPATH=str(folder),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with log.open('w') as f:
            proc=subprocess.run([sys.executable,'-m','pytest','-q','tests',f'--junitxml={xml}'],cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
        cases=[]
        if xml.exists():
            for case in ET.parse(xml).iter('testcase'):
                event=next((c for c in case if c.tag in ('failure','error','skipped')),None)
                cases.append({'id':f"{case.get('classname')}::{case.get('name')}",
                              'status':'passed' if event is None else event.tag,
                              'message':'' if event is None else (event.get('message') or (event.text or '')[:1000])})
        result.update(status='готово',exit_code=proc.returncode,
                      passed=sum(c['status']=='passed' for c in cases),failed=sum(c['status']=='failure' for c in cases),
                      errors=sum(c['status']=='error' for c in cases),skipped=sum(c['status']=='skipped' for c in cases),
                      cases_total=len(cases),failed_cases=[c for c in cases if c['status'] in ('failure','error')],
                      skipped_cases=[c for c in cases if c['status']=='skipped'],
                      log_tail=log.read_text(errors='replace')[-16000:])
    except Exception:result['error']=traceback.format_exc()
    return result


def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('тесты только Actions')
    current=run_suite(ROOT,'current');baseline=run_suite(ROOT/'baseline52','baseline52')
    def ids(r):return {x['id'] for x in r.get('failed_cases',[])}
    new=sorted(ids(current)-ids(baseline));same=sorted(ids(current)&ids(baseline))
    result={'status':'готово' if current.get('status')==baseline.get('status')=='готово' else 'ошибка тестов',
            'test301':False,'suite':'все тесты без исключений; реальные 1000-парные CSV не доступны в целях изоляции',
            'baseline_commit':'b7dc1cbcf93e982f662fd376c9021c6753bb491b',
            'current':current,'baseline52':baseline,'new_failed_test_ids':new,'same_failed_test_ids':same,
            'criterion_1_1_pass':current.get('passed',0)>=232 and current.get('failed',1)==0 and current.get('errors',1)==0 and not new}
    path=ROOT/'artifacts/regression67_tests.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('current','baseline52')},ensure_ascii=False,indent=2))
    print('current:',{k:current.get(k) for k in ('passed','failed','errors','skipped','exit_code')})
    print('baseline:',{k:baseline.get(k) for k in ('passed','failed','errors','skipped','exit_code')})
    return 0 if result['criterion_1_1_pass'] else 1

if __name__=='__main__':raise SystemExit(main())
