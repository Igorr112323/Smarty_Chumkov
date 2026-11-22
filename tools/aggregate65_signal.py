"""Свод всех десяти фолдов контроля сигнала; частичный результат не проходит гейт."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from tools.preflight65_signal import spearman


def main():
    folder=Path(sys.argv[1]); reports=[]; errors=[]
    for i in range(1,11):
        path=folder/f'signal65_fold{i:02}.json'
        if not path.is_file(): errors.append({'fold':i,'error':'нет отчёта'});continue
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('status')!='готово': errors.append({'fold':i,'error':row.get('error','ошибка')});continue
        reports.append(row)
    x=[v for r in reports for v in r['values_A']];y=[v for r in reports for v in r['values_candidate']]
    done=not errors and len(reports)==10 and sum(r['pairs'] for r in reports)==699
    rho=spearman(x,y) if done else None
    result={'status':'готово' if done else 'неполная проверка','pairs':sum(r.get('pairs',0) for r in reports),
            'subtokens':len(x),'rho_spearman':rho,'criterion_1_1_pass':bool(done and rho>=.9),
            'reference':'сохранённые OOF субтокенные вероятности q59 A',
            'checkpoint_candidate':'архивы weights51-ten, Actions 44076340166, не доказано что это checkpoint A',
            'folds':[{k:v for k,v in r.items() if k not in ('values_A','values_candidate')} for r in reports],
            'errors':errors,'test301':False,'training':False}
    target=ROOT/'artifacts/signal65_check.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'pairs':result['pairs'],'subtokens':result['subtokens'],
                      'rho_spearman':rho,'criterion_1_1_pass':result['criterion_1_1_pass'],'errors':len(errors)},ensure_ascii=False))
    return 0 if result['criterion_1_1_pass'] else 1


if __name__=='__main__': raise SystemExit(main())
