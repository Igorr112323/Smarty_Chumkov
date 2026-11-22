"""Контроль 1.1: сохранённый checkpoint схемы 51 против сохранённых OOF-оценок A.

Только инференс в Actions; никакого обучения, сохранения веса или использования
старого test 301. Это КАНДИДАТ на checkpoint, не утверждение идентичности A.
"""
import argparse
import hashlib
import json
import math
import resource
import statistics
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__); ranks=[0.0]*len(values); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        rank=(i+1+j)/2
        for k in range(i,j): ranks[order[k]]=rank
        i=j
    return ranks


def spearman(x,y):
    if len(x)!=len(y) or len(x)<2: raise ValueError('мало совпавших субтокенов')
    rx,ry=ranks(x),ranks(y); mx,my=statistics.mean(rx),statistics.mean(ry)
    vx=sum((a-mx)**2 for a in rx);vy=sum((b-my)**2 for b in ry)
    if not vx or not vy: raise ValueError('константные вероятности')
    return sum((a-mx)*(b-my) for a,b in zip(rx,ry))/math.sqrt(vx*vy)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,required=True,choices=range(1,11))
    parser.add_argument('--weights',required=True)
    parser.add_argument('--report51',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    result={'fold':args.fold,'status':'ошибка', 'source':'weights51-ten из Actions 44076340166; кандидат, не checkpoint A',
            'reference':f'q59_a_outer{args.fold}_inner{args.fold%10+1}', 'test301':False, 'training':False}
    try:
        import torch
        from spanver import lexfeat
        from spanver.evaluate import read_jsonl
        from tools.preflight_13 import id_digest
        from tools.train_base33 import grouped_folds
        from tools.nested_cv35 import probabilities, averaged
        from tools.predict_frozen38 import load_state
        from spanver.segment_model import CandidateModel
        torch.use_deterministic_algorithms(True, warn_only=True)
        lexfeat.with_question(False)
        pairs=read_jsonl(ROOT/'artifacts/ours33.jsonl')
        folds=grouped_folds(pairs,10); held=folds[args.fold-1]
        r=json.loads((ROOT/f'artifacts/q59_a_outer{args.fold}_inner{args.fold%10+1}.json').read_text())
        old_path=Path(args.report51)
        candidates=list(old_path.rglob('*.json')) if old_path.is_dir() else [old_path]
        old=[json.loads(p.read_text(encoding='utf-8')) for p in candidates]
        old=[o for o in old if o.get('outer_fold')==args.fold and o.get('inner_fold')==args.fold%10+1]
        if len(old)!=1: raise ValueError('не найден уникальный старый отчёт 51')
        old=old[0]
        keys=('train_id_sha256','inner_id_sha256','outer_id_sha256')
        result['split_identical'] = all(old.get(k)==r.get(k) for k in keys) and r.get('outer_id_sha256')==id_digest(held)
        result['split_comparison']={k:{'reference_A':r.get(k),'weights51':old.get(k)} for k in keys}
        if not result['split_identical']:
            raise ValueError('веса 51 обучены на другом разбиении; во избежание утечки проверка остановлена')
        root=Path(args.weights)
        dirs=[p for p in root.rglob('seed_0') if p.is_dir()]
        if len(dirs)!=1: raise ValueError(f'не найден единственный каталог seed_0: {len(dirs)}')
        root=dirs[0].parent
        caches=[]; checks=[]; memory=None
        for seed in range(5):
            folder=root/f'seed_{seed}'
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if manifest['seed']!=seed: raise ValueError('неверный сид в манифесте')
            tokenizer,model=lexfeat.load_feature_model(ROOT/'models/rubert-tiny2',seed=seed)
            model.base.config._attn_implementation='eager'
            # Проверяет каждый кусок, полный SHA-256 и strict state_dict.
            load_state(torch,model,folder,manifest)
            checks.append({'seed':seed,'sha256':manifest['sha256'],
                           'manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()})
            caches.append(probabilities(tokenizer,model,held))
            if args.fold==1 and seed==0:
                # Синтетический forward/backward микро-пакета, без оптимизатора,
                # обучения по данным или изменения сохранённых весов.
                sample=CandidateModel(model.base)
                sample.train(); ids=torch.ones((4,512),dtype=torch.long); mask=torch.ones_like(ids); feat=torch.zeros((4,10))
                logits,kinds=sample(ids,mask,feat)
                (logits.mean()+kinds.mean()).backward()
                memory={'micro_batch':4,'tokens':512,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                        'optimizer_step':False,'note':'CPU, синтетический прямой/обратный проход; не размер модели в пакете'}
                del sample,ids,mask,feat,logits,kinds
            del model,tokenizer
        reference=r['outer_predictions']; x=[];y=[]; align=[]
        if set(reference)!={p['pair_id'] for p in held}: raise ValueError('A не содержит полный внешний фолд')
        for pair in held:
            pid=pair['pair_id']; ref=reference[pid]
            pieces=averaged(pair,caches,range(5))['pieces']
            if [(q[0],q[1]) for q in ref]!=[(q[0],q[1]) for q in pieces]:
                raise ValueError(f'расхождение границ субтокенов для {pid}')
            for a,b in zip(ref,pieces):
                if not math.isfinite(a[2]) or not math.isfinite(b[2]): raise ValueError('нечисловая вероятность')
                x.append(a[2]);y.append(b[2])
            align.append(pid)
        result.update(status='готово', pairs=len(align), subtokens=len(x),rho=spearman(x,y),
                      checkpoint_checksums=checks,values_A=x,values_candidate=y,memory=memory)
    except Exception:
        result['error']=traceback.format_exc()
    out.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('values_A','values_candidate','error')},ensure_ascii=False))
    if result['status']!='готово': print(result['error'],file=sys.stderr); return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
