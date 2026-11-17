import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch
import transformers
from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.blend51 import blend
from tools.nested38 import GRID, file_digest, split, target_inner
from tools.nested_cv35 import GATE, averaged, probabilities
from tools.predict_frozen38 import load_state
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.sweep_ragtruth30 import compute, prepare
from tools.train_base33 import grouped_folds
from tools.train46 import train_variant
from tools.train_scheme13_full import save_json


def save_checkpoint(model, outer, seed, arch):
    import tempfile
    from tools.nested38 import PART_BYTES
    folder=ROOT / 'weights/ablation51/dual' / f'fold{outer}' / arch / f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dual51-') as temp:
        path=Path(temp)/'checkpoint.pt'
        torch.save(model.state_dict(),path)
        entry={'seed':seed,'architecture':arch,'filename':'checkpoint.pt','size':path.stat().st_size,'sha256':file_digest(path),'chunk_bytes':PART_BYTES,'chunks':[]}
        with path.open('rb') as handle:
            index=0
            while True:
                part=handle.read(PART_BYTES)
                if not part: break
                name=f'part-{index:03d}.bin'
                (folder/name).write_bytes(part)
                entry['chunks'].append({'name':name,'bytes':len(part),'sha256':hashlib.sha256(part).hexdigest()})
                index+=1
    save_json(folder/'manifest.json',entry)
    return entry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outer',type=int,required=True,choices=range(1,6))
    parser.add_argument('--inner',type=int,required=True,choices=range(1,6))
    parser.add_argument('--git-sha',required=True)
    args=parser.parse_args()
    torch.use_deterministic_algorithms(True)
    source=ROOT/'artifacts/ours33.jsonl'
    reference=json.loads((ROOT/'artifacts/nested38_summary.json').read_text(encoding='utf-8'))
    pairs=read_jsonl(source)
    if file_digest(source)!=reference['gold_train_val_sha256'] or len(pairs)!=699 or id_digest(pairs)!=EXPECTED_IDS:
        raise ValueError('неверные train/val')
    folds=grouped_folds(pairs,5)
    if [id_digest(f) for f in folds]!=reference['fold_ids_sha256']:
        raise ValueError('фолды не совпали')
    outer,inner=args.outer-1,args.inner-1
    training,tuning,held=split(folds,outer,inner)
    primary=inner==target_inner(outer)
    blind=[{k:v for k,v in r.items() if k not in ('spans','label')} for r in held]
    model_manifest=json.loads((ROOT/'models/rubert-tiny/source51.json').read_text(encoding='utf-8'))
    for name, sha in model_manifest['file_sha256'].items():
        if file_digest(ROOT/'models/rubert-tiny'/name)!=sha:
            raise ValueError('неверный SHA второй архитектуры')
    output={'status':'в процессе','variant':'dual','outer_fold':args.outer,'inner_fold':args.inner,'primary':primary,'git_sha':args.git_sha,'torch':str(torch.__version__),'transformers':transformers.__version__,'gold_sha256':file_digest(source),'fold_ids_sha256':reference['fold_ids_sha256'],'train_id_sha256':id_digest(training),'inner_id_sha256':id_digest(tuning),'outer_id_sha256':id_digest(held),'architectures':['rubert-tiny2','rubert-tiny'],'second_source':model_manifest,'seeds':[],'test301':False,'external_data':False,'sentence_snap':False,'clause_expand':True}
    path=ROOT/f'artifacts/ablation51_dual_outer{args.outer}_inner{args.inner}.json'
    save_json(path,output)
    caches={a:{'inner':[],'outer':[]} for a in ('tiny2','tiny')}
    start=time.perf_counter()
    for arch, directory in (('tiny2',ROOT/'models/rubert-tiny2'),('tiny',ROOT/'models/rubert-tiny')):
        for seed in range(5):
            tick=time.perf_counter()
            tokenizer,model=lexfeat.load_feature_model(directory,seed=seed)
            model.base.config._attn_implementation='eager'
            torch.manual_seed(seed)
            info=train_variant(model,tokenizer,training,tuning,seed,'v2',permanent_freeze=True)
            caches[arch]['inner'].append(probabilities(tokenizer,model,tuning))
            checkpoint=None
            if primary:
                original=probabilities(tokenizer,model,blind)
                checkpoint=save_checkpoint(model,args.outer,seed,arch)
                del tokenizer,model
                tokenizer,model=lexfeat.load_feature_model(directory,seed=seed)
                model.base.config._attn_implementation='eager'
                load_state(torch,model,ROOT/'weights/ablation51/dual'/f'fold{args.outer}'/arch/f'seed_{seed}',checkpoint)
                restored=probabilities(tokenizer,model,blind)
                if restored!=original:
                    raise ValueError('вероятности после восстановления не совпали')
                checkpoint['restore_predictions_equal']=True
                caches[arch]['outer'].append(restored)
            info.update({'seed':seed,'architecture':arch,'seconds':time.perf_counter()-tick,'checkpoint':checkpoint})
            output['seeds'].append(info)
            output['elapsed_s']=time.perf_counter()-start
            save_json(path,output)
            print(f'outer {args.outer} inner {args.inner} {arch} seed {seed}: готов',flush=True)
            del tokenizer,model
    def mix(rows,role):
        return {r['pair_id']:blend(averaged(r,caches['tiny2'][role],range(5)),averaged(r,caches['tiny'][role],range(5))) for r in rows}
    inner_cache=mix(tuning,'inner')
    output['inner_predictions']={k:v['pieces'] for k,v in inner_cache.items()}
    prepared=prepare(tuning,inner_cache)
    output['inner_grid']=[compute(prepared,t,t/2,GATE) for t in GRID]
    if primary:
        output['outer_predictions']={k:v['pieces'] for k,v in mix(blind,'outer').items()}
    output['status']='готово'
    output['elapsed_s']=time.perf_counter()-start
    save_json(path,output)


if __name__=='__main__':
    main()
