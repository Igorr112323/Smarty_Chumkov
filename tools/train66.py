"""Итерация 65/66: одна пара (outer, seed), запуск только GitHub Actions.

Признак: FeatureModel схемы 52, обученная здесь на train данного разбиения.
Ни checkpoint A, ни архивы 51 не используются. Test 301 отсутствует.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch
from spanver import lexfeat,segment_model
from spanver.evaluate import read_jsonl
from tools.nested38 import split,file_digest
from tools.nested_cv35 import probabilities
from tools.preflight_13 import id_digest
from tools.train_base33 import grouped_folds
from tools.train46 import train_variant
from tools import train63 as T
from tools.rows66 import build_rows

GOLD_SHA='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff'


def save(report,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outer',type=int,required=True,choices=range(1,11));ap.add_argument('--seed',type=int,required=True,choices=range(5));ap.add_argument('--git-sha',required=True);args=ap.parse_args()
    inner=args.outer%10+1; path=ROOT/f'artifacts/iter66_outer{args.outer}_seed{args.seed}.json'
    result={'status':'падение','outer_fold':args.outer,'inner_fold':inner,'seed':args.seed,'git_sha':args.git_sha,'test301':False,
            'checkpoint_A':'checkpoint прогона A в артефактах не сохранён, воспроизведён не был'}
    try:
        if os.environ.get('GITHUB_ACTIONS')!='true': raise RuntimeError('обучение только в GitHub Actions')
        source=ROOT/'artifacts/gold66_trainval.jsonl'
        if file_digest(source)!=GOLD_SHA: raise ValueError('неверный train/val SHA')
        memory=json.loads((ROOT/'artifacts/memory66_step.json').read_text())
        if not (memory.get('memory_pass') and memory.get('gold_sha256')==GOLD_SHA and memory.get('batch')==T.BATCH and memory.get('micro_batch')==T.MICRO and memory.get('sequence_width')==512): raise ValueError('не подтвержден гейт памяти')
        pairs=read_jsonl(source); folds=grouped_folds(pairs,10)
        if len(pairs)!=699: raise ValueError('не 699 пар')
        train,tune,held=split(folds,args.outer-1,inner-1)
        result.update(gold_sha256=GOLD_SHA,train_id_sha256=id_digest(train),inner_id_sha256=id_digest(tune),outer_id_sha256=id_digest(held),epochs=20,unfreeze_after=3,
                      training_protocol='feature: схема 52 v2 early stopping; segment: 20 эпох, unfrozen с 4-й',feature_source='FeatureModel текущего фолда и сида, обучена в данном прогоне; не A',stage='feature_training')
        save(result,path)
        torch.use_deterministic_algorithms(True,warn_only=True);lexfeat.with_question(False)
        tokenizer,teacher=lexfeat.load_feature_model(ROOT/'models/rubert-tiny2',seed=args.seed);teacher.base.config._attn_implementation='eager';torch.manual_seed(args.seed)
        result['feature_history']=train_variant(teacher,tokenizer,train,tune,args.seed,'v2',permanent_freeze=True)
        # SHA реально обученной и используемой FeatureModel, а не начальных весов.
        tmp=tempfile.TemporaryDirectory(prefix='feature66-')
        fp=Path(tmp.name)/'teacher.pt';torch.save(teacher.state_dict(),fp);result['feature_model_sha256']=file_digest(fp)
        result['stage']='feature_inference';save(result,path)
        cache=probabilities(tokenizer,teacher,pairs)
        subtoken={p['pair_id']:cache[p['pair_id']]['pieces'] for p in pairs}
        result['outer_subtokens']={p['pair_id']:subtoken[p['pair_id']] for p in held}
        # Teacher выгружается до сегментного обучения (как при тесте памяти).
        # На этапе скорости он восстанавливается из собственного временного SHA.
        torch.manual_seed(args.seed)
        model=segment_model.CandidateModel(teacher.base)
        del teacher
        gc.collect()
        T.build_rows=build_rows;T.UNFREEZE_AFTER=3
        questions={p['pair_id']:p['query_text'] for p in pairs}
        rows=[]
        for p in train: rows.extend(build_rows(tokenizer,p,questions[p['pair_id']],subtoken[p['pair_id']]))
        result.update(training_rows=len(rows),positive_rows=sum(r['label'] for r in rows),stage='segment_training');save(result,path)
        history=T.train_model(model,tokenizer,rows,args.seed,epochs=20)
        if any(x.get('diverged') for x in history) or len([x for x in history if 'loss' in x])!=20: raise ValueError('неполное или расходящееся обучение')
        result['history']=history;result['stage']='predictions';save(result,path)
        result['inner_candidates']=T.predict_pairs(model,tokenizer,tune,questions,subtoken)
        result['outer_candidates']=T.predict_pairs(model,tokenizer,held,questions,subtoken)
        # Все пары held, одна модель (seed 0), время включает teacher + признаки,
        # упаковку 512 и вывод кандидатов. Никаких test-пар.
        if args.seed==0:
            _,teacher=lexfeat.load_feature_model(ROOT/'models/rubert-tiny2',seed=args.seed)
            teacher.base.config._attn_implementation='eager'
            if file_digest(fp)!=result['feature_model_sha256']: raise ValueError('checkpoint teacher изменился')
            teacher.load_state_dict(torch.load(fp,map_location='cpu',weights_only=True),strict=True)
            teacher.eval()
            times=[]
            for pair in held:
                start=time.perf_counter(); p=lexfeat.pair_probabilities(tokenizer,teacher,pair)
                T.predict_pairs(model,tokenizer,[pair],questions,{pair['pair_id']:[] if p is None else p['pieces']})
                elapsed=(time.perf_counter()-start)*1000
                chars=len(pair.get('document_text') or '')
                times.append({'pair_id':pair['pair_id'],'ms':elapsed,'chars':chars,'ms_per_1000':elapsed*1000/max(1,chars)})
            result['speed_pairs']=times
        # Размер реальных обученных состояний, а не только голов: teacher и segment.
        if args.outer==1 and args.seed==0:
            with tempfile.TemporaryDirectory(prefix='pkg66-') as tmp:
                raw=Path(tmp)/'weights.pt'; gz=Path(tmp)/'weights.pt.gz'
                torch.save({'teacher':teacher.state_dict(),'segment':model.state_dict()},raw)
                with raw.open('rb') as inp,gzip.open(gz,'wb',compresslevel=9) as out:
                    for block in iter(lambda:inp.read(1024*1024),b''):out.write(block)
                result['weights_package_bytes']=gz.stat().st_size
                result['weights_package_note']='gzip полных двух обученных state_dict; без исходного кода и tokenizer; нижняя граница пакета'
        tmp.cleanup()
        result['status']='готово';result['stage']='завершено';save(result,path)
    except Exception:
        result['error']=traceback.format_exc();save(result,path);print(result['error'],file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
