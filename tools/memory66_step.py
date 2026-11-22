"""Один реальный тренировочный шаг, только в Actions. Веса не сохраняются."""
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def available_memory():
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
    available=info['MemAvailable']
    cmax=Path('/sys/fs/cgroup/memory.max'); current=Path('/sys/fs/cgroup/memory.current')
    quota=None
    if cmax.exists() and current.exists() and cmax.read_text().strip()!='max':
        quota=int(cmax.read_text().strip())-int(current.read_text().strip())+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        available=min(available, quota)
    return available,info['MemTotal'],quota


def run(report):
    import torch
    import torch.nn.functional as F
    from spanver import lexfeat,segment_model
    from spanver.evaluate import read_jsonl
    from tools.nested38 import split
    from tools.train_base33 import grouped_folds
    from tools.train63 import collate, BATCH, MICRO, LR, ENCODER_LR, GRAD_CLIP, CE_WEIGHT, IOU_WEIGHT
    from tools.rows66 import build_rows
    gold=ROOT/'artifacts/gold66_trainval.jsonl'
    digest=hashlib.sha256(gold.read_bytes()).hexdigest()
    if digest!='8e1d3012afce096292b299b3849c80bdbdd739eb3f869af6744b71b6f03928ff': raise ValueError('SHA 699-парного gold не совпал')
    pairs=read_jsonl(gold)
    if len(pairs)!=699: raise ValueError('нужно 699 пар')
    train,inner,outer=split(grouped_folds(pairs,10),0,1)
    ref={}
    for fold in range(1,11):
        r=json.loads((ROOT/f'artifacts/q59_a_outer{fold}_inner{fold%10+1}.json').read_text())
        ref.update(r['outer_predictions'])
    available,total,quota=available_memory()
    torch.manual_seed(0); torch.set_num_threads(4)
    tokenizer,base=lexfeat.load_feature_model(ROOT/'models/rubert-tiny2',seed=0)
    base.base.config._attn_implementation='eager'
    model=segment_model.CandidateModel(base.base)
    del base
    # Батч из реально размеченного train; показатели для 10 признаков построены
    # по сохранённым вероятностям A. Никакие веса A/51 не загружаются.
    rows=[]
    for pair in train:
        rows.extend(build_rows(tokenizer,pair,pair['query_text'],ref.get(pair['pair_id'],[])))
    rows.sort(key=lambda r:len(r['input_ids']),reverse=True)
    batch=rows[:BATCH]
    if len(batch)!=BATCH: raise ValueError('неполный батч')
    lengths=[len(row['input_ids']) for row in batch]
    # Наихудшая выделяемая ширина: реальные строки, padding до 512, mask=0.
    for row in batch:
        n=512-len(row['input_ids'])
        if n<0: raise ValueError('последовательность превышает 512')
        row['input_ids']+= [tokenizer.pad_token_id]*n
        row['attention_mask']+= [0]*n
    head=[p for n,p in model.named_parameters() if n.startswith(('proj.','binary.','kind.'))]
    encoder=[p for n,p in model.named_parameters() if n.startswith('base.')]
    for p in encoder: p.requires_grad_(True)
    optimizer=torch.optim.AdamW([{'params':head,'lr':LR},{'params':encoder,'lr':ENCODER_LR}],weight_decay=.01)
    model.train();optimizer.zero_grad(); start=time.perf_counter(); value=0.0
    for offset in range(0,len(batch),MICRO):
        part=batch[offset:offset+MICRO];piece=collate(part,tokenizer.pad_token_id)
        logits,kinds=model(piece['input_ids'],piece['attention_mask'],piece['features'])
        bce=F.binary_cross_entropy_with_logits(logits,piece['labels'])
        ce=F.cross_entropy(kinds,piece['types'])
        iou=segment_model.soft_iou_loss(logits,piece['labels'],piece['weights'])
        loss=bce+CE_WEIGHT*ce+IOU_WEIGHT*iou
        if not torch.isfinite(loss): raise ValueError('не конечный loss')
        (loss*len(part)/len(batch)).backward();value+=float(loss.detach())*len(part)/len(batch)
    torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
    optimizer.step()
    elapsed=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report.update({'status':'готово','gold_sha256':digest,'dataset_pairs':len(pairs),'train_pairs':len(train),
                   'outer_fold':1,'inner_fold':2,'batch':BATCH,'micro_batch':MICRO,'sequence_width':512,
                   'min_real_length':min(lengths),'max_real_length':max(lengths),
                   'feature_count':segment_model.FEATURE_COUNT,'model_heads':['binary','kind'],
                   'model_note':'код линии 63 имеет две головы и 10 признаков; сохранена вся архитектура',
                   'features_source':'сохранённые OOF вероятности A только для пробного батча',
                   'source_encoder':'публичный models/rubert-tiny2, без архивов схемы 51',
                   'forward_backward_optimizer_step':True,'encoder_unfrozen':True,'optimizer':'AdamW',
                   'elapsed_seconds':elapsed,'peak_rss_bytes':peak,'available_bytes':available,
                   'host_total_bytes':total,'cgroup_adjusted_available_bytes':quota,
                   'peak_fraction':peak/available,'limit_fraction':0.8,'memory_pass':peak/available<=.8,
                   'loss_finite':True,'weights_saved':False,'test301':False})


def main():
    out=ROOT/'artifacts/memory66_step.json';out.parent.mkdir(exist_ok=True)
    result={'status':'падение','forward_backward_optimizer_step':False,'weights_saved':False,'test301':False}
    try: run(result)
    except Exception: result['error']=traceback.format_exc()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result.get('memory_pass') else 1

if __name__=='__main__': raise SystemExit(main())
