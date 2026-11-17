import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.blend51 import blend
from tools.nested_cv35 import averaged,probabilities,predict
from tools.predict_frozen38 import load_state
from tools.speed_bench_batched import build_pairs


def main():
    import torch
    torch.set_num_threads(4)
    pairs=build_pairs(read_jsonl(ROOT/'artifacts/ours33.jsonl'),1000)
    if len(pairs)!=5 or any(len(p['document_text'])!=1000 for p in pairs):raise ValueError('неверные документы')
    models=[]
    for arch in ('tiny2','tiny'):
        group=[]
        for seed in range(5):
            folder=ROOT/'weights/ablation51/dual/fold1'/arch/f'seed_{seed}'
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if manifest['seed']!=seed or manifest['architecture']!=arch:raise ValueError('неверный сид')
            tokenizer,model=lexfeat.load_feature_model(ROOT/'models'/('rubert-tiny2' if arch=='tiny2' else 'rubert-tiny'),seed=seed)
            model.base.config._attn_implementation='eager'
            load_state(torch,model,folder,manifest)
            group.append((tokenizer,model))
        models.append(group)
    def inference():
        caches=[[probabilities(tok,model,pairs) for tok,model in group] for group in models]
        mixed=[blend(averaged(pair,caches[0],range(5)),averaged(pair,caches[1],range(5))) for pair in pairs]
        return [predict(value,0.5) for value in mixed]
    inference()
    values=[]
    for _ in range(3):
        start=time.perf_counter_ns()
        predictions=inference()
        if len(predictions)!=5:raise ValueError('неполный вывод')
        elapsed=time.perf_counter_ns()-start
        values.append(elapsed/1_000_000/(sum(len(p['document_text']) for p in pairs)/1000))
    report={'status':'готово','scope':'пять train/val документов по 1000 знаков, test301 не оценивался', 'architecture':'tiny2 и rubert-tiny, по пять обученных checkpoint фолда 1','source_run':44077009850,'samples_ms_per_1000_document_chars':values,'median_ms_per_1000_document_chars':statistics.median(values),'limit':30,'passes_limit':statistics.median(values)<=30,'details':'CPU, 4 потока; batch 4; 1 прогрев + 3 измерения, включает подготовку признаков, 10 forward, проекцию, усреднение и клаузный expand; порог 0.5 только для замера, без оценки качества; калибровка в замер не включена','answer_chars_total':sum(len(p['model_output']) for p in pairs),'torch':str(torch.__version__),'platform':platform.platform()}
    (ROOT/'artifacts/speed51_actual.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
