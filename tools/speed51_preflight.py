import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.nested_cv35 import probabilities
from tools.predict_frozen38 import load_state
from tools.speed_bench_batched import build_pairs


def main():
    import torch
    torch.set_num_threads(4)
    pairs = build_pairs(read_jsonl(ROOT / 'artifacts/ours33.jsonl'), 1000)
    if len(pairs) != 5 or any(len(p['document_text']) != 1000 for p in pairs):
        raise ValueError('не удалось подготовить 5 документов по 1000 символов')
    manifest = json.loads((ROOT / 'weights/final/manifest.json').read_text(encoding='utf-8'))
    if manifest['variant'] != 'v5' or not manifest['restore_verified']:
        raise ValueError('ожидается проверенный ансамбль V5')
    models = []
    for entry in manifest['checkpoints']:
        tokenizer, model = lexfeat.load_feature_model(ROOT / 'models/rubert-tiny2', seed=entry['seed'])
        model.base.config._attn_implementation = 'eager'
        load_state(torch, model, ROOT / 'weights/final' / f"seed_{entry['seed']}", entry)
        models.append((tokenizer, model))
    for seed in range(5):
        tokenizer, model = lexfeat.load_feature_model(ROOT / 'models/rubert-tiny', seed=seed)
        model.base.config._attn_implementation = 'eager'
        model.eval()
        models.append((tokenizer, model))
    def forward(selected):
        return [probabilities(tokenizer, model, pairs) for tokenizer, model in selected]
    forward(models[:1]); forward(models[5:6])
    data = {}
    for label, chosen in [('v5_only', models[:5]), ('two_architecture', models)]:
        repeats=[]
        for _ in range(3):
            start=time.perf_counter_ns()
            result=forward(chosen)
            elapsed=time.perf_counter_ns()-start
            if len(result)!=len(chosen) or any(len(c)!=len(pairs) for c in result):
                raise ValueError('неполные вероятности')
            repeats.append(elapsed / 1_000_000 / (sum(len(p['document_text']) for p in pairs) / 1000))
        data[label]={'models':len(chosen),'samples_ms_per_1000_document_chars':repeats,'median_ms_per_1000_document_chars':statistics.median(repeats)}
    paths=sorted(p for p in (ROOT / 'models/rubert-tiny').iterdir() if p.is_file())
    output={'status':'готово: предварительный стенд, не итоговое измерение обученного ансамбля', 'models':'пять восстановленных сидов В5 плюс пять seed-инициализаций rubert-tiny', 'warning':'руберт-tiny не дообучен; только время forward и подготовка признаков, усреднение/декодер не включены; оптимизированный инференс не измерен', 'data_scope':'пять train/val документов длиной 1000 символов, test 301 не оценивается', 'document_chars_total':5000, 'answer_chars_total':sum(len(p['model_output']) for p in pairs), 'threads':4,'batch':4, 'repeats':3, 'torch':str(torch.__version__), 'platform':platform.platform(), 'python':platform.python_version(), 'results':data, 'alternate_file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'speed_limit_ms_per_1000':30, 'package':'не измерен для этого ансамбля; существующий архив spanver не содержит интерфейс двухархитектурного ансамбля'}
    (ROOT / 'artifacts/speed51_preflight.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
