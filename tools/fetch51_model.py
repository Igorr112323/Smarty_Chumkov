import hashlib
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT=Path(__file__).resolve().parent.parent

def main():
    name='cointegrated/rubert-tiny'
    revision=HfApi().model_info(name).sha
    if not revision:
        raise ValueError('нет закреплённой ревизии модели')
    snapshot=Path(snapshot_download(repo_id=name,revision=revision,allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','model.safetensors','pytorch_model.bin']))
    output=ROOT/'models/rubert-tiny'
    output.mkdir(parents=True,exist_ok=True)
    hashes={}
    for source in sorted(snapshot.iterdir()):
        if source.is_file():
            target=output/source.name
            shutil.copyfile(source,target)
            h=hashlib.sha256()
            with target.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            hashes[target.name]=h.hexdigest()
    if 'config.json' not in hashes or not ({'model.safetensors','pytorch_model.bin'} & set(hashes)):
        raise ValueError('нет конфигурации или весов')
    manifest={'repository':name,'revision':revision,'file_sha256':hashes,'model_only_not_dataset':True}
    (output/'source51.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
