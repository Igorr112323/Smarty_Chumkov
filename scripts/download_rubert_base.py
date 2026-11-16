import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "weights/rubert-base-cased/curadoria"
TARGET = ROOT / "models/rubert-base-cased"
EXPECTED_SIZE = 714355318
FILES = ("config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_size(size):
    if abs(size - EXPECTED_SIZE) > EXPECTED_SIZE * 0.01:
        raise ValueError(f"неверный размер весов: {size}, ожидалось {EXPECTED_SIZE} ± 1%")


def download(download_fn, target=TARGET, source=SOURCE):
    target, source = Path(target), Path(source)
    for name in FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(source / name, target / name)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    error = None
    for attempt in range(1, 4):
        try:
            print(f"Загрузка DeepPavlov/rubert-base-cased, попытка {attempt}/3", flush=True)
            cached = Path(download_fn(repo_id="DeepPavlov/rubert-base-cased", filename="pytorch_model.bin"))
            validate_size(cached.stat().st_size)
            weight_hash = sha256(cached)
            temp = target / "pytorch_model.bin.tmp"
            shutil.copyfile(cached, temp)
            os.replace(temp, target / "pytorch_model.bin")
            break
        except Exception as exc:
            error = exc
            if attempt == 3:
                raise
            print(f"Попытка не удалась: {exc}", flush=True)
            time.sleep(5 * attempt)
    else:
        raise error
    size = (target / "pytorch_model.bin").stat().st_size
    manifest = {"repository": "DeepPavlov/rubert-base-cased", "file": "pytorch_model.bin", "source": "https://huggingface.co/DeepPavlov/rubert-base-cased/resolve/main/pytorch_model.bin", "size": size, "expected_size": EXPECTED_SIZE, "size_tolerance": 0.01, "sha256": weight_hash, "sha256_verified_against_official": False, "auxiliary_sha256": {name: sha256(target / name) for name in FILES}}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Размер: {size} байт; SHA-256: {weight_hash}", flush=True)
    print("Официальный эталон SHA-256 не передан; совпадение с ним не проверено", flush=True)
    return manifest


def main():
    from huggingface_hub import hf_hub_download

    download(hf_hub_download)


if __name__ == "__main__":
    main()
