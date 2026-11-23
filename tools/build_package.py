import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
LIMIT_BYTES = 614400
README = """spanver {version}

Локальная проверка ответа модели против документа.

Установка: распаковать архив рядом с рабочим каталогом, Python 3.10+.

Лексический режим (по умолчанию, только стандартная библиотека, без torch):
  python -m spanver verify --answer answer.txt --document doc.txt
  python -m spanver serve --port 8765
  страница: http://127.0.0.1:8765/

Модельный режим (нужны torch, transformers и каталог весов rubert-tiny2):
  python -m spanver verify --backend model --weights models\\rubert-tiny2 --answer a.txt --document d.txt
  python -m spanver serve --port 8765 --backend model --weights models\\rubert-tiny2
  головка и порог берутся из artifacts/model_head.json (создаёт tools/run_model_eval.py)

Tiny V5 seed 0 (веса и зависимости поставляются отдельно):
  python -m spanver verify --backend tiny --answer a.txt --document d.txt
Нужны models/rubert-tiny2, artifacts/tiny_head.json, проверенный манифест
weights/final/manifest.json и части checkpoint в weights/final/seed_0/.
При отсутствии или несовпадении контрольных сумм tiny не запускается.

Сеть и GPU не используются. Канонический сплит: spanver/splits.json.
"""


def version():
    text = (ROOT / "spanver" / "__init__.py").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


def collect():
    files = []
    for pattern in ("**/*.py", "**/*.json"):
        for path in sorted((ROOT / "spanver").glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def build(out="artifacts"):
    target_dir = Path(out)
    target_dir.mkdir(parents=True, exist_ok=True)
    name = f"spanver-{version()}.zip"
    target = target_dir / name
    files = collect()
    payload = README.format(version=version()).encode("utf-8")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            arcname = f"spanver/{path.relative_to(ROOT / 'spanver').as_posix()}"
            info = zipfile.ZipInfo(arcname, date_time=FIXED_TIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("README.txt", date_time=FIXED_TIME)
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, payload)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    size = target.stat().st_size
    summary = {
        "package": name,
        "bytes": size,
        "size_kb": round(size / 1024.0, 1),
        "limit_bytes": LIMIT_BYTES,
        "under_limit": size <= LIMIT_BYTES,
        "sha256": digest,
        "files": [f"spanver/{path.relative_to(ROOT / 'spanver').as_posix()}" for path in files] + ["README.txt"],
        "python_requires": ">=3.10",
        "version": version(),
    }
    (target_dir / "package.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"package {name}")
    print(f"bytes {size}")
    print(f"size_kb {summary['size_kb']}")
    print(f"under_limit {summary['under_limit']}")
    print(f"sha256 {digest}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
