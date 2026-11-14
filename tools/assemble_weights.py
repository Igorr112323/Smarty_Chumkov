import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = "manifest.json"
OFFICIAL = {
    "rubert-tiny2": "137fa2b1d944dae19c74456dfe8fac2f780d9acf34e037f5d1e37acba1157768",
    "ruSciBERT": "4fe0aa4316b1fedf1e213ff8688266af3ea79f8fd99884945c0be9c5984b89c1",
}


class AssembleError(RuntimeError):
    pass


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(source_dir, destination, official=None):
    folder = Path(source_dir)
    manifest_path = folder / MANIFEST
    if not manifest_path.is_file():
        raise AssembleError(f"нет манифеста: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks") or []
    if not chunks or len(chunks) != manifest.get("chunk_count"):
        raise AssembleError("манифест повреждён: список кусков не совпадает с chunk_count")
    digest = hashlib.sha256()
    total = 0
    payload = []
    for entry in chunks:
        path = folder / entry["name"]
        if not path.is_file():
            raise AssembleError(f"нет куска: {path}")
        data = path.read_bytes()
        if len(data) != entry["bytes"]:
            raise AssembleError(f"размер куска {entry['name']} не совпал: {len(data)} != {entry['bytes']}")
        found = hashlib.sha256(data).hexdigest()
        if found != entry["sha256"]:
            raise AssembleError(f"sha256 куска {entry['name']} не совпал: {found} != {entry['sha256']}")
        digest.update(data)
        payload.append(data)
        total += len(data)
    if total != manifest.get("size"):
        raise AssembleError(f"итоговый размер не совпал: {total} != {manifest.get('size')}")
    full = digest.hexdigest()
    if full != manifest.get("full_sha256"):
        raise AssembleError(f"полный sha256 не совпал: {full} != {manifest.get('full_sha256')}")
    expected = official
    if expected is None:
        expected = OFFICIAL.get(Path(source_dir).name)
    if expected and full != expected:
        raise AssembleError(f"sha256 не совпал с официальным: {full} != {expected}")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        for data in payload:
            handle.write(data)
    print(f"assembled {target}")
    print(f"bytes {total}")
    print(f"sha256 {full}")
    print(f"official_match {bool(expected) and full == expected}")
    return {"path": str(target), "bytes": total, "sha256": full, "official_match": bool(expected) and full == expected}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("destination")
    parser.add_argument("--official", default=None)
    args = parser.parse_args(argv)
    try:
        assemble(args.source_dir, args.destination, args.official)
    except AssembleError as error:
        print(f"ошибка: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
