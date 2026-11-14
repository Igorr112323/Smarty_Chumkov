import argparse
import hashlib
import json
from pathlib import Path

CHUNK_BYTES = 83886080
MANIFEST = "manifest.json"


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split(source, out_dir, chunk_bytes=CHUNK_BYTES, quiet=False):
    src = Path(source)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    chunks = []
    full = hashlib.sha256()
    with open(src, "rb") as handle:
        index = 0
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            full.update(block)
            name = f"part-{index:03d}.bin"
            path = target / name
            path.write_bytes(block)
            digest = hashlib.sha256(block).hexdigest()
            chunks.append({"name": name, "bytes": len(block), "sha256": digest})
            if not quiet:
                print(f"{name} {len(block)} {digest}")
            index += 1
    manifest = {
        "source": src.name,
        "full_sha256": full.hexdigest(),
        "size": size,
        "chunk_bytes": chunk_bytes,
        "chunk_count": len(chunks),
        "chunk_sha256": [entry["sha256"] for entry in chunks],
        "chunks": chunks,
    }
    (target / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not quiet:
        print(f"full sha256 {manifest['full_sha256']}")
        print(f"chunks {manifest['chunk_count']}")
        print(f"size {size}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("out_dir")
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    args = parser.parse_args(argv)
    split(args.source, args.out_dir, args.chunk_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
