import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def show(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v4", "v5"), required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prefix = f"weights/ablation47/{args.variant}/fold1"
    manifest = json.loads(show(args.revision, f"{prefix}/manifest.json"))
    if manifest["variant"] != args.variant or manifest["outer_fold"] != 1 or manifest["status"] != "готово" or not manifest["restore_verified"] or [entry["seed"] for entry in manifest["checkpoints"]] != [0, 1, 2, 3, 4]:
        raise ValueError("неверный исходный манифест")
    destination = ROOT / "weights/final"
    with tempfile.TemporaryDirectory(prefix="promote47-", dir=ROOT) as temp:
        staging = Path(temp)
        for entry in manifest["checkpoints"]:
            seed = entry["seed"]
            folder = staging / f"seed_{seed}"
            folder.mkdir()
            digest = hashlib.sha256()
            size = 0
            for chunk in entry["chunks"]:
                target = folder / chunk["name"]
                with target.open("wb") as handle:
                    subprocess.run(["git", "show", f"{args.revision}:{prefix}/seed_{seed}/{chunk['name']}"], cwd=ROOT, stdout=handle, check=True)
                chunk_hash = hashlib.sha256()
                with target.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        chunk_hash.update(block)
                        digest.update(block)
                        size += len(block)
                if target.stat().st_size != chunk["bytes"] or target.stat().st_size > 80 * 1024 * 1024 or chunk_hash.hexdigest() != chunk["sha256"]:
                    raise ValueError(f"повреждена часть seed {seed}")
            if size != entry["size"] or digest.hexdigest() != entry["sha256"] or not entry["restore_predictions_equal"]:
                raise ValueError(f"повреждён checkpoint seed {seed}")
            (folder / "manifest.json").write_bytes(show(args.revision, f"{prefix}/seed_{seed}/manifest.json"))
        manifest.update({"status": "кандидат", "restore_verified": False, "source": prefix, "source_revision": args.revision, "scope": "пять сидов только внешнего фолда 1; остальные четыре ансамбля находятся в weights/ablation47", "fpr_note": "порог выбран по внутренним фолдам; FPR на новых данных не гарантирован"})
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for source in sorted(staging.rglob("*")):
            if source.is_file():
                target = destination / source.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
    print(f"кандидат {args.variant}, фолд 1: части и SHA сохранены в weights/final; нужна отдельная проверка инференса")


if __name__ == "__main__":
    main()
