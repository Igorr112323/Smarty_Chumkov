import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT = 80 * 1024 * 1024


def show(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v4", "v5"), required=True)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    checks = []
    for fold in range(1, 6):
        folder = f"weights/ablation47/{args.variant}/fold{fold}"
        manifest = json.loads(show(args.revision, f"{folder}/manifest.json"))
        if manifest["variant"] != args.variant or manifest["outer_fold"] != fold or manifest["status"] != "готово" or not manifest["restore_verified"] or [entry["seed"] for entry in manifest["checkpoints"]] != [0, 1, 2, 3, 4]:
            raise ValueError("неверный манифест")
        for entry in manifest["checkpoints"]:
            whole = hashlib.sha256()
            total = 0
            for part in entry["chunks"]:
                path = f"{folder}/seed_{entry['seed']}/{part['name']}"
                process = subprocess.Popen(["git", "show", f"{args.revision}:{path}"], cwd=ROOT, stdout=subprocess.PIPE)
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
                    digest.update(block)
                    whole.update(block)
                    size += len(block)
                process.stdout.close()
                if process.wait() != 0 or size != part["bytes"] or size > LIMIT or digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"неверная часть {path}")
                total += size
            if total != entry["size"] or whole.hexdigest() != entry["sha256"] or not entry["restore_predictions_equal"]:
                raise ValueError(f"неверный checkpoint {fold}/{entry['seed']}")
            checks.append({"fold": fold, "seed": entry["seed"], "sha256": whole.hexdigest(), "size": total, "restored_predictions_checked_in_training": True})
        print(f"{args.variant} фолд {fold}: 5 checkpoint проверены", flush=True)
    result = {"variant": args.variant, "revision": args.revision, "checkpoint_count": len(checks), "all_parts_sha256_ok": True, "checkpoints": checks}
    target = ROOT / f"artifacts/ablation47_{args.variant}_weights_check.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
