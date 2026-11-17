import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT = 80 * 1024 * 1024


def check(variant):
    checks = []
    for fold in range(1, 6):
        folder = ROOT / "weights/ablation46" / variant / f"fold{fold}"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["variant"] != variant or manifest["outer_fold"] != fold or manifest["status"] != "готово" or not manifest["restore_verified"] or [entry["seed"] for entry in manifest["checkpoints"]] != [0, 1]:
            raise ValueError("неверный манифест ансамбля")
        for entry in manifest["checkpoints"]:
            digest = hashlib.sha256()
            total = 0
            for part in entry["chunks"]:
                path = folder / f"seed_{entry['seed']}" / part["name"]
                size = path.stat().st_size
                if size != part["bytes"] or size > LIMIT:
                    raise ValueError(f"неверный размер {path}")
                part_digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        part_digest.update(block)
                        digest.update(block)
                        total += len(block)
                if part_digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"неверный SHA части {path}")
            if total != entry["size"] or digest.hexdigest() != entry["sha256"] or not entry["restore_predictions_equal"]:
                raise ValueError(f"неверный checkpoint {fold}/{entry['seed']}")
            checks.append({"fold": fold, "seed": entry["seed"], "sha256": digest.hexdigest(), "size": total, "restored_predictions_checked_in_training": True})
    return {"variant": variant, "checkpoint_count": len(checks), "all_parts_sha256_ok": True, "checkpoints": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("control", "v1", "v2", "v3"))
    args = parser.parse_args()
    result = check(args.variant)
    target = ROOT / f"artifacts/ablation46_{args.variant}_weights_check.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.variant}: SHA проверены для {result['checkpoint_count']} checkpoint", flush=True)


if __name__ == "__main__":
    main()
