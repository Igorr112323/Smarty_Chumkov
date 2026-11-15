import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/negatives/epoch_1")
    parser.add_argument("--token-threshold", type=float, default=0.25)
    parser.add_argument("--pair-gate", type=float, default=0.45)
    parser.add_argument("--out", default="artifacts/frozen_config.json")
    args = parser.parse_args(argv)

    folder = Path(args.checkpoint)
    digests = {name: file_sha256(folder / name) for name in FILES if (folder / name).exists()}
    combined = hashlib.sha256()
    for name in sorted(digests):
        combined.update(name.encode("utf-8"))
        combined.update(digests[name].encode("utf-8"))
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {
        "token_threshold": args.token_threshold,
        "pair_gate": args.pair_gate,
        "nms_iou": 0.10,
        "max_spans": 1,
        "min_chars": 2,
        "snap": "expand",
        "neighbor": None,
        "checkpoint": str(folder),
        "checkpoint_files": digests,
        "checkpoint_sha256": combined.hexdigest(),
        "seed": 0,
        "git_sha": git_sha,
        "decode": "decode.flags_for(pieces, token_threshold, None) -> spans_from_flags(min_chars=2) -> snap.expand",
        "selection": "max token F1 при FPR_pair <= 0.40 и FPR_token <= 0.40 на val150",
        "dataset": "artifacts/gold.jsonl, splits.json",
        "frozen_at": "2026-11-14",
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"конфигурация: {target}", flush=True)
    print(f"sha256 конфигурации: {file_sha256(target)}", flush=True)
    print(f"sha256 чекпоинта: {config['checkpoint_sha256']}", flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
