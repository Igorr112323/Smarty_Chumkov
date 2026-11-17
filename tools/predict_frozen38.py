import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.nested_cv35 import averaged, predict, probabilities
from tools.nested38 import file_digest


def load_state(torch, model, folder, manifest):
    with tempfile.TemporaryDirectory(prefix="predict38-") as directory:
        destination = Path(directory) / "checkpoint.pt"
        digest = hashlib.sha256()
        total = 0
        with destination.open("wb") as output:
            for part in manifest["chunks"]:
                path = folder / part["name"]
                if path.stat().st_size != part["bytes"] or file_digest(path) != part["sha256"]:
                    raise ValueError("контрольная сумма части checkpoint не совпала")
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        total += len(block)
                        output.write(block)
        if total != manifest["size"] or digest.hexdigest() != manifest["sha256"]:
            raise ValueError("контрольная сумма checkpoint не совпала")
        state = torch.load(destination, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        del state


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="weights/frozen")
    parser.add_argument("--base", default="models/rubert-tiny2")
    args = parser.parse_args()
    folder = Path(args.weights)
    config = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if config.get("status") not in ("кандидат", "готово") or len(config.get("checkpoints", [])) != 5:
        raise ValueError("неполный манифест модели")
    pairs = read_jsonl(Path(args.input))
    if not pairs or any(not pair.get("pair_id") or not pair.get("model_output") for pair in pairs) or len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("нужны непустые пары с уникальными pair_id и model_output")
    caches = []
    for entry in config["checkpoints"]:
        seed = entry["seed"]
        tokenizer, model = lexfeat.load_feature_model(args.base, seed=seed)
        model.base.config._attn_implementation = "eager"
        load_state(torch, model, folder / f"seed_{seed}", entry)
        caches.append(probabilities(tokenizer, model, pairs))
        del model, tokenizer
    rows = []
    for pair in pairs:
        cache = averaged(pair, caches, range(5))
        rows.append({"pair_id": pair["pair_id"], "spans": predict(cache, config["threshold"]["start"])})
    Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"готово: {len(rows)} пар; выход {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
