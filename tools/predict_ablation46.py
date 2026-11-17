import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.nested_cv35 import averaged, predict, probabilities
from tools.predict_frozen38 import load_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", default="models/rubert-tiny2")
    args = parser.parse_args()
    folder = Path(args.weights)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "готово" or not manifest.get("restore_verified") or not manifest.get("checkpoints"):
        raise ValueError("неполный манифест ансамбля")
    pairs = read_jsonl(Path(args.input))
    if not pairs or any(not row.get("pair_id") or not row.get("model_output") for row in pairs) or len({row["pair_id"] for row in pairs}) != len(pairs):
        raise ValueError("нужны непустые пары с уникальными pair_id и model_output")
    caches = []
    for entry in manifest["checkpoints"]:
        seed = entry["seed"]
        tokenizer, model = lexfeat.load_feature_model(args.base, seed=seed)
        model.base.config._attn_implementation = "eager"
        load_state(torch, model, folder / f"seed_{seed}", entry)
        caches.append(probabilities(tokenizer, model, pairs))
        del tokenizer, model
    output = []
    for row in pairs:
        value = averaged(row, caches, range(len(caches)))
        output.append({"pair_id": row["pair_id"], "spans": predict(value, manifest["threshold"]["start"])})
    Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(f"готово: {len(output)} пар; выход {args.output}")


if __name__ == "__main__":
    main()
