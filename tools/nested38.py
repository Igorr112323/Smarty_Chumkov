import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.nested_cv35 import EPOCHS, GATE, averaged, probabilities
from tools.preflight_13 import id_digest, our_keys
from tools.prepare_ours33 import EXPECTED_IDS
from tools.sweep_ragtruth30 import compute, prepare
from tools.train_base33 import grouped_folds, train
from tools.train_scheme13_full import save_json

GRID = tuple(number / 100 for number in range(10, 96))
PART_BYTES = 80 * 1024 * 1024


def split(folds, outer, inner):
    if outer == inner:
        raise ValueError("внешний и внутренний фолды должны различаться")
    training = [row for index, bucket in enumerate(folds) if index not in (outer, inner) for row in bucket]
    groups = (training, folds[inner], folds[outer])
    keys = [{key for row in group for key in our_keys(row)} for group in groups]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("пересечение документов")
    return groups


def target_inner(outer):
    return (outer + 1) % 5


def blend(pairs, caches):
    return {pair["pair_id"]: averaged(pair, caches, range(len(caches))) for pair in pairs}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_parts(torch, model, seed):
    folder = ROOT / "weights/frozen" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frozen38-") as directory:
        raw = Path(directory) / "checkpoint.pt"
        torch.save(model.state_dict(), raw)
        value = {"seed": seed, "filename": "checkpoint.pt", "size": raw.stat().st_size, "sha256": file_digest(raw), "chunk_bytes": PART_BYTES, "chunks": []}
        reconstructed = hashlib.sha256()
        with raw.open("rb") as handle:
            index = 0
            while True:
                data = handle.read(PART_BYTES)
                if not data:
                    break
                name = f"part-{index:03d}.bin"
                (folder / name).write_bytes(data)
                reconstructed.update(data)
                value["chunks"].append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                index += 1
        if reconstructed.hexdigest() != value["sha256"]:
            raise ValueError("ошибка сборки контрольных сумм checkpoint")
        restored = torch.load(raw, map_location="cpu", weights_only=True)
        model.load_state_dict(restored, strict=True)
        del restored
    save_json(folder / "manifest.json", value)
    return value


def main():
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--inner", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    path = ROOT / "artifacts/ours33.jsonl"
    gold_sha = file_digest(path)
    pairs = read_jsonl(path)
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS or len({row["pair_id"] for row in pairs}) != 699:
        raise ValueError("наши train/val не совпадают с 699 парами")
    folds = grouped_folds(pairs, 5)
    outer, inner = args.outer - 1, args.inner - 1
    training, tuning, held = split(folds, outer, inner)
    primary = inner == target_inner(outer)
    freeze = outer == 0 and primary
    report = {"status": "в процессе", "outer_fold": args.outer, "inner_fold": args.inner, "outer_primary": primary, "frozen": freeze, "train_pairs": len(training), "inner_pairs": len(tuning), "outer_pairs": len(held), "train_id_sha256": id_digest(training), "inner_id_sha256": id_digest(tuning), "outer_id_sha256": id_digest(held), "fold_ids_sha256": [id_digest(fold) for fold in folds], "gold_train_val_sha256": gold_sha, "git_sha": args.git_sha, "torch": str(torch.__version__), "transformers": transformers.__version__, "seeds": [], "external_data": False, "test301": False}
    target = ROOT / f"artifacts/nested38_outer_{args.outer}_inner_{args.inner}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_json(target, report)
    inner_caches = []
    outer_caches = []
    for seed in range(5):
        clock = time.perf_counter()
        print(f"внешний {args.outer}/5 внутренний {args.inner}/5 seed {seed}/4; train {len(training)}", flush=True)
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(seed)
        history = train(torch, model, tokenizer, training, EPOCHS, seed * 1000)
        saved = checkpoint_parts(torch, model, seed) if freeze else None
        inner_caches.append(probabilities(tokenizer, model, tuning))
        if primary:
            outer_caches.append(probabilities(tokenizer, model, held))
        report["seeds"].append({"seed": seed, "history": history, "checkpoint": saved, "seconds": round(time.perf_counter() - clock, 3)})
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(target, report)
        print(f"внешний {args.outer}, внутренний {args.inner}, seed {seed} готов", flush=True)
        del tokenizer, model
    cache = blend(tuning, inner_caches)
    prepared = prepare(tuning, cache)
    report["inner_grid"] = [compute(prepared, start, start / 2, GATE) for start in GRID]
    if any(not row["support"]["negative_pairs"] or not row["support"]["negative_tokens"] for row in report["inner_grid"]):
        raise ValueError("нет отрицательной поддержки FPR")
    if primary:
        averaged_outer = blend(held, outer_caches)
        report["outer_predictions"] = {key: value["pieces"] for key, value in averaged_outer.items()}
    if freeze:
        manifest = {"status": "веса собраны; калибровка не завершена", "outer_fold": args.outer, "inner_fold": args.inner, "train_id_sha256": report["train_id_sha256"], "gold_train_val_sha256": gold_sha, "git_sha": args.git_sha, "checkpoints": [seed["checkpoint"] for seed in report["seeds"]]}
        save_json(ROOT / "weights/frozen/manifest.json", manifest)
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    save_json(target, report)
    print(f"внешний {args.outer}, внутренний {args.inner}: завершено за {report['elapsed_s']} с", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
