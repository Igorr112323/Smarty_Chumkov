import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import transformers

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.nested38 import GRID, PART_BYTES, file_digest, split, target_inner
from tools.nested_cv35 import GATE, averaged, probabilities
from tools.predict_frozen38 import load_state
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.sweep_ragtruth30 import compute, prepare
from tools.train_base33 import grouped_folds
from tools.train46 import train_variant as original_train_variant
from tools.train_scheme13_full import save_json

SEEDS = (0, 1, 2, 3, 4)


def save_checkpoint(model, variant, outer, seed):
    folder = ROOT / "weights/ablation47" / variant / f"fold{outer}" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weights47-") as directory:
        raw = Path(directory) / "checkpoint.pt"
        torch.save(model.state_dict(), raw)
        entry = {"seed": seed, "filename": "checkpoint.pt", "size": raw.stat().st_size, "sha256": file_digest(raw), "chunk_bytes": PART_BYTES, "chunks": []}
        with raw.open("rb") as handle:
            index = 0
            while True:
                data = handle.read(PART_BYTES)
                if not data:
                    break
                name = f"part-{index:03d}.bin"
                (folder / name).write_bytes(data)
                entry["chunks"].append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                index += 1
    save_json(folder / "manifest.json", entry)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("v4", "v5"))
    parser.add_argument("--outer", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--inner", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    source = ROOT / "artifacts/ours33.jsonl"
    summary = json.loads((ROOT / "artifacts/nested38_summary.json").read_text(encoding="utf-8"))
    pairs = read_jsonl(source)
    if file_digest(source) != summary["gold_train_val_sha256"] or len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS:
        raise ValueError("неверный набор train/val")
    folds = grouped_folds(pairs, 5)
    if [id_digest(fold) for fold in folds] != summary["fold_ids_sha256"]:
        raise ValueError("изменено разбиение")
    outer, inner = args.outer - 1, args.inner - 1
    training, tuning, held = split(folds, outer, inner)
    primary = inner == target_inner(outer)
    blind = [{key: value for key, value in row.items() if key not in ("spans", "label")} for row in held]
    path = ROOT / f"artifacts/ablation47_{args.variant}_outer{args.outer}_inner{args.inner}.json"
    report = {"status": "в процессе", "variant": args.variant, "outer_fold": args.outer, "inner_fold": args.inner, "primary": primary, "git_sha": args.git_sha, "torch": str(torch.__version__), "transformers": transformers.__version__, "gold_sha256": file_digest(source), "fold_ids_sha256": summary["fold_ids_sha256"], "train_id_sha256": id_digest(training), "inner_id_sha256": id_digest(tuning), "outer_id_sha256": id_digest(held), "seeds": [], "test301": False, "external_data": False}
    save_json(path, report)
    inner_caches, outer_caches = [], []
    reference_path = ROOT / f"artifacts/ablation46_v2_outer{args.outer}_inner{args.inner}.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if args.variant == "v4" else None
    started = time.perf_counter()
    for seed in SEEDS:
        clock = time.perf_counter()
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(seed)
        info = original_train_variant(model, tokenizer, training, tuning, seed, "v2", permanent_freeze=args.variant == "v5")
        if reference is not None and seed in (0, 1):
            original_seed = reference["seeds"][seed]
            old_history = original_seed["history"]
            new_history = info["history"]
            errors = [abs(old[name] - new[name]) for old, new in zip(old_history, new_history) for name in ("train_loss", "inner_loss")]
            info["paired_v2"] = {"epochs_match": info["epochs_ran"] == original_seed["epochs_ran"], "best_epoch_match": info["best_epoch"] == original_seed["best_epoch"], "max_loss_abs_diff": max(errors, default=0.0), "torch_match": str(torch.__version__) == reference["torch"], "transformers_match": transformers.__version__ == reference["transformers"]}
        inner_caches.append(probabilities(tokenizer, model, tuning))
        checkpoint = None
        if primary:
            original = probabilities(tokenizer, model, blind)
            checkpoint = save_checkpoint(model, args.variant, args.outer, seed)
            del model, tokenizer
            tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
            model.base.config._attn_implementation = "eager"
            load_state(torch, model, ROOT / "weights/ablation47" / args.variant / f"fold{args.outer}" / f"seed_{seed}", checkpoint)
            restored = probabilities(tokenizer, model, blind)
            if restored != original:
                raise ValueError("вероятности после восстановления не совпали")
            checkpoint["restore_predictions_equal"] = True
            outer_caches.append(restored)
        info.update({"seed": seed, "seconds": round(time.perf_counter() - clock, 3), "checkpoint": checkpoint})
        report["seeds"].append(info)
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(path, report)
        print(f"{args.variant} outer {args.outer} inner {args.inner} seed {seed}: готов", flush=True)
        del model, tokenizer
    inner_cache = {row["pair_id"]: averaged(row, inner_caches, range(len(SEEDS))) for row in tuning}
    prepared = prepare(tuning, inner_cache)
    report["inner_grid"] = [compute(prepared, threshold, threshold / 2, GATE) for threshold in GRID]
    if primary:
        outer_cache = {row["pair_id"]: averaged(row, outer_caches, range(len(SEEDS))) for row in blind}
        report["outer_predictions"] = {key: value["pieces"] for key, value in outer_cache.items()}
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    save_json(path, report)
    print(f"{args.variant} outer {args.outer} inner {args.inner}: завершено", flush=True)


if __name__ == "__main__":
    main()
