import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import transformers

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.nested38 import file_digest, split, target_inner
from tools.nested_cv35 import EPOCHS, averaged, evaluate, probabilities, verify_fast
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.train_base33 import grouped_folds, train
from tools.train_scheme13_full import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    summary = json.loads((ROOT / "artifacts/nested38_summary.json").read_text(encoding="utf-8"))
    path = ROOT / "artifacts/ours33.jsonl"
    pairs = read_jsonl(path)
    if file_digest(path) != summary["gold_train_val_sha256"] or len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS:
        raise ValueError("неверный train/val")
    folds = grouped_folds(pairs, 5)
    if [id_digest(fold) for fold in folds] != summary["fold_ids_sha256"]:
        raise ValueError("разбиение изменилось")
    outer = args.fold - 1
    inner = target_inner(outer)
    training, tuning, held = split(folds, outer, inner)
    reference = summary["folds"][outer]
    blind = [{key: value for key, value in row.items() if key not in ("spans", "label")} for row in held]
    path = ROOT / f"artifacts/reproduce45_fold{args.fold}.json"
    report = {"status": "в процессе", "outer_fold": args.fold, "inner_fold": inner + 1, "gold_train_val_sha256": summary["gold_train_val_sha256"], "train_id_sha256": id_digest(training), "inner_id_sha256": id_digest(tuning), "outer_id_sha256": id_digest(held), "git_sha": args.git_sha, "source_git_sha": summary["git_sha"], "torch": str(torch.__version__), "transformers": transformers.__version__, "threshold": reference["threshold"], "seeds": [], "test301": False, "external_data": False}
    save_json(path, report)
    caches = []
    started = time.perf_counter()
    for seed in range(5):
        clock = time.perf_counter()
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(seed)
        history = train(torch, model, tokenizer, training, EPOCHS, seed * 1000)
        caches.append(probabilities(tokenizer, model, blind))
        report["seeds"].append({"seed": seed, "history": history, "seconds": round(time.perf_counter() - clock, 3)})
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        save_json(path, report)
        print(f"фолд {args.fold}, seed {seed}: готов", flush=True)
        del tokenizer, model
    cache = {row["pair_id"]: averaged(row, caches, range(5)) for row in blind}
    threshold = reference["threshold"]["start"]
    actual = evaluate(held, cache, {row["pair_id"]: threshold for row in held})
    verify_fast(held, cache, threshold, actual)
    report.update({"status": "готово", "metrics": actual, "expected_metrics": reference["metrics"], "metrics_equal": actual == reference["metrics"], "elapsed_s": round(time.perf_counter() - started, 3)})
    save_json(path, report)
    print(json.dumps({"fold": args.fold, "metrics_equal": report["metrics_equal"], "actual": actual, "expected": reference["metrics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
