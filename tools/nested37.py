import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import lexfeat
from spanver.evaluate import read_jsonl
from tools.nested_cv35 import EPOCHS, GATE, averaged, choose, evaluate, partition, probabilities, verify_fast
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS
from tools.sweep_ragtruth30 import compute, prepare, within_fpr_limits
from tools.train_base33 import grouped_folds, train
from tools.train_scheme13_full import save_json


def fine_grid(center):
    value = round(center * 100)
    return [number / 100 for number in range(max(10, value - 5), min(95, value + 5) + 1)]


def fine_choose(inner, cache, center):
    prepared = prepare(inner, cache)
    rows = [compute(prepared, threshold, threshold / 2, GATE) for threshold in fine_grid(center)]
    if any(not row["support"]["negative_pairs"] or not row["support"]["negative_tokens"] for row in rows):
        raise ValueError("нет отрицательной поддержки для проверки FPR")
    eligible = [row for row in rows if within_fpr_limits(row)]
    if not eligible:
        return None, rows
    return max(eligible, key=lambda row: (row["token_f1_exact"], -row["start"])), rows


def checkpoint_digest(torch, model):
    with tempfile.TemporaryDirectory(prefix="nested37-") as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(model.state_dict(), path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size, "stored": False}


def compare_metrics(actual, reference):
    return {key: {"old": reference[key], "new": actual.get(key)} for key in reference if key not in actual or actual[key] != reference[key]}


def evaluate_at(pairs, cache, threshold):
    result = evaluate(pairs, cache, {row["pair_id"]: threshold for row in pairs})
    verify_fast(pairs, cache, threshold, result)
    return result


def blend(pairs, caches, count):
    return {row["pair_id"]: averaged(row, caches, range(count)) for row in pairs}


def main():
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    clock = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS or len({row["pair_id"] for row in pairs}) != 699:
        raise ValueError("не совпадают 699 train/val пар")
    folds = grouped_folds(pairs, 5)
    training, inner, outer, inner_index = partition(folds, args.fold - 1)
    prior = json.loads((ROOT / f"artifacts/nested36_fold_{args.fold}.json").read_text(encoding="utf-8"))
    digests = [id_digest(fold) for fold in folds]
    if prior["status"] != "готово" or prior["fold_ids_sha256"] != digests or prior["train_id_sha256"] != id_digest(training) or prior["inner_id_sha256"] != id_digest(inner) or prior["outer_id_sha256"] != id_digest(outer):
        raise ValueError("разбиение не совпало с предыдущим прогоном")
    report = {"status": "в процессе", "fold": args.fold, "inner_fold": inner_index + 1, "train_pairs": len(training), "inner_pairs": len(inner), "outer_pairs": len(outer), "train_id_sha256": id_digest(training), "inner_id_sha256": id_digest(inner), "outer_id_sha256": id_digest(outer), "fold_ids_sha256": digests, "git_sha": args.git_sha, "python": platform.python_version(), "torch": str(torch.__version__), "transformers": transformers.__version__, "seeds": [], "prior_run": 43899408927, "model_files_saved": False, "test301": False, "external_data": False, "pair_features_added": False, "fine_selection": "seed 0 на внутреннем фолде; максимум точного token F1 при обоих FPR <= 0.40, tie -> min start; без fallback"}
    target = ROOT / f"artifacts/nested37_fold_{args.fold}.json"
    save_json(target, report)
    outer_caches = []
    coarse = fine = None
    for seed in (0, 1, 2, 3, 4):
        started = time.perf_counter()
        print(f"фолд {args.fold}/5, seed {seed}/4: обучение {len(training)}, внутренняя {len(inner)}, внешняя {len(outer)}", flush=True)
        tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2", seed=seed)
        model.base.config._attn_implementation = "eager"
        torch.manual_seed(seed)
        history = train(torch, model, tokenizer, training, EPOCHS, seed * 1000)
        checkpoint = checkpoint_digest(torch, model)
        if seed == 0:
            inner_cache = probabilities(tokenizer, model, inner)
            selected, coarse_grid = choose(inner, inner_cache)
            if selected is None:
                raise ValueError("нет допустимой точки грубой сетки")
            coarse = selected["start"]
            selected_fine, fine_rows = fine_choose(inner, inner_cache, coarse)
            if selected_fine is None:
                raise ValueError("нет допустимой точки мелкой сетки")
            fine = selected_fine["start"]
            inner_metrics = evaluate_at(inner, inner_cache, fine)
            report["thresholds"] = {"coarse": coarse, "fine": fine, "fine_extend": fine / 2, "fine_inner": selected_fine, "fine_inner_metrics": inner_metrics, "fine_grid": fine_rows, "fine_eligible": sum(within_fpr_limits(row) for row in fine_rows), "prior_coarse": prior["threshold"]["start"], "coarse_reproduced": coarse == prior["threshold"]["start"]}
            del inner_cache
        cache = probabilities(tokenizer, model, outer)
        outer_caches.append(cache)
        actual = evaluate_at(outer, cache, coarse)
        row = {"seed": seed, "shuffle_seed": seed * 1000, "history": history, "outer_metrics_coarse": actual, "checkpoint": checkpoint, "seconds": round(time.perf_counter() - started, 3)}
        if seed < 3:
            row["prior_metric_differences"] = compare_metrics(actual, prior["seeds"][seed]["outer_metrics"])
        report["seeds"].append(row)
        report["elapsed_s"] = round(time.perf_counter() - clock, 3)
        save_json(target, report)
        print(f"фолд {args.fold}, seed {seed}: coarse={coarse}, fine={fine}, внешний token F1={actual['token_f1']}", flush=True)
        del model, tokenizer
    three = blend(outer, outer_caches, 3)
    five = blend(outer, outer_caches, 5)
    triple_metrics = evaluate_at(outer, three, coarse)
    report["results"] = {"seed0_coarse": report["seeds"][0]["outer_metrics_coarse"], "seed3_coarse": triple_metrics, "seed5_coarse": evaluate_at(outer, five, coarse), "seed5_fine": evaluate_at(outer, five, fine)}
    report["prior_ensemble_differences"] = compare_metrics(triple_metrics, prior["ensemble"]["outer_metrics"])
    report["prior_metric_reproduction"] = report["thresholds"]["coarse_reproduced"] and not report["prior_ensemble_differences"] and all(not item["prior_metric_differences"] for item in report["seeds"][:3])
    report["status"] = "готово"
    report["elapsed_s"] = round(time.perf_counter() - clock, 3)
    save_json(target, report)
    print(f"фолд {args.fold} готов: 3 сида {triple_metrics['token_f1']}; 5 сидов {report['results']['seed5_coarse']['token_f1']}; 5+сетка {report['results']['seed5_fine']['token_f1']}; воспроизведение {report['prior_metric_reproduction']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
