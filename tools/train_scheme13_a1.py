import argparse
import gzip
import io
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.preflight_13 import components, digest, id_digest, rag_keys, split_groups


def read_pairs():
    rows = []
    for name in ("ragtruth_ru.jsonl", "ragtruth_ru_negative.jsonl"):
        with (ROOT / "Dataset/RAGTruth_ru" / name).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    folds = split_groups(components(rows, rag_keys), 5)
    return [row for fold in folds[1:] for row in fold], folds[0]


def evaluate(tokenizer, model, pairs, block=16):
    from spanver import decode, lexfeat
    from spanver.evaluate import metrics
    from spanver.snap import snapped
    from tools.train_cv_feat import token_fpr

    cache = {}
    for start in range(0, len(pairs), block):
        cache.update(lexfeat.batched_probabilities(tokenizer, model, pairs[start:start + block], top=4, batch=4))
        if (start // block + 1) % 10 == 0:
            print(f"оценка {min(start + block, len(pairs))}/{len(pairs)}", flush=True)

    def predictor(pair):
        value = cache.get(pair["pair_id"])
        if not value or max((piece[2] for piece in value["pieces"]), default=0.0) < 0.45:
            return []
        flags = decode.flags_for(value["pieces"], 0.25, None)
        spans = decode.spans_from_flags(value["answer"], value["pieces"], flags, 2)
        return snapped(spans, value["answer"], "expand")

    result = metrics(pairs, predictor)
    return {"pairs": len(pairs), "token": result["token_level"], "span": result["span_level"], "pair": result["pair_level"], "fpr_pair": result["fpr"], "fpr_token": token_fpr(pairs, predictor)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, required=True)
    parser.add_argument("--train-eval-limit", type=int, default=256)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/stage_a_e1")
    parser.add_argument("--report", default="artifacts/scheme13_a1.json")
    args = parser.parse_args()
    import torch
    from spanver import lexfeat
    from spanver.stage_train import ENCODER_RATE, HEAD_RATE, stage_a_plans

    started = time.perf_counter()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    training, heldout = read_pairs()
    if len(training) != 6042 or len(heldout) != 1458:
        raise ValueError("разбиение не совпадает с принятым preflight")
    if not 1 <= args.train_limit <= len(training):
        raise ValueError("неверный объём обучения")
    expected = json.loads((ROOT / "artifacts/preflight_13.json").read_text(encoding="utf-8"))
    if id_digest(training) != expected["rag_train"]["id_sha256"] or id_digest(heldout) != expected["rag_holdout"]["id_sha256"]:
        raise ValueError("разбиение отличается от preflight")
    selected = random.Random(0).sample(training, args.train_limit)
    train_eval = random.Random(1).sample(selected, min(len(selected), args.train_eval_limit))
    print(f"обучение {len(selected)}/{len(training)}, holdout {len(heldout)}", flush=True)
    tokenizer, model = lexfeat.load_feature_model(ROOT / "models/rubert-tiny2")
    model.base.config._attn_implementation = "eager"
    torch.set_num_threads(2)
    target = ROOT / args.checkpoint
    if args.evaluate_only:
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        payload = target / "state_dict_fp16.pt.gz"
        if digest(payload) != manifest["checkpoint_sha256"] or manifest["train_id_sha256"] != id_digest(selected) or manifest["holdout_id_sha256"] != id_digest(heldout):
            raise ValueError("чекпоинт или разбиение не совпадают с манифестом")
        with gzip.open(payload, "rb") as handle:
            saved = torch.load(io.BytesIO(handle.read()), map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        del saved
    else:
        if target.exists():
            raise FileExistsError(f"чекпоинт уже существует: {target}")
        table = lexfeat.build_table(tokenizer, selected)
        if len(table) != len(selected):
            raise ValueError("не все тренировочные пары преобразованы")
        plans = stage_a_plans(list(table), 1, size=16, seed=0)
        print(f"таблица {len(table)}, шагов {len(plans[0])}", flush=True)
        epoch_started = time.perf_counter()
        history = lexfeat.train(torch, model, table, plans, HEAD_RATE, ENCODER_RATE, tokenizer.pad_token_id, focal_gamma=2.0)
        epoch_seconds = time.perf_counter() - epoch_started
        del table, plans
        target.mkdir(parents=True, exist_ok=False)
        half_state = {key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu() for key, value in model.state_dict().items()}
        buffer = io.BytesIO()
        torch.save({"state_dict": half_state}, buffer)
        payload = target / "state_dict_fp16.pt.gz"
        with payload.open("wb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as output:
                output.write(buffer.getbuffer())
        del buffer
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest = {
            "stage": "a_e1", "seed": 0, "epochs": 1, "git_sha": git_sha,
            "training_code_sha256": digest(Path(__file__)),
            "preflight_code_sha256": digest(ROOT / "tools/preflight_13.py"),
            "data_sha256": {name: digest(ROOT / "Dataset/RAGTruth_ru" / name) for name in ("ragtruth_ru.jsonl", "ragtruth_ru_negative.jsonl")},
            "base_weights_sha256": digest(ROOT / "models/rubert-tiny2/pytorch_model.bin"),
            "train_available": len(training), "train_used": len(selected), "train_id_sha256": id_digest(selected),
            "train_positive": sum(bool(row["spans"]) for row in selected),
            "train_negative": sum(not row["spans"] for row in selected),
            "holdout": len(heldout), "holdout_id_sha256": id_digest(heldout),
            "recipe": f"python tools/train_scheme13_a1.py --train-limit {args.train_limit} --train-eval-limit {args.train_eval_limit}",
            "config": {"feature_count": 5, "focal_gamma": 2.0, "encoder_rate": ENCODER_RATE, "head_rate": HEAD_RATE, "micro_batch": 4, "accumulate": 4, "seed": 0, "threads": 2, "torch": torch.__version__, "scheduler": "one-epoch linear warmup/decay", "resume": "fp16 model weights; optimizer and scheduler restart on continuation"},
            "checkpoint_sha256": digest(payload), "checkpoint_bytes": payload.stat().st_size,
            "epoch_seconds": round(epoch_seconds, 3), "history": history,
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"чекпоинт сохранён {payload} sha256 {manifest['checkpoint_sha256']}", flush=True)
        model.load_state_dict(half_state)
        del half_state
    model.eval()
    report = {"status": "оценка", "checkpoint": str(target.relative_to(ROOT)), "training": manifest, "train_eval_id_sha256": id_digest(train_eval)}
    report_path = ROOT / args.report
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("оценка тренировочной выборки", flush=True)
    report["train_sample"] = evaluate(tokenizer, model, train_eval)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("оценка holdout", flush=True)
    report["holdout"] = evaluate(tokenizer, model, heldout)
    report["gap_train_minus_holdout"] = {name: round(report["train_sample"][name]["f1"] - report["holdout"][name]["f1"], 4) for name in ("token", "span", "pair")}
    report["status"] = "готово" if len(selected) == len(training) else "готово на сокращённой обучающей выборке"
    report["total_seconds"] = round(time.perf_counter() - started, 3)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "train_sample", "holdout", "gap_train_minus_holdout", "total_seconds")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
