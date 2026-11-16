import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from spanver import decode, lexfeat
from spanver.evaluate import metrics, read_jsonl
from spanver.snap import snapped
from spanver.stage_train import ENCODER_RATE, HEAD_RATE
from tools.preflight_13 import id_digest
from tools.sweep_hyst import flags_with
from tools.sweep_ragtruth30 import compute, percentile, prepare, within_fpr_limits
from tools.train_base33 import BATCH, GATE, GRID, grouped_folds, tune
from tools.train_cv_feat import token_fpr


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def probability_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None, "quantiles": {}}
    return {"count": len(values), "min": values[0], "median": statistics.median(values), "max": values[-1], "quantiles": {str(q): percentile(values, q) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}}


def count_changes(model, saved):
    totals = {}
    for name, parameter in model.named_parameters():
        if name.startswith("base.bert.encoder."):
            group = "encoder_layers"
        elif name.startswith("base.bert.embeddings."):
            group = "embeddings"
        elif name.startswith("proj") or "classifier" in name:
            group = "head"
        else:
            group = "other"
        record = totals.setdefault(group, {"parameters": 0, "trainable_parameters": 0, "changed_elements": 0, "changed_tensors": 0, "tensor_count": 0, "max_abs_delta": 0.0})
        record["parameters"] += parameter.numel()
        record["trainable_parameters"] += parameter.numel() if parameter.requires_grad else 0
        if name not in saved or saved[name].shape != parameter.shape:
            raise ValueError(f"параметр не совпадает с исходным: {name}")
        delta = (saved[name].detach().cpu() - parameter.detach().cpu()).abs()
        changed = int(torch.count_nonzero(delta))
        record["changed_elements"] += changed
        record["changed_tensors"] += int(changed > 0)
        record["tensor_count"] += 1
        record["max_abs_delta"] = max(record["max_abs_delta"], float(delta.max()) if delta.numel() else 0.0)
        del delta
    return totals


def schedule(train_count, epochs):
    steps_epoch = math.ceil(train_count / BATCH)
    total = steps_epoch * epochs
    warmup = max(1, int(total * 0.1))
    def factor(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (total - step) / max(1, total - warmup))
    rows = []
    for epoch in range(epochs):
        values = [factor(step) for step in range(epoch * steps_epoch, (epoch + 1) * steps_epoch)]
        rows.append({"epoch": epoch + 1, "encoder_lr_first": ENCODER_RATE * values[0], "encoder_lr_last": ENCODER_RATE * values[-1], "encoder_lr_min": ENCODER_RATE * min(values), "encoder_lr_max": ENCODER_RATE * max(values), "head_lr_first": HEAD_RATE * values[0], "head_lr_last": HEAD_RATE * values[-1], "head_lr_min": HEAD_RATE * min(values), "head_lr_max": HEAD_RATE * max(values)})
    return {"basis": "восстановлено по исходному коду планировщика и фактическим размерам фолда; журнал фактических lr не сохранялся", "steps_per_epoch": steps_epoch, "total_steps": total, "warmup_steps": warmup, "epochs": rows}


def predict(value, start, extend, gate=True):
    pieces = value["pieces"]
    if gate and max((p[2] for p in pieces), default=0.0) < GATE:
        return []
    flags = flags_with(pieces, start, extend, 0.0)
    return snapped(decode.spans_from_flags(value["answer"], pieces, flags, 2), value["answer"], "expand")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="models/rubert-base-cased")
    parser.add_argument("--original", required=True)
    parser.add_argument("--out", default="artifacts/base34_diagnostic.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    original = json.loads((ROOT / "artifacts/actions33_cv_report_base3.json").read_text(encoding="utf-8"))
    if original["folds"] != 2 or len(original["rows"]) != 2:
        raise ValueError("ожидался исходный отчёт двух фолдов")
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    buckets = grouped_folds(pairs, 2)
    if [id_digest(bucket) for bucket in buckets] != original["fold_ids_sha256"]:
        raise ValueError("фолды не совпадают с исходным отчётом")
    source = Path(args.source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "artifacts/actions33_weights_manifest.json").read_text(encoding="utf-8"))
    if manifest["sha256"] != expected["sha256"] or digest(source / "pytorch_model.bin") != expected["sha256"]:
        raise ValueError("исходные веса отличаются от прошлого прогона")
    tokenizer, model = lexfeat.load_feature_model(source)
    model.base.config._attn_implementation = "eager"
    output = {"source_run": 43828152726, "source_sha256": expected["sha256"], "folds": [], "epoch_1_to_3": "не проверено: состояние эпохи 1 не сохранялось", "training": "не проводилось"}
    for index in (0, 1):
        checkpoint = Path(args.original) / "artifacts/cv/base3" / f"fold_{index + 1}" / "feature_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if index:
            del model
            tokenizer, model = lexfeat.load_feature_model(source)
            model.base.config._attn_implementation = "eager"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["state_dict"]
        changes = count_changes(model, state)
        optimizer_encoder = sum(parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad and not (name.startswith("proj") or "classifier" in name))
        optimizer_head = sum(parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad and (name.startswith("proj") or "classifier" in name))
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("чекпоинт не загрузился строго")
        del payload, state
        model.eval()
        record = {"fold": index + 1, "checkpoint_loaded_strictly": True, "checkpoint_sha256": digest(checkpoint), "parameters": changes, "optimizer_encoder_parameters_reconstructed": optimizer_encoder, "optimizer_head_parameters_reconstructed": optimizer_head, "optimizer_total_parameters_reconstructed": optimizer_encoder + optimizer_head, "learning_rates": schedule(original["rows"][index]["train_pairs"], original["epochs"])}
        if index == 0:
            held = buckets[0]
            cache = {}
            for begin in range(0, len(held), 16):
                cache.update(lexfeat.batched_probabilities(tokenizer, model, held[begin:begin + 16], top=4, batch=4))
                if (begin // 16 + 1) % 5 == 0:
                    print(f"fold 1: оценено {min(begin + 16, len(held))}/{len(held)}", flush=True)
            if len(cache) != len(held):
                raise ValueError("не все пары получили вероятности")
            original_tuning = original["rows"][0]["tuning"]
            start, extend = original_tuning["start"], original_tuning["extend"]
            pred = {row["pair_id"]: predict(cache[row["pair_id"]], start, extend) for row in held}
            ev = metrics(held, lambda row: pred[row["pair_id"]])
            recalculated = {f"{level}_{metric}": ev[f"{level}_level"][metric] for level in ("token", "span") for metric in ("precision", "recall", "f1")}
            recalculated.update({"pair_f1": ev["pair_level"]["f1"], "fpr_pair": ev["fpr"], "fpr_token": token_fpr(held, lambda row: pred[row["pair_id"]]), "type_accuracy": ev["type_accuracy"], "type_correct": ev["type_correct"], "matched_spans": ev["span_level"]["tp"]})
            if recalculated != original_tuning["metrics"]:
                raise ValueError(f"повторная оценка расходится с исходной: {recalculated}")
            sample = random.Random(34).sample(held, 20)
            positives, negatives, sample_rows = [], [], []
            for row in sample:
                value = cache[row["pair_id"]]
                scores = [p[2] for p in value["pieces"]]
                for left, right, score in value["pieces"]:
                    target = positives if any(max(left, span["start"]) < min(right, span["end"] + 1) for span in row["spans"]) else negatives
                    target.append(score)
                sample_rows.append({"pair_id": row["pair_id"], "gold_spans": len(row["spans"]), "subwords": len(scores), "min": min(scores) if scores else None, "median": statistics.median(scores) if scores else None, "max": max(scores) if scores else None, "at_or_above_start": sum(score >= start for score in scores), "predicted_spans": len(pred[row["pair_id"]])})
            prepared = prepare(held, cache)
            grid = [compute(prepared, a, b, GATE) for a in GRID for b in GRID if b <= a]
            eligible = [row for row in grid if within_fpr_limits(row)]
            best_eligible = max(eligible, key=lambda row: (row["token"]["f1"], -row["start"], -row["extend"])) if eligible else None
            best_unrestricted = max(grid, key=lambda row: (row["token"]["f1"], -row["start"], -row["extend"]))
            record["holdout"] = {"pairs": len(held), "cached_pairs": len(cache), "chosen_start": start, "chosen_extend": extend, "pair_gate": GATE, "pairs_with_subword_ge_start": sum(any(piece[2] >= start for piece in cache[row["pair_id"]]["pieces"]) for row in held), "pairs_passing_gate": sum(max((piece[2] for piece in cache[row["pair_id"]]["pieces"]), default=0.0) >= GATE for row in held), "predicted_pairs": sum(bool(value) for value in pred.values()), "predicted_spans": sum(len(value) for value in pred.values()), "metrics_reproduced": recalculated, "grid_count": len(grid), "eligible_count": len(eligible), "eligible_positive_token_f1": sum(row["token"]["f1"] > 0 for row in eligible), "best_eligible": best_eligible, "best_unrestricted_diagnostic_only": best_unrestricted, "sample_seed": 34, "sample_pairs": sample_rows, "sample_gold_positive_subwords": probability_summary(positives), "sample_gold_negative_subwords": probability_summary(negatives)}
        output["folds"].append(record)
        print(f"fold {index + 1}: loaded, параметры {optimizer_encoder}+{optimizer_head}", flush=True)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"отчёт: {target}", flush=True)


if __name__ == "__main__":
    main()
