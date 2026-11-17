import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    manifest = read(ROOT / "weights/final/manifest.json")
    verification = read(ARTIFACTS / "final47_restore_verification.json")
    variant = manifest["variant"]
    summary = read(ARTIFACTS / f"ablation47_{variant}_summary.json")
    pooled = read(ARTIFACTS / f"ablation47_{variant}_pooled.json")
    if manifest["status"] != "готово" or not manifest["restore_verified"] or not verification["matches_original_fold"] or verification["variant"] != variant or summary["status"] != "готово":
        raise ValueError("отдельная проверка модели не завершена")
    ensembles = []
    for fold in range(1, 6):
        inner = fold % 5 + 1
        record = read(ARTIFACTS / f"ablation47_{variant}_outer{fold}_inner{inner}.json")
        if [seed["seed"] for seed in record["seeds"]] != [0, 1, 2, 3, 4] or not record["primary"] or any(not seed["checkpoint"]["restore_predictions_equal"] for seed in record["seeds"]):
            raise ValueError("неполный ансамбль")
        ensembles.append({"outer_fold": fold, "inner_fold": inner, "train_id_sha256": record["train_id_sha256"], "outer_id_sha256": record["outer_id_sha256"], "threshold": summary["folds"][fold - 1]["threshold"], "checkpoints": [{"seed": seed["seed"], "best_epoch": seed["best_epoch"], "epochs_ran": seed["epochs_ran"], "sha256": seed["checkpoint"]["sha256"], "size": seed["checkpoint"]["size"], "manifest": f"weights/ablation47/{variant}/fold{fold}/seed_{seed['seed']}/manifest.json"} for seed in record["seeds"]]})
    reference = read(ARTIFACTS / "nested38_summary.json")
    config = {"status": "готово: восстановлен ансамбль первого внешнего фолда", "deployable": True, "variant": variant, "model": manifest["model"], "seeds": [0, 1, 2, 3, 4], "max_epochs": 15, "patience": 3, "embedding_freeze": "всё обучение" if variant == "v5" else "первые 5 эпох, затем LR x0,1", "ema": False, "clause_expand": True, "sentence_snap": False, "snap_compensation": False, "gate": 0.45, "folds": 5, "inner_folds_per_outer": 4, "training_folds_per_inner": 3, "data": "только train+val, 699 пар", "gold_train_val_sha256": manifest["gold_train_val_sha256"], "fold_ids_sha256": reference["fold_ids_sha256"], "git_sha": summary["git_sha"], "frozen_model_fold": 1, "frozen_model_training_id_sha256": manifest["train_id_sha256"], "frozen_model_threshold": manifest["threshold"], "model_manifest": "weights/final/manifest.json", "checkpoints": [{"seed": seed["seed"], "sha256": seed["sha256"], "size": seed["size"], "manifest": f"weights/final/seed_{seed['seed']}/manifest.json"} for seed in manifest["checkpoints"]], "cv_ensembles": ensembles, "selection_rule": "максимум среднего token F1 по четырём внутренним фолдам, FPR_pair и FPR_token <=0,40 на каждом; внешние метки не участвуют в выборе порога", "restore_verified": True, "restore_verification": "artifacts/final47_restore_verification.json", "limitations": ["пять checkpoint в weights/final — один ансамбль внешнего фолда 1, не вся CV", "фолды использовались ранее для экспериментов; независимого подтверждения нет", "отложенная по документам часть 1200 пар не оценивалась"]}
    write(ARTIFACTS / "frozen_final.json", config)
    report_path = ARTIFACTS / "report.json"
    report = read(report_path)
    fields = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")
    metric_rows = [fold["metrics"] for fold in summary["folds"]]
    means = {field: {"mean": round(statistics.mean(row[field] for row in metric_rows), 4), "std": round(statistics.pstdev(row[field] for row in metric_rows), 4)} for field in fields}
    macro = statistics.mean(row["token_f1"] for row in metric_rows)
    report["headline"] = {"description": f"{variant.upper()}, адаптивная вложенная CV на 699 train/val парах; не независимый договорный тест", "token_f1": round(macro, 4), "metrics_mean_std": means, "pooled_token_f1": pooled["pooled_token_f1"], "pooled_token_f1_exact": pooled["pooled_token_f1_exact"], "pooled_fpr_token": pooled["pooled_fpr_token"], "max_outer_fpr_token": summary["summary"]["fpr_token"]["max"], "max_outer_fpr_pair": summary["summary"]["fpr_pair"]["max"], "target_met_hybrid_macro_f1_pooled_fpr_cv": macro >= 0.5 and pooled["pooled_fpr_token"] <= 0.4, "target_met_pooled_f1_and_fpr_cv": pooled["pooled_token_f1_exact"] >= 0.5 and pooled["pooled_fpr_token"] <= 0.4, "contract_confirmed": False, "gate_passed": None, "gate_note": "699 ранее изученных train/val вместо независимой договорной отложенной части 1200; единой оценкой договора это не является", "test301_evaluated": False, "config": "artifacts/frozen_final.json", "details": f"artifacts/ablation47_{variant}_summary.json"}
    write(report_path, report)
    print(f"{variant}: манифест и headline сохранены; договор независимо не подтверждён")


if __name__ == "__main__":
    main()
