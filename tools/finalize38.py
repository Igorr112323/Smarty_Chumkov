import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    artifacts = ROOT / "artifacts"
    summary = json.loads((artifacts / "nested38_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "готово":
        raise ValueError("вложенная CV не завершена")
    folder = ROOT / "weights/frozen"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["outer_fold"] != 1 or manifest["inner_fold"] != 2 or manifest["git_sha"] != summary["git_sha"] or manifest["gold_train_val_sha256"] != summary["gold_train_val_sha256"] or [row["seed"] for row in manifest["checkpoints"]] != list(range(5)):
        raise ValueError("манифест весов не соответствует первому фолду")
    threshold = summary["folds"][0]["threshold"]
    manifest.update({"status": "кандидат", "model": "rubert-tiny2 + 5 token lexical features", "threshold": threshold, "gate": 0.45, "fold_ids_sha256": summary["fold_ids_sha256"], "fpr_note": "порог выбран по четырём внутренним фолдам; на новых данных FPR не гарантирован", "input_weights_sha256": "137fa2b1d944dae19c74456dfe8fac2f780d9acf34e037f5d1e37acba1157768"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {"status": "кандидат, проверка восстановления весов не завершена", "deployable": False, "model": manifest["model"], "seeds": [0, 1, 2, 3, 4], "epochs": 3, "folds": 5, "inner_folds_per_outer": 4, "training_folds_per_inner": 3, "data": "только train+val, 699 пар", "gold_train_val_sha256": summary["gold_train_val_sha256"], "fold_ids_sha256": summary["fold_ids_sha256"], "seed": 0, "git_sha": summary["git_sha"], "thresholds_by_outer_fold": [{"fold": row["outer_fold"], "start": row["threshold"]["start"], "extend": row["threshold"]["extend"]} for row in summary["folds"]], "frozen_model_fold": 1, "frozen_model_training_id_sha256": manifest["train_id_sha256"], "frozen_model_threshold": {"start": threshold["start"], "extend": threshold["extend"]}, "selection_rule": summary["selection"], "checkpoints": [{"seed": row["seed"], "sha256": row["sha256"], "size": row["size"], "manifest": f"weights/frozen/seed_{row['seed']}/manifest.json"} for row in manifest["checkpoints"]], "model_manifest": "weights/frozen/manifest.json", "limitations": ["внешние фолды использованы в предыдущем выборе конфигурации, независимого подтверждения нет", "модель сохранена для заранее выбранного внешнего фолда 1; пять ансамблей CV не являются одной моделью", "внутреннее ограничение худшего фолда не гарантирует ограничение на внешних данных"]}
    (artifacts / "frozen_final.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = artifacts / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    values = summary["summary"]
    observed = values["token_f1"]["mean"] >= 0.5 and summary["max_outer_fpr_token"] <= 0.4 and summary["max_outer_fpr_pair"] <= 0.4
    report["headline"] = {"description": "вложенная CV; порог на внутренних фолдах по ограничению худшего внутреннего фолда", "metrics_mean_std": values, "max_outer_fpr_token": summary["max_outer_fpr_token"], "max_outer_fpr_pair": summary["max_outer_fpr_pair"], "target_met_on_observed_folds": observed, "gate_passed": None, "gate_note": "оценка получена на фолдах, использованных для выбора конфигурации; независимого подтверждения нет, test 301 израсходован ранее на другой конфигурации", "test301_evaluated": False, "config": "artifacts/frozen_final.json", "details": "artifacts/nested38_summary.json"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Сохранён кандидат и обновлён headline; проверка восстановления ещё не проведена.")


if __name__ == "__main__":
    main()
