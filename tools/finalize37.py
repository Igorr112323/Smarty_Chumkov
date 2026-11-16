import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    folder = ROOT / "artifacts"
    summary = json.loads((folder / "nested37_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "готово" or not summary["nominal_target_by_mean"]["seed5_fine"]:
        raise ValueError("нет полного результата с достижением целевого среднего")
    folds = [json.loads((folder / f"nested37_fold_{index}.json").read_text(encoding="utf-8")) for index in range(1, 6)]
    config = {"status": "зафиксировано описание CV, веса не сохранены и к развёртыванию не готовы", "evaluation_only": True, "deployable": False, "model": "rubert-tiny2", "initial_weights_sha256": "137fa2b1d944dae19c74456dfe8fac2f780d9acf34e037f5d1e37acba1157768", "git_sha": summary["git_sha"], "seeds": [0, 1, 2, 3, 4], "epochs": 3, "folds": 5, "train_folds": 3, "inner_folds": 1, "external_folds": 1, "fold_grouping": "связные компоненты всех doc_id, seed 0", "fold_ids_sha256": summary["fold_ids_sha256"], "thresholds": summary["thresholds"], "threshold_selection": "seed 0: грубая сетка 0.10..0.95 шаг 0.05; затем ±0.05 шаг 0.01; максимум token F1 на внутреннем фолде при FPR_pair и FPR_token <= 0.40; T_extend=T_start/2", "prediction": "среднее вероятностей пяти сидов на внешнем фолде, затем порог fine соответствующего фолда", "pair_gate": 0.45, "token_features": 5, "new_pair_features": False, "loss": "focal gamma=2, label_smoothing=0.1, pair loss weight=0.3", "encoder_learning_rate": 0.00002, "head_learning_rate": 0.0005, "checkpoint_sha256": summary["checkpoint_hashes"], "checkpoint_note": "SHA-256 временного torch.save(state_dict); сами файлы удалены, побайтовое восстановление и развёртывание не проверены", "versions_by_fold": [{"fold": fold["fold"], "torch": fold["torch"], "transformers": fold["transformers"], "python": fold["python"]} for fold in folds], "external_data": False, "test301": False, "limitations": ["те же внешние фолды повторно использованы после изучения предыдущих результатов; оценка адаптивна", "внешний FPR_token превышает 0.40 в одном фолде", "сохранённой развёртываемой модели нет"]}
    (folder / "frozen_final.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path = folder / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["headline"] = {"description": "вложенный протокол, порог на внутреннем фолде; пять сидов, мелкая сетка", "metrics_mean_std": summary["modes"]["seed5_fine"], "target": "token F1 >= 0.50 и FPR_token <= 0.40", "target_met_by_fold_means": True, "gate_passed": None, "gate_note": "планка по средним выполнена исследовательски; независимое подтверждение отсутствует, в одном внешнем фолде FPR_token > 0.40, модельные файлы отсутствуют", "reused_external_folds": True, "test301_evaluated": False, "config": "artifacts/frozen_final.json", "details": "artifacts/nested37_report.md"}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Сохранены описание протокола и headline без повторной оценки test.")


if __name__ == "__main__":
    main()
