import json
import os
from pathlib import Path

METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")


def main():
    lines = ["# rubert-base: CV на наших train/val", "", "Порог гистерезиса выбирается и измеряется на одном фолде: метрики оптимистичны. Полный gold и test не обрабатываются; внешних данных в обучении нет."]
    path = Path("artifacts/cv_report_base3.json")
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        lines += ["", f"Статус: {report['status']}; завершено {len(report['rows'])}/{report['folds']} фолдов; эпох на фолд: {report['epochs']}; секунд: {report.get('elapsed_s', 'неизвестно')}."]
        if "summary" in report:
            lines += ["", "| Метрика | mean ± std |", "|---|---:|"]
            for name in METRICS:
                value = report["summary"][name]
                lines.append(f"| {name} | {value['mean']:.4f} ± {value['std']:.4f} |")
            span = report["summary"]["span_f1"]["mean"]
            lines.append(f"\nSpan F1 {span:.4f}: {'выше' if span > 0.2982 else 'не выше'} прежней базы 0,2982; состав фолдов и режим обучения отличаются.")
    else:
        lines += ["", "Отчёт CV отсутствует: обучение не завершилось или не началось."]
    speed = Path("artifacts/speed_report_base.json")
    if speed.exists():
        rows = json.loads(speed.read_text(encoding="utf-8"))["rows"]
        lines += ["", "| Длина документа | мс/1000 знаков, модель | мс/1000 знаков, ensemble | batch |", "|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['length']} | {row['model_ms_per_1000_doc']} | {row['ensemble_ms_per_1000_doc']} | {row['batch']} |")
        lines.append("\nДля длины 20000 использованы повторения реальных документов только в нагрузочном замере; в обучении их нет. Таймер модели включает подготовку и инференс, но не постобработку.")
    else:
        lines += ["", "Замер скорости не выполнен."]
    manifest = Path("models/rubert-base-cased/manifest.json")
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        lines += ["", f"Веса: {info['size']} байт; SHA-256 `{info['sha256']}`. Совпадение с официальным SHA не проверено: эталон не задан."]
    text = "\n".join(lines) + "\n"
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
