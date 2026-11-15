import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def main():
    before = read("diagnose_scheme13_before.json")
    after = read("diagnose_scheme13_after.json")
    boundaries = read("diagnose_scheme13_boundaries.json")
    truncated = read("diagnose_scheme13_truncated.json")
    snap = read("diagnose_snap_28.json")
    bce = read("cv_report_feat5.json")
    focal = read("cv_report_focal.json")
    report = {
        "status": "диагностика завершена; новое обучение не выполнялось",
        "audit_50": {"before": before["audit_totals"], "after": after["audit_totals"], "examples": [{key: row[key] for key in ("pair_id", "train_positive_subwords", "eval_positive_words_total", "matched_positive_positions", "disagreement_fraction")} for row in before["audit_50"][:3]], "detail": "artifacts/diagnose_scheme13_before.json и artifacts/diagnose_scheme13_after.json"},
        "dataset_token_stats_before_fix": before["token_stats"],
        "loss_batch_before_fix": before["loss_batch"],
        "negatives_before_fix": before["negatives"],
        "chunking_before_fix": {key: value for key, value in before["chunking"].items() if key != "long_examples"},
        "twenty_longest_answers": "artifacts/diagnose_scheme13_before.json: chunking.long_examples",
        "truncation_detail": {"pairs": truncated["truncated_pairs"], "partial_spans": truncated["partial_spans"], "file": "artifacts/diagnose_scheme13_truncated.json"},
        "boundary_check_all": {key: value for key, value in boundaries.items() if not key.endswith("examples")},
        "boundary_examples": "artifacts/diagnose_scheme13_boundaries.json",
        "duplicate_spans_diagnostic": {key: snap[key] for key in ("pairs", "raw_spans", "snapped_spans", "unique_snapped_spans", "duplicate_snapped_spans", "raw_metrics_on_50_with_corrected_evaluator", "deduplicated_metrics_on_50_with_corrected_evaluator")},
        "reference_cv": {
            "scheme12_bce": {"loss": bce.get("loss", "bce по умолчанию"), "span_precision": bce["summary"]["span_precision"], "span_f1": bce["summary"]["span_f1"]},
            "focal2": {"loss": focal["loss"], "span_precision": focal["summary"]["span_precision"], "span_f1": focal["summary"]["span_f1"]},
            "note": "разные данные и состав батчей; эти ранее сохранённые CV-метрики не пересчитывались после исправления метрики",
        },
        "changes": ["исправлено пересечение half-open субтокена с включительной границей спана в обучающих метках", "токен метрики засчитывается при любом пересечении со спаном", "исправлена лишняя единица в координатах выходных субтокенов обоих методов инференса", "на будущее продолжение обучения защищено проверкой исходного кода; старые чекпоинты автоматически не продолжаются с новой разметкой"],
        "not_done": ["третья эпоха и ступень B не запускались", "окна ответов и потеря частично обрезанных спанов не исправлены", "BCE с весами не запускался: диагностические проверки не все чистые", "полные метрики старых чекпоинтов после изменения оценки не пересчитывались", "test 301 не запускался"],
        "time_seconds": {"full_before": before["seconds"], "audit_after": after["seconds"], "boundary_check": boundaries["seconds"], "truncation_detail": truncated["seconds"], "snap_sample": snap["seconds"]},
    }
    target = ROOT / "artifacts/diagnose_scheme13_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
