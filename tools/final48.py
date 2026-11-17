import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"


def load(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def write(name, data):
    (ART / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def best_variant(v4, v5):
    return "v5" if v5["pooled_exact_from_counts"]["token_f1"] >= v4["pooled_exact_from_counts"]["token_f1"] else "v4"


def main():
    v4, v5 = load("analysis48_v4.json"), load("analysis48_v5.json")
    best = best_variant(v4, v5)
    data = v5 if best == "v5" else v4
    manifest = json.loads((ROOT / "weights/final/manifest.json").read_text(encoding="utf-8"))
    if manifest["variant"] != best or manifest["status"] != "готово" or not manifest["restore_verified"]:
        raise ValueError("не проверен финальный ансамбль выбранного варианта")
    frozen = load("frozen_final.json")
    if frozen["variant"] != best:
        raise ValueError("frozen_final не описывает выбранный вариант")
    comparison = {"selection_rule": "больший неокруглённый сводный token F1 на 699 CV-парах; при равенстве В5", "selected": best, "v4_pooled_token_f1": v4["pooled_exact_from_counts"]["token_f1"], "v5_pooled_token_f1": v5["pooled_exact_from_counts"]["token_f1"], "extra": "исходный список extra 200 отсутствует; по решению пользователя оценка пропущена", "no_v6": True, "contract_confirmed": False}
    write("decision48.json", comparison)
    frozen["selection48"] = comparison
    frozen["cv_metrics_exact"] = {"macro": data["macro_exact_from_counts"], "pooled": data["pooled_exact_from_counts"]}
    write("frozen_final.json", frozen)
    report = load("report.json")
    old = report.get("headline")
    if isinstance(old, dict):
        report.setdefault("headline_history", []).append(old)
    report["headline"] = data["pooled_exact_from_counts"]["token_f1"]
    report["headline_context"] = {"variant": best, "metric": "pooled_token_f1", "scope": data["scope"], "not_independent_contract_evaluation": True, "bootstrap": "artifacts/analysis48_" + best + ".json", "comparison": "artifacts/decision48.json"}
    write("report.json", report)
    macro, pooled = data["macro_exact_from_counts"], data["pooled_exact_from_counts"]
    lines = ["# Промпт 48: В4, В5 и доверительные интервалы", "", f"Выбран **{best.upper()}** по точному сводному token F1 на 699 CV-парах; при равенстве выбирается В5. Никакого заявления о выполнении цели здесь нет: выбор адаптивен, независимая договорная оценка не проведена.", "", f"В4: {v4['pooled_exact_from_counts']['token_f1']}; В5: {v5['pooled_exact_from_counts']['token_f1']}. В5 — В4: {v5['pooled_exact_from_counts']['token_f1'] - v4['pooled_exact_from_counts']['token_f1']}. В6 и последующие варианты не запускались.", "", "## Четыре числа рядом", "", "Macro — среднее точных метрик пяти внешних фолдов, вычисленных по счётчикам без предварительного округления. Сводная CV — по объединённым 699 out-of-fold предсказаниям. Колонка extra пуста: исходный идентифицируемый список 200 пар отсутствует, и пользователь разрешил пропустить оценку. Требования указаны только там, где они заданы.", "", "| Метрика | CV macro | CV сводный | extra 200 | что требует договор |", "|---|---:|---:|---:|---|"]
    rows = [("token precision", "token_precision", "—"), ("token recall", "token_recall", "—"), ("token F1", "token_f1", "≥ 0,50"), ("FPR_token", "fpr_token", "≤ 0,40"), ("span precision (IoU ≥ 0,5)", "span_precision", "—"), ("span recall (IoU ≥ 0,5)", "span_recall", "—"), ("span F1 (IoU ≥ 0,5)", "span_iou_f1", "—"), ("span exact F1", "span_exact_f1", "—"), ("span overlap F1", "span_overlap_f1", "—"), ("pair precision", "pair_precision", "—"), ("pair recall", "pair_recall", "—"), ("pair F1", "pair_f1", "—")]
    for title, key, contract in rows:
        lines.append(f"| {title} | {macro[key]} | {pooled[key]} | не оценено | {contract} |")
    comparison_detail = load("comparison48.json")
    lines += ["", f"Попарный аудит В4/В5: границы спанов совпали на всех {comparison_detail['pairs']} парах, но сырые вероятности различались на {comparison_detail['raw_probability_difference_pairs']} парах, полные объекты спанов — на {comparison_detail['full_predicted_span_difference_pairs']}; SHA совпал у {comparison_detail['same_checkpoint_sha']} из {comparison_detail['checkpoint_count']} checkpoint. Равенство метрик не означает побайтовую идентичность вариантов.", "", "Какую из оценок считать договорной — решение заказчика, не исполнителя. Приведены все. При этом 699 CV-пар не являются единой независимой отложенной по документам выборкой из 1200 пар; extra 200, даже если будут предоставлены, сами по себе тоже не составляют 1200 пар.", "", "FPR_token = FP токенов / число отрицательных токенов. Положительный pair TP требует совпадения спанов при IoU ≥ 0,5; exact — идентичные границы, overlap — любое пересечение, сопоставление 1:1. Максимум FPR_token по фолдам приведён как диагностика, не самостоятельное условие договора.", "", f"В4 FPR_token по фолдам: средний {v4['macro_exact_from_counts']['fpr_token']}, максимум {max(m['fpr_token'] for m in v4['fold_metrics_exact'])}; В5: средний {v5['macro_exact_from_counts']['fpr_token']}, максимум {max(m['fpr_token'] for m in v5['fold_metrics_exact'])}.", "", "## Bootstrap по документам", "", f"10 000 ресэмплов с возвращением, seed 48; единица — связная компонента пар по общим doc_ids (их {data['bootstrap']['groups']}). Процентильный 95% интервал (2,5% и 97,5%; линейная интерполяция). Метрика пересчитана из сумм счётчиков каждого ресэмпла; фиксированные out-of-fold прогнозы, без переобучения или перенастройки порогов.", "", "| Метрика | Точечная оценка | 95% CI | Ширина |", "|---|---:|---|---:|"]
    for key in ("token_f1", "fpr_token"):
        result = data["bootstrap"]["metrics"][key]
        lines.append(f"| {key} | {result['point']} | [{result['ci_95'][0]}; {result['ci_95'][1]}] | {result['width']} |")
    ci = data["bootstrap"]["metrics"]["token_f1"]["ci_95"]
    if ci[0] <= 0.5 <= ci[1]:
        lines.append(f"\nТочечная оценка {pooled['token_f1']}, 95% CI накрывает 0,50: по этому интервалу статистически неотличимо от требуемого значения. Это не доказательство равенства, эквивалентности или достижения цели. Интервал условный, не учитывает вариативность обучения и адаптивный выбор по этой CV.")
    else:
        lines.append("\n95% CI token F1 не накрывает 0,50. Интервал условный: он не учитывает вариативность обучения и адаптивный выбор по этой CV.")
    lines += ["", "## Происхождение и ограничения", "", "В каждом варианте проверены SHA входных фолдов и train/inner/outer ID для 20 заданий с 5 сидами. Обучение выполнялось только на трёх из пяти фолдов исходных 699 train/val; пересечений doc_id между фолдами нет. Исторический список extra 200 не сохранился: проверить пересечения его pair_id/doc_id с обучением невозможно, поэтому однозначное отсутствие пересечений документов не установлено. По решению пользователя extra не оценивались. Историческое упоминание RuFacts не доказывает тождество набора; чужой датасет не подменялся.", "", "Test 301 не оценивался; внешние данные не подключались. Пороги отбирались по внутренним фолдам при ограничениях FPR ≤ 0,40. Клаузный expand сохранён; предложенческий снап и компенсация не применялись. В5: до 15 эпох, patience 3, embeddings заморожены всё обучение; различие результатов не доказывает причинный эффект разморозки: эксперименты и прочие факторы могли отличаться.", "", f"В `weights/final/` — пять сидов {best.upper()} только первого внешнего ансамбля, а не пять моделей, воспроизводящих всю CV. Восстановление проверено в `artifacts/final47_restore_verification.json`; происхождение, SHA и параметры — в `artifacts/frozen_final.json` и манифестах. Остальные ансамбли: `weights/ablation47/{best}/`. Исторические артефакты остаются в Git.", "", "Полные точные данные, счётчики и интервалы: `artifacts/analysis48_v4.json`, `artifacts/analysis48_v5.json`; выбор: `artifacts/decision48.json`. Сводка промпта 47 сохранена отдельно в `artifacts/ablation47_report.md`."]
    (ART / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False))


if __name__ == "__main__":
    main()
