import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("token_precision", "token_recall", "token_f1", "span_precision", "span_recall", "span_f1", "pair_f1", "fpr_pair", "fpr_token", "type_accuracy")


def display(value):
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def main():
    reports = {}
    for name in ("bce", "focal"):
        path = ROOT / f"artifacts/stage_b31_{name}.json"
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    lines = ["# Промт 31: ступень B на наших данных", "", "Здесь использованы только 699 пар train/val, разбиение по связным компонентам всех doc_id; test 301 не использовался. Для каждого полного фолда модель заново загружена из checkpoint ступени A, оптимизатор запущен с нуля, выполнены 3 эпохи по всем парам остальных фолдов. Пять лексических признаков сохранены. Encoder LR 0,000004, head LR 0,0001 — в пять раз ниже ступени A. Ограничение по памяти: один fold-модель в процессе; промежуточные результаты записываются после каждого фолда.", "", "BCE использует прежний weighted BCE pos_weight=11 без pair loss; focal использует focal gamma=2 и дополнительный pair loss с весом 0,3. Это различающиеся режимы обучения, а не контролируемое сравнение только предобучения. Прежние базовые CV также отличались составом обучения и фолдами; точные значения не являются парным сравнением.", "", "В каждом отложенном фолде на нём же выбирается гистерезис: максимум token F1 при обоих FPR ≤ 0,40, иначе максимум без ограничений. Сетка: start 0,25…0,95 с шагом 0,10, extend 0,10…0,90 с шагом 0,10, только extend < start; pair gate 0,45, snap expand. Затем метрики измеряются на том же фолде: эти оценки оптимистичны для выбора порога, не независимы. Оценка типа относится только к совпавшим спанам, класс предсказывается существующей постобработкой.", "", "| метрика | BCE mean ± std | focal mean ± std |", "|---|---:|---:|"]
    for metric in METRICS:
        values = [display(reports[name]["summary"][metric]) if name in reports and reports[name]["status"] == "готово" else "не выполнено" for name in ("bce", "focal")]
        lines.append(f"| {metric} | {values[0]} | {values[1]} |")
    for name in ("bce", "focal"):
        if name not in reports:
            lines += ["", f"## {name}", "", "Не выполнено."]
            continue
        report = reports[name]
        lines += ["", f"## {name}: {report['status']}, {len(report['rows'])}/{report['identity']['folds']} фолдов", "", "| фолд | train / holdout | start / extend | оба FPR ≤ 0,40 | token P/R/F1 | span P/R/F1 | pair F1 | FPR_pair / FPR_token | type accuracy (совпадений) |", "|---:|---|---|---|---|---|---:|---|---|"]
        for row in report["rows"]:
            tune = row["tuning"]
            m = tune["measurements"]
            token = "/".join(f"{m['token_' + key]:.4f}" for key in ("precision", "recall", "f1"))
            span = "/".join(f"{m['span_' + key]:.4f}" for key in ("precision", "recall", "f1"))
            lines.append(f"| {row['fold']} | {row['train_pairs']} / {row['held_pairs']} | {tune['start']:.2f} / {tune['extend']:.2f} | {'да' if tune['eligible'] else 'нет, fallback'} | {token} | {span} | {m['pair_f1']:.4f} | {m['fpr_pair']:.4f} / {m['fpr_token']:.4f} | {m['type_accuracy']:.4f} ({m['matched_spans']}) |")
    lines += ["", "## Стыки перевода", "", "30 случайных позитивных пар (seed 31): механическое слипание без пробела — 1/30 (3,3%); отдельно разрыв пунктуации — 12/30. Если учитывать оба вида, 13/30 (43,3%). Контексты, полная разметка и ограничения — `artifacts/seams31.md`. Причинное влияние не установлено; перевод не переделывался.", "", "## Ограничение вывода по архитектуре", "", "Отсутствие прироста относительно 0,3263 само по себе не доказывает, что причина — размер модели: различаются данные, режимы обучения, разбиения и настройка, а порог выбирался на оцениваемом фолде. Сравнение с rubert-base-cased или LettuceDetect здесь не проведено; о достаточности скорости, квантовании и причине недостижения качества данных нет.", "", "Пилот BCE со старой узкой сеткой остановлен после первого фолда и начала второго: на первом не нашлось точек с обоими FPR ≤ 0,40. Он не включён в основное CV; `artifacts/stage_b31_bce_pilot.json`. Сетка для итогового CV заранее расширена; RAGTruth повторно не перебирался."]
    (ROOT / "artifacts/stage_b31_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
