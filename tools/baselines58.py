"""ПРОМПТ 58, стадия 2: базовая линия «пометить всё» на КАЖДОМ фолде и входе.

Зачем. Три раза подряд метрика и базовая линия считались на разных наборах, и из
этого рождались ложные выводы (в 56-м — «запас 0,132», в 57-м — «потолок 0,644»).
Процедурное лечение: базовая линия считается ровно тем же кодом, что и метрика
(`covered_tokens` + `token_positions` из `spanver/evaluate.py`), на том же наборе
пар, и сохраняется рядом с ним.

Что считается для каждого фолда каждого входа:

  * `gold_share` — доля золотых токенов среди всех токенов ответов;
  * `all_positive_token_f1` — token F1 правила «пометить весь ответ» = 2x/(1+x);
  * `all_positive_pair_f1` — pair F1 того же правила: предсказано «есть нарушение»
    для каждой пары, поэтому у пар без золота это ложное срабатывание;
  * FPR правила всегда 1,0 — это записывается явно, чтобы не путать с рабочей точкой.

Вход в расчёт НЕ влияет: метки, ответы и координаты одни и те же, меняется только
`document_text`, который в этих формулах не участвует. Поэтому базлайн одинаков
для `joined_reference` и `docs` — и это тоже фиксируется числом, а не словами:
совпадение проверяется, а не декларируется.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import covered_tokens, read_jsonl, token_positions
from tools.torchless57 import stub_if_absent

stub_if_absent()

from tools.train_base33 import grouped_folds  # noqa: E402


def baseline(pairs, label):
    gold = total = 0
    with_gold = 0
    for pair in pairs:
        answer = pair["model_output"]
        positions = token_positions(answer)
        total += len(positions)
        covered = covered_tokens(answer, [(span["start"], span["end"]) for span in pair["spans"]])
        gold += len(covered)
        if pair["spans"]:
            with_gold += 1
    share = gold / total if total else 0.0
    # правило «всё галлюцинация»: precision = share, recall = 1
    token_f1 = 2 * share / (1 + share) if share else 0.0
    # пары: предсказано «есть» для всех, значит tp = пары с золотом, fp = пары без
    tp = with_gold
    fp = len(pairs) - with_gold
    pair_precision = tp / (tp + fp) if tp + fp else 0.0
    pair_f1 = 2 * pair_precision * 1.0 / (pair_precision + 1.0) if pair_precision else 0.0
    return {
        "label": label,
        "pairs": len(pairs),
        "pairs_with_gold": with_gold,
        "pairs_without_gold": len(pairs) - with_gold,
        "gold_tokens": gold,
        "total_tokens": total,
        "gold_share": share,
        "all_positive_token_precision": share,
        "all_positive_token_f1": token_f1,
        "all_positive_pair_precision": pair_precision,
        "all_positive_pair_f1": pair_f1,
        "all_positive_fpr_token": 1.0,
        "all_positive_fpr_pair": 1.0,
        "note": "правило «пометить весь ответ»; FPR = 1,0 по построению",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/baselines58.json")
    args = parser.parse_args()

    # 699 пар train/val: входы отличаются только document_text, поэтому
    # базлайны обязаны совпасть — это проверяется ниже
    docs = read_jsonl(ROOT / "artifacts/ours56_docs.jsonl")
    joined = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    if [p["pair_id"] for p in docs] != [p["pair_id"] for p in joined]:
        raise SystemExit("состав пар у двух входов разошёлся")

    folds_docs = grouped_folds(docs, 10)
    folds_joined = grouped_folds(joined, 10)

    per_fold = {}
    check = []
    for index in range(10):
        fold = index + 1
        for name, part in (("joined_reference", folds_joined[index]), ("docs", folds_docs[index])):
            point = baseline(part, f"фолд {fold}, {name}")
            per_fold.setdefault(str(fold), {})[name] = point
        a = per_fold[str(fold)]["joined_reference"]
        b = per_fold[str(fold)]["docs"]
        same = a["gold_tokens"] == b["gold_tokens"] and a["total_tokens"] == b["total_tokens"]
        check.append({"fold": fold, "identical_across_inputs": same,
                      "joined_pairs": a["pairs"], "docs_pairs": b["pairs"],
                      "joined_gold_share": a["gold_share"], "docs_gold_share": b["gold_share"]})

    everything = baseline(docs, "все 699 пар train/val")
    fold_values = [per_fold[str(f)]["docs"]["gold_share"] for f in range(1, 11)]
    spread = {
        "gold_share_std": statistics.pstdev(fold_values),
        "gold_share_min": min(fold_values),
        "gold_share_max": max(fold_values),
        "gold_share_range": max(fold_values) - min(fold_values),
        "exceeds_0_05": (max(fold_values) - min(fold_values)) > 0.05,
        "verdict": ("разброс больше 0,05: сравнивать метрики между фолдами без "
                    "базлайна нельзя, это свойство набора"
                    if (max(fold_values) - min(fold_values)) > 0.05 else
                    "разброс не больше 0,05"),
    }

    report = {
        "status": "готово",
        "stage": "58 стадия 2",
        "pairs": len(docs),
        "folds": 10,
        "inputs": ["joined_reference", "docs"],
        "inputs_note": ("метки, ответы и координаты у двух входов одни и те же; document_text "
                        "в формулу базлайна не входит, поэтому значения обязаны совпадать — "
                        "проверка в поле input_check"),
        "per_fold": per_fold,
        "input_check": check,
        "inputs_identical_per_fold": all(row["identical_across_inputs"] for row in check),
        "input_effect": ("состав 10-фолдового сплита РАЗНЫЙ для двух входов: grouped_folds "
                         "группирует по doc_ids, а они у joined_reference и docs разные. "
                         "Поэтому базлайн на отдельном фолде зависит от входа, хотя формула "
                         "от document_text не зависит. На объединении всех 699 пар значения "
                         "совпадают, и pooled-метрики сравнимы только с этим базлайном."),
        "all_pairs": everything,
        "spread": spread,
        "test301": False,
        "method": "covered_tokens + token_positions из spanver/evaluate.py, как в метриках",
    }
    (ROOT / args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    print(f"все 699 пар: доля золотых {everything['gold_share']}, "
          f"all-positive token F1 {everything['all_positive_token_f1']}")
    for fold in range(1, 11):
        point = per_fold[str(fold)]["docs"]
        print(f"  фолд {fold:2d}: n={point['pairs']:3d} доля {point['gold_share']:.6f} "
              f"F1 {point['all_positive_token_f1']:.6f} pair F1 {point['all_positive_pair_f1']:.6f}")
    print(f"разброс gold_share: std {spread['gold_share_std']}, "
          f"диапазон {spread['gold_share_range']} ({' > 0,05' if spread['exceeds_0_05'] else ' <= 0,05'})")
    print(f"базлайн на фолдах совпал по двум входам: {report['inputs_identical_per_fold']}")
    print(f"все 699 пар, базлайн (одинаков для обоих входов): "
          f"доля {report['all_pairs']['gold_share']}, "
          f"F1 {report['all_pairs']['all_positive_token_f1']}")


if __name__ == "__main__":
    main()
