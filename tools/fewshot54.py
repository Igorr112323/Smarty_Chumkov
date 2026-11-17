"""ПРОМПТ 54, задача 2: материалы для ручного few-shot разметчика.

Скрипт ничего не размечает. Он готовит куратору две вещи:

  artifacts/fewshot54_examples.json — 12 пар с нашими человеческими спанами
    (вход: вопрос, документ, ответ; эталон: спаны с типом и исходным
    человеческим типом). Это содержимое промпта;
  artifacts/fewshot54_blind.json — 30 пар БЕЗ спанов и без типа, которые
    куратор размечает вслепую.

Пары для примеров и для слепой разметки берутся из разных компонент doc_id,
чтобы примеры не подсказывали ответы. Слепые 30 пар берутся только из
внутренних фолдов (не из отложенного теста), test 301 не трогается.

Оценка согласия — `tools/score54.py`.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.preflight_13 import our_keys

EXAMPLES = 12
BLIND = 30


def pick(pairs, count, used_docs, require_spans=True):
    chosen = []
    for pair in pairs:
        if require_spans and not pair["spans"]:
            continue
        keys = {key for key in our_keys(pair)}
        if keys & used_docs:
            continue
        chosen.append(pair)
        used_docs |= keys
        if len(chosen) == count:
            break
    return chosen


def main():
    pairs = read_jsonl(ROOT / "artifacts/ours33.jsonl")
    positives = [pair for pair in pairs if pair["spans"]]
    # примеры и слепой набор — из разных фолдов, чтобы документы не пересекались
    used = set()
    examples = pick(positives, EXAMPLES, used)
    blind_pool = [pair for pair in pairs if pair["pair_id"] not in {p["pair_id"] for p in examples}]
    blind_strict = []
    for pair in blind_pool:
        keys = {key for key in our_keys(pair)}
        if keys & used:
            continue
        blind_strict.append(pair)
        used |= keys
        if len(blind_strict) == BLIND:
            break

    def public(pair, with_gold):
        item = {
            "pair_id": pair["pair_id"],
            "question": None,
            "document": pair["document_text"],
            "answer": pair["model_output"],
        }
        item["question"] = lookup_question(pair["pair_id"])
        if with_gold:
            item["gold_spans"] = [
                {"start": span["start"], "end": span["end"], "text": pair["model_output"][span["start"]: span["end"] + 1],
                 "type": span["type"], "human_type": span.get("type_src")}
                for span in pair["spans"]
            ]
        return item

    payload = {
        "status": "готово",
        "instruction": (
            "Разметь ответ так, как размечали наши 300 человеческих спанов: "
            "пометь фрагмент, который не подтверждён контекстом применительно к вопросу. "
            "Учти: 170 из 300 человеческих спанов помечены как Partial или Unconfirmed, "
            "медиана длины спана 157 знаков, 64.67 % спанов — часть предложения, не всё предложение."
        ),
        "examples": [public(pair, True) for pair in examples],
        "blind": [public(pair, False) for pair in blind_strict],
        "counts": {"examples": len(examples), "blind": len(blind_strict)},
        "test301": False,
    }
    (ROOT / "artifacts/fewshot54_examples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (ROOT / "artifacts/fewshot54_blind.json").write_text(
        json.dumps({"status": "ждёт ручной разметки куратора", "items": payload["blind"]},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))
    print("медиана длины в примерах:",
          sorted(span["end"] - span["start"] + 1 for item in payload["examples"] for span in item["gold_spans"])[
              len(payload["examples"]) // 2])


def lookup_question(pair_id):
    """Вопрос хранится только в исходном CSV, в парах его нет."""
    from spanver.io_csv import read_gold

    prefix = pair_id[:4]
    table = {"sber": "sberquad-rag.csv", "rusc": "ruscibench-rag.csv"}
    for key, name in table.items():
        if pair_id.startswith(key):
            rows = list(read_gold(ROOT / "Dataset/1_rushallu_rag_gold" / name))
            index = int(pair_id[4:])
            if index < len(rows):
                return rows[index].get("query_text")
    return None


if __name__ == "__main__":
    main()
