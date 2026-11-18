"""ПРОМПТ 56, критерий 1.8: few-shot материалы на новом входе (`docs`).

Отличия от 54:

  * `document` — новый `document_text` (склейка всех пяти документов `docs`);
  * `question` — из `query_text` исходного CSV;
  * `instruction` — ДОСЛОВНЫЕ определения шести типов, извлечённые из поля `prompt`
    в `Dataset/2_rushallu_rag_pairs/sberquad-train-rag.parquet`. Своих формулировок
    правил нет: текст копируется из блока `## Hallucination Types` без правок.

12 примеров с эталонными спанами и 30 слепых пар набираются так, чтобы множества
`doc_id` не пересекались; слепые берутся только из train/val, test 301 не трогается.
Слепые пары не содержат ни спанов, ни типов.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from spanver.io_csv import read_gold
from tools.preflight_13 import our_keys

EXAMPLES = 12
BLIND = 30
TYPE_BLOCK_START = "## Hallucination Types"
TYPE_BLOCK_END = "## Multi-Label Detection"
TYPE_NAMES = ("Contradiction", "Unconfirmed", "Missing", "Excess", "Partial", "Oversight")
QUESTIONS = {"sber": "sberquad-rag.csv", "rusci": "ruscibench-rag.csv"}


def type_definitions():
    """Дословный текст определений шести типов из поля `prompt`."""
    import pyarrow.parquet as pq

    table = pq.ParquetFile(ROOT / "Dataset/2_rushallu_rag_pairs/sberquad-train-rag.parquet")
    prompt = table.read_row_group(0).column("prompt")[0].as_py()
    start = prompt.find(TYPE_BLOCK_START)
    end = prompt.find(TYPE_BLOCK_END)
    if start < 0 or end < 0:
        raise ValueError("в prompt нет блока определений типов")
    block = prompt[start:end]
    # контроль: все шесть названий присутствуют в извлечённом блоке
    missing = [name for name in TYPE_NAMES if name not in block]
    if missing:
        raise ValueError(f"в блоке нет типов: {missing}")
    return block.strip() + "\n", prompt


def load_questions():
    table = {}
    for prefix, name in QUESTIONS.items():
        rows = list(read_gold(ROOT / "Dataset/1_rushallu_rag_gold" / name))
        table[prefix] = [row.get("query_text") or "" for row in rows]
    return table


def question_for(pair_id, table):
    prefix = "sber" if pair_id.startswith("sber") else "rusci"
    index = int(pair_id[4:])
    rows = table.get(prefix) or []
    return rows[index] if index < len(rows) else None


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--examples", default="artifacts/fewshot56_examples.json")
    parser.add_argument("--blind", default="artifacts/fewshot56_blind.json")
    args = parser.parse_args()

    pairs = read_jsonl(ROOT / args.pairs)
    positives = [pair for pair in pairs if pair["spans"]]
    definitions, _ = type_definitions()
    questions = load_questions()

    used = set()
    examples = pick(positives, EXAMPLES, used)
    chosen_ids = {pair["pair_id"] for pair in examples}

    blind = []
    for pair in pairs:
        if pair["pair_id"] in chosen_ids:
            continue
        keys = {key for key in our_keys(pair)}
        if keys & used:
            continue
        blind.append(pair)
        used |= keys
        if len(blind) == BLIND:
            break

    def public(pair, with_gold):
        item = {
            "pair_id": pair["pair_id"],
            "question": question_for(pair["pair_id"], questions),
            "document": pair["document_text"],
            "answer": pair["model_output"],
            "doc_ids": list(pair["doc_ids"]),
        }
        if with_gold:
            item["gold_spans"] = [
                {
                    "start": span["start"],
                    "end": span["end"],
                    "text": pair["model_output"][span["start"]:span["end"] + 1],
                    "type": span["type"],
                    "human_type": span.get("type_src"),
                }
                for span in pair["spans"]
            ]
        return item

    payload = {
        "status": "готово",
        "entry": "склейка всех документов поля docs: '## <doc_id>\\n<doc_text>', разделитель \\n\\n",
        "instruction_source": "Dataset/2_rushallu_rag_pairs/sberquad-train-rag.parquet, поле prompt, "
                              "блок '## Hallucination Types' — дословно",
        "instruction": definitions,
        "examples": [public(pair, True) for pair in examples],
        "blind": [public(pair, False) for pair in blind],
        "counts": {"examples": len(examples), "blind": len(blind)},
        "test301": False,
    }

    example_ids = {pair["pair_id"] for pair in examples}
    blind_ids = {pair["pair_id"] for pair in blind}
    example_docs = {ident for pair in examples for ident in pair["doc_ids"]}
    blind_docs = {ident for pair in blind for ident in pair["doc_ids"]}
    payload["checks"] = {
        "pair_id_overlap": len(example_ids & blind_ids),
        "doc_id_overlap": len(example_docs & blind_docs),
        "blind_items_with_spans": sum(1 for item in payload["blind"] if item.get("gold_spans")),
        "questions_present": sum(1 for item in payload["examples"] for _ in [0] if item["question"]),
        "examples_without_question": sum(1 for item in payload["examples"] if not item["question"]),
        "blind_without_question": sum(1 for item in payload["blind"] if not item["question"]),
        "median_example_span": sorted(
            span["end"] - span["start"] + 1
            for item in payload["examples"] for span in item["gold_spans"]
        )[max(0, len([1 for item in payload["examples"] for _ in item["gold_spans"]]) // 2)],
        "test301": False,
    }
    if payload["checks"]["pair_id_overlap"] or payload["checks"]["doc_id_overlap"]:
        raise ValueError(f"пересечение непустое: {payload['checks']}")

    (ROOT / args.examples).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (ROOT / args.blind).write_text(
        json.dumps({"status": "ждёт ручной разметки куратора", "instruction": definitions,
                    "items": payload["blind"]}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))
    print(json.dumps(payload["checks"], ensure_ascii=False))
    print("медиана длины документа в примерах:",
          sorted(len(item["document"]) for item in payload["examples"])[len(payload["examples"]) // 2])


if __name__ == "__main__":
    main()
