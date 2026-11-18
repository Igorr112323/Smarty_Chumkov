"""ПРОМПТ 56, стадия 2: подготовка входа для схемы 52 на `docs`.

Основная идея: скрипты схемы 52 (`train51_ten.py`, `collect51_ten.py`,
`ceiling52.py`, `speed52.py`) получают необязательный `--pairs`. По умолчанию
он равен `artifacts/ours33.jsonl`, то есть прежнее поведение не меняется и
цифры отчёта 52 остаются воспроизводимыми. Новый вход подаётся явно.

Когда `--pairs` указывает не на `ours33.jsonl`, проверка «sha256 файла равен
зафиксированному в nested38_summary» заменяется проверкой меток: набор
`pair_id` и все человеческие метки обязаны совпасть с `ours33`. Это не
ослабление контроля, а его перенос: сверяется то, что действительно не должно
меняться (метки и состав пар), а меняется ровно `document_text`.

Этот файл — проверка входа перед запуском: он падает, если в новом файле
изменилось что-то, кроме `document_text`/`doc_ids`.

Запуск: python3 tools/entry56.py --pairs artifacts/ours56_docs.jsonl
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from tools.preflight_13 import id_digest
from tools.prepare_ours33 import EXPECTED_IDS

REFERENCE = ROOT / "artifacts/ours33.jsonl"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_rows(rows):
    return {
        row["pair_id"]: {
            "model_output": row["model_output"],
            "spans": [(span["start"], span["end"], span["type"], span["type_src"], span["text"])
                      for span in row["spans"]],
        }
        for row in rows
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="artifacts/ours56_docs.jsonl")
    parser.add_argument("--reference", default=str(REFERENCE))
    parser.add_argument("--report", default="artifacts/entry56.json")
    args = parser.parse_args()

    source = ROOT / args.pairs
    reference = Path(args.reference)
    if not source.exists():
        raise SystemExit(f"нет файла {source}; сначала tools/rebuild56.py")
    new = read_jsonl(source)
    old = read_jsonl(reference)

    if len(new) != 699 or sum(len(row["spans"]) for row in new) != 300:
        raise SystemExit(f"неверное число пар или спанов: {len(new)}")
    if id_digest(new) != EXPECTED_IDS:
        raise SystemExit("не совпал sha256 набора pair_id")

    new_labels = label_rows(new)
    old_labels = label_rows(old)
    if set(new_labels) != set(old_labels):
        raise SystemExit("изменился состав пар")
    mismatched = [pair_id for pair_id in new_labels if new_labels[pair_id] != old_labels[pair_id]]
    if mismatched:
        raise SystemExit(f"изменены человеческие метки в {len(mismatched)} парах: {mismatched[:5]}")

    changed_documents = sum(1 for row in new
                            if row["document_text"] != {r["pair_id"]: r for r in old}[row["pair_id"]]["document_text"])

    report = {
        "status": "готово",
        "entry": "docs",
        "pairs_arg": str(args.pairs),
        "sha256": digest(source),
        "sha256_reference": digest(reference),
        "pairs": len(new),
        "spans": sum(len(row["spans"]) for row in new),
        "id_sha256": id_digest(new),
        "labels_identical": True,
        "documents_changed": changed_documents,
        "document_length_median": sorted(len(row["document_text"]) for row in new)[len(new) // 2],
        "ok_to_run": True,
        "test301": False,
    }

    rebuild = ROOT / "artifacts/rebuild56.json"
    if rebuild.exists() and str(args.pairs).endswith("ours56_docs.jsonl"):
        declared = json.loads(rebuild.read_text(encoding="utf-8"))["c5_sha256"]
        report["rebuild56_sha256"] = declared
        report["rebuild56_match"] = declared == report["sha256"]
        if not report["rebuild56_match"]:
            raise SystemExit("sha256 не совпал с опубликованным rebuild56.json")

    (ROOT / args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
