import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.sentence_snap import sentence_intervals


def classify(row):
    text = row["answer"]
    sentences = sentence_intervals(text)
    starts = {start: index for index, (start, _) in enumerate(sentences)}
    ends = {end: index for index, (_, end) in enumerate(sentences)}
    result = Counter()
    for span in row["gold_spans"]:
        left = span["start"]
        right = span["end"]
        if not 0 <= left <= right < len(text):
            raise ValueError("gold-спан вне границ ответа")
        first = starts.get(left)
        last = ends.get(right)
        count = last - first + 1 if first is not None and last is not None and last >= first else 0
        result[count] += 1
    return result


def summarize(counts):
    total = sum(counts.values())
    single = counts[1]
    double = counts[2]
    triple = counts[3]
    four_or_more = sum(number for length, number in counts.items() if length >= 4)
    return {"spans": total, "single_sentence": single, "two_sentences": double, "three_sentences": triple, "four_or_more_sentences": four_or_more, "not_sentence_aligned": counts[0], "single_percent": 100 * single / total if total else None, "one_neighbor_total": single + double, "one_neighbor_percent": 100 * (single + double) / total if total else None, "one_neighbor_each_side_total": single + double + triple, "one_neighbor_each_side_percent": 100 * (single + double + triple) / total if total else None}


def main():
    manifest = json.loads((ROOT / "artifacts/predictions40_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "baseline nested38 recovered, not round40" or len(manifest["outputs"]) != 5:
        raise ValueError("ожидались пять фолдов старой baseline")
    totals = Counter()
    folds = []
    seen = set()
    for index, item in enumerate(manifest["outputs"], 1):
        raw = (ROOT / item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("SHA предсказаний не совпал")
        rows = [json.loads(line) for line in raw.splitlines() if line]
        if len(rows) != item["pairs"] or any(row["source"] != "nested38_baseline" or row["fold"] != index for row in rows):
            raise ValueError("неверное происхождение фолда")
        counts = Counter()
        for row in rows:
            if row["pair_id"] in seen:
                raise ValueError("повтор пары во внешних фолдах")
            seen.add(row["pair_id"])
            counts.update(classify(row))
        totals.update(counts)
        folds.append({"fold": index, "pairs": len(rows), **summarize(counts)})
    if len(seen) != 699 or sum(totals.values()) != 300:
        raise ValueError("ожидались 699 train/val пар и 300 gold-спанов")
    summary = summarize(totals)
    output = {"status": "порог одного соседнего предложения не достигнут; раунд не запускался", "source": "пять сохранённых внешних фолдов nested38, только наши train+val", "pairs": len(seen), "protocol": "Границы gold неизменные, включительные. Границы предложения заданы SENTENCE_BREAK из spanver/sentence_snap.py: знак . ! ? … перед пробелом или перевод строки; краевые пробелы предложений исключены. Точное совпадение начала и конца с объединением последовательных предложений. Два предложения означают одно добавленное слева ИЛИ справа; три — по одному с обеих сторон. Направление одного соседа не определяется без выбранного базового предложения.", "counts_by_number_of_sentences": {str(key): value for key, value in sorted(totals.items())}, "summary": summary, "folds": folds, "single_neighbor_threshold_percent": 50, "single_neighbor_threshold_reached": (summary["one_neighbor_total"] * 2 >= summary["spans"]), "both_sides_threshold_reached": (summary["one_neighbor_each_side_total"] * 2 >= summary["spans"])}
    target = ROOT / "artifacts/sentence_alignment41.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
