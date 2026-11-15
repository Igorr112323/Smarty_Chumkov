import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFECTS = {"rtru-11672": "точка перед продолжением «включая»", "rtru-4985": "точка перед продолжением «и»", "rtru-6609": "точка перед продолжением «известен»", "rtru-15830": "точка перед продолжением «и»", "rtru-5756": "точка перед продолжением «но»", "rtru-9439": "точка перед продолжением «с»", "rtru-15758": "точка перед продолжением «и»", "rtru-2138": "точка перед продолжением «повторяя»", "rtru-4742": "точка перед продолжением «призывая»", "rtru-11102": "нет пробела: «обслуживаниеи»", "rtru-4101": "двойная точка на границе", "rtru-17151": "точка перед продолжением «или»", "rtru-9477": "точка перед продолжением «и»"}


def main():
    path = ROOT / "Dataset/RAGTruth_ru/ragtruth_ru.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    sample = random.Random(31).sample(rows, 30)
    if not set(DEFECTS).issubset({row["pair_id"] for row in sample}):
        raise ValueError("ручная разметка не относится к выбранным парам")
    report = []
    lines = ["# Промт 31: ручной осмотр стыков RAGTruth-ru", "", "Случайно выбраны 30 разных позитивных пар из 6000, seed 31. Показаны все спаны каждой пары и до 40 символов с каждой стороны; если ответ заканчивается раньше, показан доступный остаток. Вертикальная черта обозначает границу, исходный текст не менялся.", "", "Строго по указанным механическим категориям — нет пробела, слипание слов или обрыв слова — явный дефект у 1 из 30 пар (3,3%): rtru-11102. Дополнительно у 12 из 30 пар видны ошибочная точка перед синтаксическим продолжением или двойная точка. Вместе 13 из 30 (43,3%), если включать такие пунктуационные разрывы. Это ручная оценка одного проверяющего, не доказательство причины качества модели; оценка перевода целиком не проводилась.", ""]
    for index, row in enumerate(sample, 1):
        text = row["model_output"]
        spans = []
        for span in row["spans"]:
            left, right = span["start"], span["end"] + 1
            spans.append({"left": text[max(0, left - 40):left], "span": text[left:right], "right": text[right:right + 40], "start": left, "end": right - 1})
        defect = DEFECTS.get(row["pair_id"])
        report.append({"pair_id": row["pair_id"], "spans": spans, "defect": defect, "strict_mechanical": row["pair_id"] == "rtru-11102"})
        lines.append(f"## {index}. {row['pair_id']} — {defect or 'явного дефекта стыка не отмечено'}")
        lines.append("")
        for value in spans:
            left = json.dumps(value["left"], ensure_ascii=False)
            middle = json.dumps(value["span"], ensure_ascii=False)
            right = json.dumps(value["right"], ensure_ascii=False)
            lines.append(f"{left} | {middle} | {right}")
            lines.append("")
    (ROOT / "artifacts/seams31.json").write_text(json.dumps({"seed": 31, "source": "6000 позитивных пар RAGTruth-ru", "pairs": 30, "strict_mechanical_pairs": 1, "punctuation_pairs": 12, "combined_pairs": 13, "sample": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "artifacts/seams31.md").write_text("\n".join(lines), encoding="utf-8")
    print("проверено пар 30, механические дефекты 1, дополнительные пунктуационные 12")


if __name__ == "__main__":
    main()
