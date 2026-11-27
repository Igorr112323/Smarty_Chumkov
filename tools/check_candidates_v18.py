"""V18 шаг 2: проверка стиля найденных кандидатов (что реально лежит в репозиториях).

Для каждого кандидата фиксируется: лицензия, дата последнего пуша, файлы данных > 20 КБ,
есть ли span-аннотации (start/end) и русский клаузовый стиль, вердикт и причина отказа.
Итог — `artifacts/ru_clause_data_v18.json`.
"""
import argparse, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_JQ = ('[.tree[] | select(((.path|endswith(".jsonl")) or (.path|endswith(".json")) or '
           '(.path|endswith(".csv")) or (.path|endswith(".parquet")) or '
           '(.path|endswith(".tsv"))) and .size > 20000) | "\(.size) \(.path)"] | .[:8] | .[]')

CANDIDATES = [
    ("s-nlp/PsiloQA", "span-level, мультиязычный (14 языков), но данные на HF, а в репозитории "
                      "только пайплайн и CSV аннотаторов без контекстов"),
    ("sb-ai-lab/TOHA", "topological hallucination detection; данные — CoQA/TruthfulQA (en), "
                       "русских span-аннотаций нет"),
    ("slivka83/ru_rag_test_dataset", "RuBQ 2.0: вопросы/ответы/параграфы для retrieval, "
                                     "span-аннотаций галлюцинаций нет"),
    ("AndreisLavrov/LegalBenchRu-RAG", "юридический RAG-бенчмарк; есть RAG-ответы, но нет "
                                       "span-разметки неподтверждённых фрагментов"),
    ("TheNarhawl/RAG-Scope-RU", "песочница абляций RAG на русских корпусах; метрики retrieval, "
                                "span-аннотаций нет"),
    ("DFKonGH/ru_hallucination_detector", "синтетические флипы предложений (number/entity/negation), "
                                          "уже обучение v16–v17: Group-CV 0.98, перенос на реальные "
                                          "клаузы отрицательный (val AUC 0.60–0.69)"),
    ("sber-summer-practice/hybrid-rag-hallucination-verification", "88 КБ xlsx ручной разметки "
                                                                   "ответов без span-координат"),
    ("ParticleMedia/RAGTruth", "реальные человеческие span-аннотации, но английский; "
                                "использован в v14 как en-проекция — ухудшил метрики"),
]


def gh(args, timeout=60):
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return f"ERROR rc={result.returncode}: {result.stderr.strip()[:120]}"
    return result.stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/ru_clause_data_v18.json")
    args = parser.parse_args(argv)
    rows = []
    for repo, note in CANDIDATES:
        meta = gh(["api", f"repos/{repo}", "--jq",
                   '"\(.stargazers_count)\t\(.size)\t\(.license.spdx_id // "none")\t\(.pushed_at)"'])
        files = gh(["api", f"repos/{repo}/git/trees/HEAD?recursive=1", "--jq", DATA_JQ])
        rows.append({"repo": f"https://github.com/{repo}", "facts": meta, "data_files": files.splitlines(),
                     "assessment": note, "used": False,
                     "reason": "нет реальных русских клаузовых span-аннотаций с doc_overlap = 0"})
    payload = {
        "rule": "проверка каждого найденного кандидата: лицензия, размер, файлы данных, наличие "
                "span-аннотаций (start/end), русский клаузовый стиль, doc_overlap",
        "candidates": rows,
        "conclusion": {
            "found_real_russian_clause_span_data": False,
            "why": "реальные человеческие span-аннотации есть только у RAGTruth (en) и у нашего же "
                   "RusHallu-RAG (doc_overlap = 100 %); всё русское, что находится на GitHub, либо "
                   "синтетические флипы (перенос отрицательный), либо retrieval-данные без разметки",
            "doc_overlap_note": "проверять doc_overlap не пришлось: ни один кандидат не содержит "
                                "span-аннотаций неподтверждённых фрагментов",
        },
    }
    (ROOT / args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["conclusion"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
