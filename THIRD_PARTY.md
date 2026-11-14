# Сторонний код

## LettuceDetect

| параметр | значение |
|---|---|
| проект | LettuceDetect |
| URL | https://github.com/KRLabsOrg/LettuceDetect |
| зафиксированный коммит | dcce161ff7c8a46147cae5f0bda14118bef7a2d6 (2026-11-09) |
| лицензия | MIT, текст в `LICENSE-LETTUCEDETECT` |
| правообладатель | Copyright (c) 2025 KR Labs |

Приняты четыре файла:

| файл в репозитории | файл в проекте |
|---|---|
| `spanver/lt/base.py` | `lettucedetect/detectors/base.py` |
| `spanver/lt/transformer.py` | `lettucedetect/detectors/transformer.py` |
| `spanver/lt/trainer.py` | `lettucedetect/models/trainer.py` |
| `spanver/lt/evaluator.py` | `lettucedetect/models/evaluator.py` |

Остальные части LettuceDetect (клиенты LLM, LangChain, генерация синтетических данных, API, CLI, типологическая головка) не переносились.

### Что изменено

1. В начало каждого файла добавлен блок из трёх строк с указанием правообладателя, лицензии MIT, URL проекта и зафиксированного коммита. Оригинальные тексты файлов не сокращались.
2. Импорты `lettucedetect.*` заменены на `spanver.lt.*`; вместо отсутствующих модулей проекта используется наш слой `spanver/lt/compat.py` (`Lang`, `LANG_TO_PASSAGE`, `PromptUtils`, `HallucinationDataset.prepare_tokenized_input`, `HallucinationSample`, `HallucinationDetector`).
3. В `evaluator.py` импорт `sklearn.metrics` сделан отложенным (через `sklearn_metrics()`), чтобы пакет не требовал sklearn.
4. В `transformer.py` импорт типологической головки заменён на явную ошибку: эта часть не переносится.
5. Метрики качества считаются нашим `spanver/evaluate.py` по правилу договора (IoU ≥ 0,5 без условия типа); `evaluator.py` оставлен как референс архитектуры оценки.

### Данные

Синтетические данные LettuceDetect (HallucinationGenerator, препроцессинг RAGTruth/RAGBench) не использовались. Обучение — только на gold-парах репозитория.
