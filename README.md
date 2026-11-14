# Smarty_Chumkov

Программный модуль span-уровневой верификации контента с определением доли участия ИИ для корпоративных систем. Проект № 117812.

Модуль принимает русскоязычный ответ нейросети и документ-источник и возвращает перечень неподтверждённых фрагментов ответа: координаты первого и последнего символа, риск 0–1, тип расхождения, уверенность 0–1, а также интегральную оценку доли участия ИИ 0–1. Работа полностью локальная: без видеокарты, без сетевых обращений, без внешних API.

Репозиторий содержит датасеты для разработки и оценки модуля (папка `Dataset`), документацию по состоянию работ (`STATUS.md`) и, по мере готовности, исходный код модуля, сборку и контрольные суммы.

## Датасеты

В репозитории 6 папок датасетов с нумерацией 1, 2, 3, 5, 6, 7 (папки 4 нет) и 10 файлов данных.

| путь | что за данные | записей | span-разметка |
|---|---|---|---|
| Dataset/1_rushallu_rag_gold/sberquad-rag.csv | RusHallu-RAG: вопрос, документ-источник, ответ модели, ручная span-разметка галлюцинаций | 655 | есть, 175 фрагментов |
| Dataset/1_rushallu_rag_gold/ruscibench-rag.csv | RusHallu-RAG на аннотациях eLibrary: вопрос, документ-источник, ответ модели, ручная span-разметка | 345 | есть, 248 фрагментов |
| Dataset/2_rushallu_rag_pairs/sberquad-train-rag.parquet | пары «вопрос — документ — ответ модели» без разметки | 1 626 | нет |
| Dataset/2_rushallu_rag_pairs/ruscibench-train-rag.parquet | пары без разметки, файл в Git LFS, 53 301 312 байт | не проверено, файл локально не получен | нет |
| Dataset/3_rufacts/rufacts_train.json | RuFacts: пара «источник — утверждение», бинарная метка искажения | 4 677 | нет, метка 1 / 0 |
| Dataset/3_rufacts/rufacts_val.json | то же, валидационная часть | 1 559 | нет, метка 1 / 0 |
| Dataset/5_ruscifact/ruscifact_train.parquet | ruSciFact: аннотация статьи и утверждение, бинарная метка | 564 | нет, метка 1 / 0 |
| Dataset/5_ruscifact/ruscifact_val.parquet | то же, вторая часть | 564 | нет, метка 1 / 0 |
| Dataset/6_sberquad/sberquad_train.parquet | SberQuAD: контекст, вопрос, ответ и позиция ответа; 9 078 уникальных контекстов | 45 328 | нет |
| Dataset/7_ruscibench/ru_sci_bench_ru.parquet | RuSciBench: аннотации eLibrary, единственный источник длинных документов; файл в Git LFS, 149 013 239 байт | не проверено, файл локально не получен | нет |

Суммарный объём файлов данных в репозитории — 57 873 992 байта без учёта содержимого LFS-объектов и 260 188 543 байта с ним.

Разметка RusHallu-RAG устроена так: поле `answer` содержит JSON-массив объектов вида `{"type": ..., "span": ...}`; у 667 пар из 1 000 массив равен `[{"span": null, "type": null}]`, то есть расхождений не найдено. Всего размечено 423 фрагмента: Unconfirmed — 116, Partial — 125, Contradiction — 68, Missing — 56, Oversight — 36, Excess — 22. Все 423 span находятся в поле `model_output` точным поиском подстроки.

Длины документов (`joined_reference`, `context`, `ru_abstract`, `evidence`) в доступных файлах: максимум 17 241 знак, записей длиннее 100 000 знаков нет.

## Хранение больших файлов

Файлы больше 100 МБ и часть парных parquet-файлов хранятся в Git LFS. Перечень задан в `.gitattributes`.

Добавление или замена файла:

```
git lfs install
git lfs track "Dataset/путь/к/файлу.parquet"
git add .gitattributes "Dataset/путь/к/файлу.parquet"
git commit -m "Обновлён файл данных"
git push origin main
```

Получение файлов после клонирования:

```
git lfs install
git lfs pull --include="Dataset/**"
```

Контроль целостности:

```
git lfs ls-files --long
sha256sum Dataset/7_ruscibench/ru_sci_bench_ru.parquet
```

Хэш sha256 полученного файла должен совпадать с полем `oid` в файле-указателе.

Предупреждение: файлы `Dataset/2_rushallu_rag_pairs/ruscibench-train-rag.parquet` (133 байта) и `Dataset/7_ruscibench/ru_sci_bench_ru.parquet` (134 байта) в дереве репозитория выглядят как мусор, но это нормально. Это указатели Git LFS с хэшем и реальным размером объекта; сами данные лежат в LFS-хранилище и подставляются клиентом при `git lfs pull`. Считать признаком порчи файла размер меньше 100 КБ можно только для файлов, не перечисленных в `.gitattributes`.

## Использование

Python 3.10+, только стандартная библиотека, из корня репозитория.

```
python -m spanver verify --answer answer.txt --document doc.txt
python -m spanver serve --port 8765
python -m spanver report
python tools/make_calibration.py
python tools/build_package.py
```

Веб-интерфейс: `http://localhost:8765/` — поля «ответ» и «документ», кнопка «Проверить», таблица найденных фрагментов (координаты, риск, тип, confidence). Страница самодостаточна, внешних ресурсов не использует.

Переносимый пакет: `artifacts/spanver-0.2.0.zip` (43 025 байт, sha256 cde7945292b4b23fc8a0a35574118a0d0d6720060f6f521ccbbe6e2e7e36cdb8), собирается `python tools/build_package.py`; в архив входят только файлы пакета `spanver/` (оба бэкенда и канонический сплит) и краткая инструкция, без данных, артефактов, тестов и кэша.

## Модельный режим

Два бэкенда: `lexical` (по умолчанию, только стандартная библиотека, пакет ≤ 0,6 МБ) и `model` (torch + transformers + локальные веса `ai-forever/ruSciBERT`). Импорт torch/transformers выполняется только внутри модельных функций, поэтому лексический режим и переносимый пакет работают без них.

Модельный режим считает три договорных признака внимания — `doc_attention_share`, `attention_entropy`, `cos_proximity` — и добавляет к ним лексические `doc_overlap` и `uncertainty`. Агрегация во фрагмент — mean и max по токенам, головка получает 10 чисел. Документ не гоняется целиком: на фрагмент берутся top-2 сегмента по лексическому совпадению (≤ 256 токенов), на пару — не более 50 кандидатов.

Канонический сплит зафиксирован в `spanver/splits.json` (test 301, val 637, train 1624), оценка идёт строго по нему.

### Команды на машине с весами

```
chcp 65001
git pull
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface_hub
python -m spanver build --dataset Dataset --out artifacts
huggingface-cli download ai-forever/ruSciBERT --local-dir models\ruSciBERT
python tools\run_model_eval.py --weights models\ruSciBERT --limit 20
python tools\run_model_eval.py --weights models\ruSciBERT
```

Скрипт `tools/run_model_eval.py` проверяет sha256 `artifacts/gold.jsonl` (должен быть `fddd62ce05baf028b64cb6db81b89053bdc5444a540c6e18e34e318577c8b32b`), пересобирает калибровочные смеси, считает модельные признаки с кэшем `artifacts/model_cache.jsonl`, обучает головку на подмножестве train (≤ 400 пар), выбирает порог на полном val (сетка 0,50…0,99, FPR ≤ 0,40), оценивает test по splits.json, снимает стресс на 2 000 000 знаков документа и ответа и печатает `artifacts/model_eval_report.json`. Полный прогон — 1–3 часа.

Проверка после прогона:

```
python -m spanver verify --backend model --weights models\ruSciBERT --answer answer.txt --document doc.txt
python -m spanver serve --port 8765 --backend model --weights models\ruSciBERT
```

## Состояние работ

Текущее состояние, фактические метрики и открытые вопросы ведутся в `STATUS.md`.
