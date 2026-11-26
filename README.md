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

Python 3.10+, только стандартная библиотека в рантайме, из корня репозитория.

```
python -m spanver verify --answer answer.txt --document doc.txt
python -m spanver serve --port 8765
python -m spanver report
python tools/make_calibration.py
python tools/build_package.py
```

### Воспроизводимый ранкер (по умолчанию)

Лексический бэкенд по умолчанию использует двухкаскадный ранкер `spanver/ranker.py`:
логистическая модель кандидата (37 признаков) и гейт пары (9 признаков). Веса —
`spanver/ranker_weights.json` (2 643 байта, sha256 `a7b0e19a260fe2c3ffc2a6ee55eeb3c1e3315aa332d9bc858825c2b2b25106ed`),
пороги зафиксированы на train/val: gate 0,50, candidate threshold 0,60, max_spans 1.
Рантайм не требует numpy/scikit-learn; training-only версии зафиксированы в `requirements-train.txt`.

Воспроизведение полного цикла:

```
python -m spanver build --dataset Dataset --out artifacts
python tools/build_gold1200.py
python tools/train_ranker.py
python tools/build_gold1200_v2.py
python tools/cv_local_ranker.py
python tools/train_local_ranker.py
python tools/eval_ranker.py --gold artifacts/gold1200_v2.jsonl --report artifacts/ranker_eval_v2.json
python tools/speed_ranker.py
python tools/build_package.py
python -m pytest -q
```

`tools/build_gold1200.py` собирает 1 200 пар: 1 000 RusHallu-RAG + 200 удержанных
русских пар mFAVA_ru; ещё 48 пар mFAVA_ru идут только в дополнительный train.
Версия 2 (`tools/build_gold1200_v2.py`) использует позиционную конвертацию
mFAVA_ru и детерминированное деление 150 train / 50 held; оба набора сохраняются.

В ранкере два каскада: глобальный logistic-скрипт для RusHallu и локальный
`HistGradientBoostingClassifier`, экспортированный в JSON и работающий на
чистом Python (`spanver/local_ranker.py`). Гейт режима и пороги выбраны на
train/val/CV с учётом ожидаемой доли источников в удержанной выборке;
`artifacts/local_cv_report.json` и `artifacts/local_ranker_report.json`
содержат решётку и выбранные значения. `tools/eval_ranker.py` оценивает
удержанные сплиты без подбора порогов. Явный legacy-режим:
`verify(..., threshold=0.98, use_ranker=False)`.

Веб-интерфейс: `http://localhost:8765/` — поля «ответ» и «документ», кнопка «Проверить», таблица найденных фрагментов (координаты, риск, тип, confidence). Страница самодостаточна, внешних ресурсов не использует.

Переносимый пакет: `artifacts/spanver-0.2.0.zip` (523 986 байт, sha256 6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a, эталон в `package.sha256`), собирается `python tools/build_package.py`; в архив входят только файлы пакета `spanver/` (оба бэкенда и канонический сплит) и краткая инструкция, без данных, артефактов, тестов и кэша; архив собирается без сжатия, поэтому его sha256 не зависит от версии zlib и совпадает на Python 3.10 и 3.11.

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

## Модель

Модельный бэкенд работает на собственном маленьком энкодере, обученном с нуля в этой среде (pretrained-веса из HuggingFace/LFS недоступны):

| параметр | значение |
|---|---|
| файл | `models/local_encoder.pt` (обычный git-файл, не LFS) |
| размер | 1 920 837 байт (1,9 МБ) |
| параметров | 476 546 |
| архитектура | char-level энкодер: словарь 102 символа, dim 128, 3 слоя, 4 головы, ff 256 |
| обучение | train-сплит, 549 пар, 730 примеров, 6 эпох, seed 0, loss 0,454 |
| sha256 | записан в `models/local_encoder.sha256` |
| детерминизм | две независимые сборки дают побайтно одинаковый файл |

Признаки на токен из этой модели: `doc_attention_share`, `attention_entropy`, `cos_proximity`; плюс лексические `doc_overlap`, `uncertainty` и лексический риск как вход каскада. Агрегация mean/max, головка — 10 чисел.

## Как считаются метрики

Метрики считаются workflow **evaluate** (`.github/workflows/evaluate.yml`) на каждом пуше и по кнопке `workflow_dispatch`:

1. собирает данные (`tools/read_parquet.py`, `python -m spanver build`) и проверяет sha256 `gold.jsonl`;
2. берёт закоммиченную модель `models/local_encoder.pt` (и переобучает её, если файла нет), сверяет sha256 с `models/local_encoder.sha256`, отдельным шагом повторяет обучение и сравнивает байты;
3. собирает пакет и печатает его размер и sha256;
4. запускает `python tools/run_model_eval.py --weights models`: калибровка доли ИИ, головка, порог на val, test по `spanver/splits.json`, стресс 2 000 000 знаков;
5. печатает таблицу всех метрик договора в лог и в Summary рана, грузит модель, отчёт и пакет артефактами.

Чтобы посмотреть: вкладка **Actions** → workflow **evaluate** → последний ран → блок Summary. Там же внизу артефакт `model-eval`.

Workflow **ci** проверяет тесты на Python 3.10 и 3.11, размер пакета (≤ 614 400 байт) и sha256 пакета.

## Основано на

Модуль разработан на основе открытой архитектуры LettuceDetect (MIT), адаптированной для русского языка: русский энкодер rubert-tiny2, калибровка доли ИИ, 7 типов расхождений, признаки внимания. Принятые файлы, коммит и список изменений — в `THIRD_PARTY.md`, лицензия — `LICENSE-LETTUCEDETECT`.

## Приём весов модели

Веса приходят обычными git-файлами кусками (Git LFS из этой среды недоступен). Куски лежат в `weights/rubert-tiny2/`, собираются командой и проверяются по официальному sha256.

Команды на машине с весами (по одной строке):

```
chcp 65001
python tools\split_weights.py models\rubert-tiny2\pytorch_model.bin weights\rubert-tiny2
git add weights\rubert-tiny2 models\rubert-tiny2
git commit -m "Веса rubert-tiny2: куски и конфиг"
git push
```

Ожидаемые значения для cointegrated/rubert-tiny2:

| параметр | значение |
|---|---|
| размер pytorch_model.bin | 117 546 024 байта |
| полный sha256 | `137fa2b1d944dae19c74456dfe8fac2f780d9acf34e037f5d1e37acba1157768` |
| размер куска | 83 886 080 байт |
| число кусков | 2 |

Сборка в песочнице и в Actions:

```
python tools\assemble_weights.py weights\rubert-tiny2 models\rubert-tiny2
```

Сборщик сверяет sha256 каждого куска с `manifest.json`, склеивает файл, сверяет полный sha256 с официальным и при любом расхождении файл не пишет. Собранный `models/rubert-tiny2/pytorch_model.bin` в git не попадает, куски и мелкие файлы (`config.json`, `vocab.txt` и прочие) — попадают.

## Восстановление чекпоинтов

Обученные чекпоинты в git не попадают: git-lfs в песочнице недоступен (GitHub release assets закрыты), а файл весов 117 МБ больше лимита GitHub в 100 МБ на файл. Вместо весов коммитятся манифест `artifacts/checkpoints/<имя>/manifest.json` с sha256, рецепт обучения и голова модели (`head.pt`, 1,3 МБ), которая вместе с базой из `weights/rubert-tiny2` даёт точку восстановления.

Порядок восстановления:

```
chcp 65001
python tools\assemble_weights.py weights\rubert-tiny2 models\rubert-tiny2
python tools\train_cv_feat.py --folds 5 --epochs 1 --report artifacts\cv_report_feat5.json
```

Обучение детерминированно (seed 0): повторный прогон дал те же метрики по фолдам 1-3 (token F1 0,4234 / 0,5973 / 0,5693). Время на 2 ядрах — 824 с на 5 фолдов по 1 эпохе.

## Соответствие договору

Таблица соответствия собирается из двух отчётов (`artifacts/module_report.json`, `artifacts/model_eval_report.json`) и печатается первой в Summary workflow **evaluate**:

```
python -m spanver compliance --markdown
```

Файл `artifacts/compliance.json` содержит пункты 1–9 договора со статусом (`выполнено`, `частично`, `не выполнено`, `нет данных`) и числами, взятыми только из отчётов. Бэкенд по умолчанию — `lexical`.

## Что осталось

| задача | состояние |
|---|---|
| веса rubert-tiny2 (`weights/rubert-tiny2`) | ждём куски от куратора, инструменты и workflow готовы |
| обучение span-детектора на rubert-tiny2 | после весов: smoke ≤ 200 пар, затем полный прогон на 1624 парах |
| переключение дефолтного бэкенда на модель | только если val F1 модели > 0,2595 (val F1 лексического) |
| F1 ≥ 0,50 на test | не достигнут, текущий договорный факт 0,2835 (лексический бэкенд) |

## Состояние работ

Текущее состояние, фактические метрики и открытые вопросы ведутся в `STATUS.md`.
