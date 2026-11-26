# Финальный отчёт по проекту № 117812 (УМНИК)

Дата: 2026-11-25. Ветка: `arena/963b6d7a-smarty-chumkov`. Код зафиксирован коммитом `d97b06317426e33259ca0c25afa79dcdf165851d` («Добавить воспроизводимый ранкер, набор 1200 пар и отчёты оценки»); правка размера весов в README — коммитом `7cbeb52b57d88c10611233638ff5ddb057e75630`, сам отчёт зафиксирован отдельным коммитом после него. Исходная точка — `e1ec868a681f7686c630dd68e06b7117a94f9d42`.

## 1. Executive summary

Завершён автономный цикл: прочитан код, восстановлен gold-набор, собран набор из **1 200 пар**, обучен и интегрирован двухкаскадный лексический ранкер без внешних зависимостей в рантайме, зафиксированы веса/пороги, пересобран воспроизводимый package, прогнаны тесты и замеры скорости.

**Главная цель договора не достигнута.** Лучший честный результат на удержанных 501 паре (301 RusHallu-RAG `test` + 200 mFAVA_ru `extra`): **span F1 = 0,0928 при FPR_pair = 0,2995**; token F1 = 0,4351, pair F1 = 0,2200. На каноническом `test301`: span F1 = 0,2903, token F1 = 0,4945, pair F1 = 0,3750 при FPR = 0,2995. Требуется F1 ≥ 0,50 при FPR ≤ 0,40 — разрыв по span F1 в 5,4 раза на удержанных 501 паре.

При этом все инженерные требования, проверяемые в этой среде, закрыты: offline/stdlib, формат `spans` + `ai_share`, калибровка 0/0,25/…/1, CLI/HTTP/Web, package 294 889 байт с единственным SHA-256, скорость 50k 402,8 мс и 2M 12,1115 с. `.exe` не собран: в среде нет Windows-цели, Wine и PyInstaller.

## 2. Метрики до/после

Все строки получены запуском `tools/eval_ranker.py` на одном и том же наборе `artifacts/gold1200.jsonl`, IoU ≥ 0,5. «Legacy» — зафиксированный лексический бэкенд `verify(..., threshold=0.98, use_ranker=False)`; «Ranker» — по умолчанию `spanver/ranker.py` с порогами из весов. Пороги подбирались только на train/val, удержанные сплиты в подборе не участвуют.

| Сплит | Режим | Пар | Span P/R/F1 | Token P/R/F1 | Pair F1 | FPR_pair |
|---|---|---:|---|---|---|---|
| test301 | legacy (до) | 301 | — / — / **0,2835** | 0,2856 / 0,6259 / **0,3923** | 0,3547 | 0,3527 |
| test301 | ranker (после) | 301 | 0,2880 / 0,2927 / **0,2903** | 0,3617 / 0,7817 / **0,4945** | 0,3750 | **0,2995** |
| extra200 | legacy (до) | 200 | — / — / **0,0056** | 0,2541 / 0,9981 / **0,4051** | 0,0198 | 0,0000 |
| extra200 | ranker (после) | 200 | 0,0435 / 0,0155 / **0,0229** | 0,2656 / 0,9407 / **0,4143** | 0,0769 | 0,0000 |
| held501 | legacy (до) | 501 | — / — / **0,0786** | 0,2610 / 0,8738 / **0,4020** | 0,1877 | 0,3527 |
| held501 | ranker (после) | 501 | 0,1424 / 0,0689 / **0,0928** | 0,2881 / 0,8876 / **0,4351** | 0,2200 | **0,2995** |

Дополнительные подтверждённые строки:
- val150 после обучения: span F1 **0,3364**, token F1 **0,6252**, pair F1 **0,4444**, FPR **0,1635**, gate 0,50 / candidate threshold 0,60 / max_spans 1 (`artifacts/ranker_report.json`).
- Исторический model-бэкенд на test301: span F1 0,1274 при FPR 0,087; не смешивается с таблицей выше (`artifacts/module_report.json`, README).
- Скорость ранкера: 1 000 знаков — 8,593 мс/1000; 10 000 — 5,606; 50 000 — **402,8 мс**; 2 000 000 — **12,1115 с** (6,056 мс/1000). Лимиты 1 с / 60 с / 30 мс на 1000 соблюдены (`artifacts/speed_ranker.json`).
- Сборка weights воспроизводима: повторный `tools/train_ranker.py` дал тот же SHA-256 `a7b0e19a…06ed`.

## 3. Список изменений по файлам

| Файл | Изменение |
|---|---|
| `spanver/ranker.py` (новый) | Рантайм-ранкер на stdlib: 37 признаков кандидата, 9 признаков гейта, NMS, калиброванный риск; читает веса из JSON; опция `collect_support` для `ai_share` без повторного пересчёта. |
| `spanver/ranker_weights.json` (новый) | Обученные веса logistic-кандидата и gate, seed 0, зафиксированные пороги gate=0,50 / threshold=0,60 / max_spans=1 / nms_iou=0,10. |
| `spanver/module.py` | `verify()` по умолчанию вызывает ранкер; legacy-режим сохранён через `threshold=...`/`use_ranker=False`; `_finish()` переиспользует support кандидатов и не ломает калибровку `ai_share`. |
| `tools/build_gold1200.py` (новый) | Сборка 1 000 gold + 200 held mFAVA и 48 train-only mFAVA; проверка непересечения doc_id; SHA-256 обоих файлов. |
| `tools/train_ranker.py` (новый) | Обучение на train + 48 mFAVA, подбор порогов на val, запись весов и `artifacts/ranker_report.json` с версиями/seed/решёткой. |
| `tools/eval_ranker.py` (новый) | Финальная оценка test301/extra200/held501 без подбора параметров, сравнение с legacy, запись `artifacts/ranker_eval.json`. |
| `tools/speed_ranker.py` (новый) | Замер 1k/10k/50k/2M знаков в stdlib-режиме. |
| `tests/test_ranker.py` (новый) | Формат весов, запуск `verify()` под `python -S` без numpy/sklearn, детерминизм, границы риска, legacy-режим. |
| `tests/test_module.py` | Порог в schema-тесте согласован с конфигурацией ранкера. |
| `requirements-train.txt` (новый) | training-only пины: numpy 1.26.4, scikit-learn 1.9.1, pytest 9.2.0. |
| `README.md`, `STATUS.md`, `artifacts/compliance.json` | Обновлены команды, sha пакета (294 889 байт), факты оценки и статус цели. |
| `package.sha256`, `artifacts/spanver-0.2.0.zip`, `artifacts/package.json` | Пакет пересобран детерминированно и содержит новый ранкер/веса. |
| `.gitignore` | Разрешены новые воспроизводимые артефакты. |

## 4. Команды воспроизведения

```bash
git checkout d97b06317426e33259ca0c25afa79dcdf165851d
python -m venv .venv
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python -m spanver build --dataset Dataset --out artifacts
.venv/bin/python tools/build_gold1200.py
.venv/bin/python tools/train_ranker.py
.venv/bin/python tools/eval_ranker.py
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
.venv/bin/python -m pytest -q
```

Среда прогона: Python 3.11.2, numpy 1.26.4, scikit-learn 1.9.1, pytest 9.2.0. smoke-проверка CLI/HTTP: `python -m spanver verify --answer a.txt --document d.txt` и `python -m spanver serve --port 8765` (страница `http://localhost:8765/`). Рантайм требует только stdlib Python 3.10+.

Сплиты: `spanver/splits.json` → train 549 / val 150 / test 301 (document-held-out, `artifacts/gold.jsonl` sha `fddd62ce…c8b32b`); дополнительно train_extra 48 / held_extra 200 из mFAVA_ru. Пересечения doc_id между gold и held — 0 (`artifacts/gold1200_report.json`).

## 5. Артефакты с SHA-256 и размером

| Путь | Байты | SHA-256 |
|---|---:|---|
| `artifacts/gold.jsonl` | 5 315 573 | `fddd62ce05baf028b64cb6db81b89053bdc5444a540c6e18e34e318577c8b32b` |
| `artifacts/gold1200.jsonl` | 7 785 304 | `7a27f08b33249ede2bd07a3c952162c62341f4f78ac222fbb779118fc2ff9b1c` |
| `artifacts/gold1200_report.json` | 379 | `f7243337a42f5b5bed171c9a0f284b24e010a42956e725010455523ab8dbf808` |
| `artifacts/mfava_train_extra.jsonl` | 633 186 | `46b6e342d667f6de482a6005763d7dc6bb9a05db35bf56ee5ca0cf365262ee09` |
| `spanver/ranker_weights.json` | 2 643 | `a7b0e19a260fe2c3ffc2a6ee55eeb3c1e3315aa332d9bc858825c2b2b25106ed` |
| `artifacts/ranker_report.json` | 4 279 | `a517cde30d6cbafbd7c81c663a4c823cd0be8ed95f038cc9a6203e4aac545c7f` |
| `artifacts/ranker_eval.json` | 4 504 | `f4a12d9167303dd781da31af0f04a4b21ab08bc03c24b4104ccab17ef024ad60` |
| `artifacts/speed_ranker.json` | 751 | `0fddafdc2557e53812d06deb02fb3e499eaece650aca71821cc480ef61900dfe` |
| `artifacts/spanver-0.2.0.zip` | 294 889 | `5e5ad5d66621c212b24e31e5ff63394af19f58071ee1f38c3ac8810e2dae033c` |
| `artifacts/package.json` | 1 353 | `fef5938d968ca2c9cf64d0eab5764ad639c82d3a7e36c3501aba31733c1db0c0` |
| `package.sha256` | 84 | `f6a03fa339b18021aefd412fad93c60d9fbeeb26b76bc93dc40df4779698ff4d` |
| `artifacts/compliance.json` | 3 508 | `47e3d231573174640a3ea97ac306ef43394271c3168f4df016ffc2145257d82d` |
| `spanver/ranker.py` | 7 987 | `8fb31514e1357b21c29b745aceb7095f79b483e3ef87ef69f24022cd449bfa62` |
| `tools/train_ranker.py` | 7 392 | `a0ec9b49f1f3837684a8d247a2935a4f8b6824d3824a06e731becbcb4e564a8e` |
| `tools/eval_ranker.py` | 2 976 | `09df1125bea4f129debfa9029e0f32b7ad0ec1ff891db08651f71223c00d4e8a` |
| `requirements-train.txt` | 218 | `0b548c398b4aba2101ff7b6ff213e1020efc52d302d87eb876963aa86172355e` |

SHA в `package.sha256` совпадает с фактическим SHA пакета; тест `tests/test_package.py` проверяет побайтовую воспроизводимость сборки.

## 6. Тесты

- `.venv/bin/python -m pytest -q` → **141 passed in 17.03s** на коммите `d97b063` (после правки README changed только docs; код не менялся).
- Проверено в тестах: схема ответа и диапазоны, детерминизм, пустой ответ, пять признаков на токен, калибровка шагом 0,25 и монотонность, CLI (два сценария), HTTP `/verify`, повторяемость package и его SHA, новый рангер.
- `tests/test_ranker.py` запускает `verify()` через `python -S` и проверяет, что `numpy`/`sklearn` не импортируются: рантайм остаётся автономным.
- Legacy-режим `verify(..., threshold=0.98, use_ranker=False)` сохранён и покрыт тестом.

## 7. Риски и блокеры

1. **Качество (главный блокер).** Цель F1 ≥ 0,50 при FPR ≤ 0,40 не достигнута: held501 span F1 0,0928. На extra200 span F1 0,0229: для длинных ответов mFAVA ранкер часто выбирает весь ответ (token recall 0,9407, но span IoU с локальными эталонами мал). Oracle coverage кандидатов на train_extra — 86,6%, то есть проблема ранжирования/границ, а не только генерации кандидатов.
2. **Метрическая ловушка.** Высокий token recall на extra200 создаёт видимость качества, но contract-локализация (IoU ≥ 0,5) провалена; в отчёте приведены все три уровня.
3. **Данные.** Два parquet-файла в `Dataset` — LFS-указатели, содержимое не выгружено; оценка опирается на 1 000 real gold + 200 mFAVA. `artifacts/model_eval_report.json` отсутствует в репозитории; модельная строка в таблице — историческая, не новая.
4. **Окружение.** `.exe` и Windows-испытания не выполнены: нет Wine/PyInstaller/Windows-цели. Поставляется portable Python package ≤ 0,6 МБ.
5. **Зависимости.** scikit-learn нужен только для переобучения; версии зафиксированы, но `.venv` не входит в snapshot.
6. **РИД/заявка** не готовилась по прямому указанию пользователя.

## 8. Вердикт

**ЧАСТИЧНО ГОТОВО.**

| Критерий | Оценка 0–10 | Обоснование |
|---|---:|---|
| Качество по заданию | **3/10** | FPR и интерфейсы в норме, но F1 далёк от 0,50; честный результат 0,0928 на 501 удержанной паре. |
| Инженерная готовность | **8/10** | stdlib-рантайм, воспроизводимые веса/сплиты/SHA, 141 тест, package и speed-лимиты; нет `.exe`. |
| Готовность к сдаче | **5/10** | можно сдавать как работающий прототип с честным разрывом по F1; для приёмки по ТЗ нужен прорыв в локализации mFAVA-спанов + Windows-сборка. |

Рекомендуемый следующий шаг (не выполнен в этом сегменте): обучить сегментную модель/sequence labeling для внутрипредложных исправлений mFAVA и провести blind-прогон на тех же 501 парах, сохранив пороги train/val-only.
