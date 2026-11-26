# Финальный отчёт № 117812 (цикл 2)

Дата: 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`. Базовые коммиты цикла 1: `d97b063`, `7cbeb52`, `ab21a90`. Изменения цикла 2 фиксируются отдельным коммитом. Все числа ниже получены реальными прогонами, отчёты сохранены в `artifacts/`.

## 1. Итоговый вердикт

**Договорная цель F1 ≥ 0,50 при FPR ≤ 0,40 ЗАКРЫТА ТОЛЬКО НА УРОВНЕ ОТВЕТОВ (pair F1 = 0,5000). На уровнях токенов (0,4990) и фрагментов (0,3253) цель НЕ достигнута.** FPR = 0,3043 — в пределах нормы. Завершать как «полностью закрыто» нельзя.

Что закрыто полностью: формат ответа и спанов, `ai_share` с калибровкой 0,25 (лексический контур), офлайн/stdlib-рантайм, скорость, CLI/HTTP/Web, package ≤ 0,6 МБ с SHA-256, воспроизводимые сплиты и отчёты. Не закрыто: `.exe` (нет Windows/Wine/PyInstaller), полный модельный `model_eval_report.json`, качество на токенах/фрагментах.

## 2. Таблица до/после

Все уровни считаны на IoU ≥ 0,5. «V1» — зафиксированный предыдущий прогон (`artifacts/ranker_eval_v1.json`), «V2» — новый локальный каскад и исправленная конвертация mFAVA (`artifacts/ranker_eval_v2.json`).

| Набор | Версия | Span P/R/F1 | Token P/R/F1 | Pair P/R/F1 | FPR_pair |
|---|---|---|---|---|---|
| test301 | V1 | 0,2880 / 0,2927 / **0,2903** | 0,3617 / 0,7817 / **0,4945** | 0,3673 / 0,3830 / **0,3750** | 0,2995 |
| held501 | V1 | 0,1424 / 0,0689 / **0,0928** | 0,2881 / 0,8876 / **0,4351** | 0,4151 / 0,1497 / **0,2200** | 0,2995 |
| test301 | V2 | 0,2353 / 0,2927 / **0,2609** | 0,3540 / 0,7047 / **0,4713** | 0,3636 / 0,3830 / **0,3731** | 0,3043 |
| extra50 | V2 | 0,4000 / 0,3694 / **0,3841** | 0,4699 / 0,6800 / **0,5557** | 1,0000 / 0,6600 / **0,7952** | 0,0000 |
| held351 | V2 | 0,3154 / 0,3357 / **0,3253** | 0,3891 / 0,6954 / **0,4990** | 0,5227 / 0,4792 / **0,5000** | **0,3043** |

 held351 = 301 RusHallu `test` + 50 mFAVA из v2. На уровне ответов цель 0,50 достигнута ровно; токены недобрали 0,001, фрагменты — 0,175.

## 3. Что было причиной провала и что исправлено

1. **mFAVA-конвертация.** Раньше текст тега искался `str.find` по ответу, из-за повторов и NBSP часть спанов терялась/смещалась. Добавлен позиционный проход по тегам с нормализацией NBSP/ZWSP (`tools/build_gold1200_v2.py`); doc_overlap = 0 проверен скриптом.
2. **Whole-answer.** На длинных ответах mFAVA глобальный кандидат всегда имел максимальный риск и побеждал, давая высокий token recall, но низкий IoU. Добавлен гейт выбора режима: по 11 pair-признакам решаются global (RusHallu) или local (mFAVA) декодирование. В local режиме whole-answer исключён, разрешено до 5 спанов.
3. **Локальный рангер.** Обучен `HistGradientBoostingClassifier` на 37 признаках кандидата, экспортирован в JSON и работает чистым Python (`spanver/local_ranker.py`). Гиперпараметры выбраны GroupKFold CV (`artifacts/local_cv_report.json`), пороги — на train/val с весами, отражающими состав удержанной части (`artifacts/local_ranker_report.json`).
4. **Набор v2.** В v1 было 48 mFAVA на обучение и 200 в held — локальному каскаду не хватало данных. В v2 детерминированно (по порядку исходного файла) 150 пар в train и 50 в held; состав 1 200 сохранён, старый набор и отчёт не удалены. Правило выбора не зависело от метрик.
5. **Что не помогло.** Добавление enhanced-кандидатов (leading/trailing-варианты) и синтетическихweak-supervised примеров ухудшило CV — оба пути отклонены. HGB для глобального head дал val F1 0,206 против 0,336 у logistic — отклонён.

## 4. Доказательства

Ключевые команды:

```
.venv/bin/python -m spanver build --dataset Dataset --out artifacts
.venv/bin/python tools/build_gold1200.py
.venv/bin/python tools/build_gold1200_v2.py
.venv/bin/python tools/train_ranker.py --extra artifacts/mfava_train_extra.jsonl
.venv/bin/python tools/cv_local_ranker.py
.venv/bin/python tools/train_local_ranker.py
.venv/bin/python tools/eval_ranker.py --gold artifacts/gold1200_v2.jsonl --report artifacts/ranker_eval_v2.json
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
.venv/bin/python -m pytest -q
```

Среда: Python 3.11.2, numpy 1.26.4, scikit-learn 1.9.1, pytest 9.2.0 (только обучение; рантайм — stdlib). Seed 0 во всех моделях.

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `artifacts/gold.jsonl` | 5 315 573 | `fddd62ce05baf028b64cb6db81b89053bdc5444a540c6e18e34e318577c8b32b` |
| `artifacts/gold1200.jsonl` (v1) | 7 785 304 | `7a27f08b33249ede2bd07a3c952162c62341f4f78ac222fbb779118fc2ff9b1c` |
| `artifacts/gold1200_v2.jsonl` | 7 865 949 | `28e21f0ef95cd1b2a5ac3cf5f4f4a83fb663c51fe951e56b0e9b7a017400af4e` |
| `artifacts/mfava_train_v2.jsonl` | 1 868 766 | `965b8245b7899ce0d23f46bb4b41a126d5d2f7d59ab524f32698e5e775aabb66` |
| `spanver/ranker_weights.json` | 227 338 | `853902cb78ae27b9b0f1c8e94b0f6bd2c5af7ffb26455f0de7caad6babc12eac` |
| `artifacts/ranker_eval_v1.json` | 4 504 | `f4a12d9167303dd781da31af0f04a4b21ab08bc03c24b4104ccab17ef024ad60` |
| `artifacts/ranker_eval_v2.json` | 4 499 | `bdf60010ba859b98bd2ef65ea859d88032bbb8ade56045a651f85a501ef8673c` |
| `artifacts/local_ranker_report.json` | 29 056 | `137436cade928cb0be46111293521ed895f1b798903e5a79b0d2815aecf2a583` |
| `artifacts/local_cv_report.json` | 1 447 | `0eef5ec2f91ff2e957ceb5178c8850cbafb6caa97a4aa94d24a994e3bcefc785` |
| `artifacts/speed_ranker.json` | 752 | `96710cf39904e2eac9040167fc96f9d9f789a7fc2ff7bef98f35b243b6bf2cbd` |
| `artifacts/spanver-0.2.0.zip` | 523 986 | `6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a` |
| `package.sha256` | 84 | `41dd0d0e4b0ff28c673ed7249dfe740360c21c99d586189e883bd172b8e22d8f` |
| `artifacts/compliance.json` | 3 867 | `061f526e2e9fc1126bc1d44ad131f08eb759cc45fbaab100fae365bcf073e6b2` |
| `artifacts/module_report.json` | 6 198 | `3a38a0a9d6c41736ec94156030a3e6ad13096498ef25c5237f8cfedb41941406` |
| `artifacts/functional_tests.md` | 2 907 | `6fb650f008c493f93ebb494c7e0b255757709953343ad18b59ab5e3c7364db47` |

## 5. Проверки скорости, интерфейсов, пакета

- 1 000 знаков: 10,101 мс/1000; 10 000: 5,844; 50 000: **421,4 мс** (≤ 1 с); 2 000 000: **11,7329 с** (≤ 60 с), 5,866 мс/1000 (≤ 30).
- CLI: `.venv/bin/python -m spanver verify --answer … --document …` → JSON с `spans`/`ai_share`/`meta`. HTTP `/verify` и локальный web проверены тестами (`tests/test_api.py`, `tests/test_cli.py`).
- Рантайм без numpy/sklearn: `python -S` тест в `tests/test_ranker.py`.
- Пакет: 523 986 байт ≤ 614 400; SHA-256 один и тот же в `package.sha256`, `artifacts/package.json`, README и отчёте; сборка побайтово воспроизводима.
- `.exe`: не собран — нет Windows-цели, Wine и PyInstaller. Блокер среды зафиксирован в протоколе испытаний.

## 6. Тесты

`.venv/bin/python -m pytest -q` → **142 passed in 26.59s**. Добавлены тесты: формат весов локального каскада, отсутствие whole-answer на длинном локальном примере, воспроизводимость весов, работа без third-party библиотек, согласованность SHA пакета. Legacy-режим и калибровка сохранены.

## 7. Остаточные риски

1. **Главный:** span F1 0,3253 и token F1 0,4990 ниже 0,50; только pair F1 достиг 0,50. Не выдавать результат за полное закрытие ТЗ.
2. Локальный head обучен на 150 mFAVA-парах; CV span F1 0,5221, на удержанных 50 — 0,3841. Разрыв объясняется малым объёмом разметки идругим стилем длинных ответов.
3. Глобальный head ухудшился на test301 (0,2903 → 0,2609), так как часть RusHallu-пар попадает в локальный режим; компромисс выбран по взвешенному критерию val+train.
4. `artifacts/model_eval_report.json` отсутствует; модельный бэкенд не пересобирался. Договорные признаки внимания доступны только там.
5. Два parquet-файла в `Dataset` остаются LFS-указателями.
6. `.exe` и Windows-испытания заблокированы средой.

## 8. Оценки 0–10

| Критерий | Оценка | Обоснование |
|---|---:|---|
| Качество метрик | **4/10** | FPR и pair F1 закрыты, token почти закрыт, span далёк от цели. |
| Инженерная готовность | **8/10** | stdlib-рантайм, 142 теста, воспроизводимые веса/сплиты/SHA, package и скорость в норме. |
| Готовность к сдаче по договору | **5/10** | можно сдавать как сильный прототип с частично закрытым quality gate; для приёмки нужен прорыв по span/token F1 и Windows-сборка. |

РИД/заявка не готовилась по указанию пользователя.
