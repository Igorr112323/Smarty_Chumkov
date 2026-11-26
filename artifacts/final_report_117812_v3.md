# Финальный отчёт № 117812 — цикл 3 (research iteration)

Дата 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`. Текущий HEAD перед фиксацией отчёта: `fa9ba1f` (восстановление состояния цикла 2: среда была пересоздана, история коммитов сброшена до `e1ec868`, файлы сохранены в workspace). Ниже — честный результат третьего цикла: зафиксированного улучшения качества нет, все новые гипотезы проверены на train/val/CV и отклонены.

## 1. Вердикт

**Договорная цель НЕ ЗАКРЫТА полностью.** На договорном document-held-out наборе v2 (1 200 пар, held351):

- Span F1 = **0,3253** при FPR = 0,3043 — не закрыто;
- Token F1 = **0,4990** при FPR = 0,3043 — не закрыто (не хватило 0,001);
- Pair F1 = **0,5000** при FPR = 0,3043 — закрыто на границе.

FPR ≤ 0,40 соблюдён на всех уровнях. Формат, офлайн-рантайм, lexical-калибровка `ai_share`, скорость, интерфейсы и пакет остаются рабочими; калибровка модельного бэкенда — не закрыта. `.exe` не собран: нет Windows/PyInstaller/Wine.

## 2. Таблица метрик до/после

Все уровни посчитаны скриптом `tools/eval_ranker.py`, IoU ≥ 0,5. V1 — старый набор 1200 (held501), V2 — текущий договорной набор (held351). Ниже P / R / F1.

| Набор | Уровень | V1 | V2 (текущий) | FPR V1 / V2 |
|---|---|---|---|---|
| test301 | Span | 0,2880 / 0,2927 / **0,2903** | 0,2353 / 0,2927 / **0,2609** | 0,2995 / 0,3043 |
| test301 | Token | 0,3617 / 0,7817 / **0,4945** | 0,3540 / 0,7047 / **0,4713** | 0,2995 / 0,3043 |
| test301 | Pair | 0,3673 / 0,3830 / **0,3750** | 0,3636 / 0,3830 / **0,3731** | 0,2995 / 0,3043 |
| held501 / held351 | Span | 0,1424 / 0,0689 / **0,0928** | 0,3154 / 0,3357 / **0,3253** | 0,2995 / 0,3043 |
| held501 / held351 | Token | 0,2881 / 0,8876 / **0,4351** | 0,3891 / 0,6954 / **0,4990** | 0,2995 / 0,3043 |
| held501 / held351 | Pair | 0,4151 / 0,1497 / **0,2200** | 0,5227 / 0,4792 / **0,5000** | 0,2995 / 0,3043 |
| extra200 / extra50 | Span | 0,0435 / 0,0155 / **0,0229** | 0,4000 / 0,3694 / **0,3841** | 0 / 0 |
| extra200 / extra50 | Token | 0,2656 / 0,9407 / **0,4143** | 0,4699 / 0,6800 / **0,5557** | 0 / 0 |
| extra200 / extra50 | Pair | 1,0000 / 0,0400 / **0,0769** | 1,0000 / 0,6600 / **0,7952** | 0 / 0 |

Источник: `artifacts/ranker_eval_v1.json`, `artifacts/ranker_eval_v2.json`. Новый held-out в этом цикле не прогонялся: конфигурация не менялась, поэтому зафиксированный результат цикла 2 остаётся действительным. Повторный прогон на `artifacts/gold1200_v2.jsonl` дал те же цифры.

## 3. Что изменилось и что проверено

Код модели, веса, пороги и сплиты **не менялись**: все новые гипотезы проиграли baseline на train/val-диагностике. Добавлен только артефакт error analysis `artifacts/error_analysis_v3.json`.

Диагностика (только train/val/mFAVA-train):
- oracle candidate recall: train 226/245 = 92,2%; val 46/55 = 83,6%; mFAVA-train 332/398 = 83,4%;
- короткие gold-спаны < 40 знаков: oracle recall train 44%, val 36%, mFAVA 35% — главный источников потерь;
- длинные gold-спаны: oracle recall ~100%, но scoring часто выбирает клаузу целиком;
- FP на отрицательных парах: 127 train, 19 val;
- long answer > 500: whole-answer предсказан в 68/113 train и 11/20 val (где это часто корректно по gold).

Проверенные и отклонённые гипотезы ( weighted val×6 + mFAVA-train; baseline span/token/pair/FPR = 0,4731 / 0,6519 / 0,5995 / 0,1635):

| Гипотеза | Лучший weighted span F1 | Причина отказа |
|---|---:|---|
| Новые token-window кандидаты + переобучение голов | 0,4359 | ниже baseline по всем трём уровням |
| Boundary refinement по лучшему короткому кандидату | 0,4426 | ниже baseline, ухудшает val |
| Добавление коротких кандидатов к уже выбранным спанам | 0,4439 | ниже baseline, FPR вырос до 0,2788 |
| Token-level logistic | 0,2140 | границы нестабильны |
| Token-level GBDT | 0,2326 | границы нестабильны |
| Замороженный ruBERT-скоринг кандидатов | 0,3800 | ниже baseline |
| Fine-tuned ruBERT token classifier (2 эпохи) | 0,0811 | FPR 0,9519, неприменим |
| Merge соседних спанов | 0,2713 | ломает multi-span mFAVA |

Подробности — `artifacts/error_analysis_v3.json`. Ни одна гипотеза не дала права менять зафиксированную конфигурацию, поэтому честно оставляем результат цикла 2.

## 4. Доказательства

Ключевые команды этого цикла:

```
.venv/bin/python -m spanver build --dataset Dataset --out artifacts
.venv/bin/python tools/eval_ranker.py --gold artifacts/gold1200_v2.jsonl --report /tmp/repro_v2.json
.venv/bin/python /tmp/error_analysis.py   # прототип, сохранённый результат в artifacts/error_analysis_v3.json
.venv/bin/python -m pytest -q
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
```

Среда после пересоздания: Python 3.11.2, numpy 1.26.4, scikit-learn 1.9.1, torch 2.0.1+cu117, transformers 4.38.2, pytest 9.2.0.

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `artifacts/error_analysis_v3.json` | 16 232 | `1c0d567d20581af2a0d4c44892094b99a0fb6b6ae94d3ed4c17efcf6f4da8398` |
| `artifacts/gold1200_v2.jsonl` | 7 865 949 | `28e21f0ef95cd1b2a5ac3cf5f4f4a83fb663c51fe951e56b0e9b7a017400af4e` |
| `spanver/ranker_weights.json` | 227 338 | `853902cb78ae27b9b0f1c8e94b0f6bd2c5af7ffb26455f0de7caad6babc12eac` |
| `artifacts/ranker_eval_v2.json` | 4 499 | `bdf60010ba859b98bd2ef65ea859d88032bbb8ade56045a651f85a501ef8673c` |
| `artifacts/local_ranker_report.json` | 29 056 | `137436cade928cb0be46111293521ed895f1b798903e5a79b0d2815aecf2a583` |
| `artifacts/local_cv_report.json` | 1 447 | `0eef5ec2f91ff2e957ceb5178c8850cbafb6caa97a4aa94d24a994e3bcefc785` |
| `artifacts/speed_ranker.json` | 750 | `4f50c4df5218a9af0c0678d8c8a0146b4c513e7311c2fd79835481ae427f081a` |
| `artifacts/spanver-0.2.0.zip` | 523 986 | `6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a` |
| `package.sha256` | 84 | `41dd0d0e4b0ff28c673ed7249dfe740360c21c99d586189e883bd172b8e22d8f` |

## 5. Тесты, скорость, пакет, интерфейсы

- `.venv/bin/python -m pytest -q` → **142 passed in 24.90s**.
- Скорость: 1k — 9,453 мс/1000; 50k — **0,3815 с** (≤ 1 с); 2M — **11,5397 с** (≤ 60 с), 5,770 мс/1000.
- Пакет: 523 986 байт ≤ 614 400; SHA совпадает с `package.sha256`, сборка воспроизводима.
- CLI `python -m spanver verify`, HTTP `/verify`, web `http://localhost:8765/` покрыты тестами.
- Калибровка `ai_share` шагом 0,25: lexical 5 точек, max|Δ| = 0, LOO = 0,25 — принято; модельный бэкенд 5 точек, max|Δ| = 0,7524, LOO = 0,75 — **не закрыто**. Источник: `artifacts/compliance.json` (п. 4, статус «частично»).
- Рантайм не требует numpy/sklearn/torch: тест с `python -S` проходит.

## 6. Остаточные риски

1. Главный: span/token F1 ниже 0,50 при корректном FPR. Блокер — качество границ и коротких спанов: oracle recall коротких спанов 35–44%, а обученные скоринг/sequence-labeling подходы не улучшили baseline на train/val/CV.
2. Pair F1 ровно 0,5000 — на границе; любые будущие изменения могут его просесть.
3. `.exe` не собран из-за отсутствия Windows/PyInstaller/Wine; доступен только Python-пакет.
4. `artifacts/model_eval_report.json` отсутствует;модельный бэкенд не пересобирался.
5. Два parquet-файла в `Dataset` остаются LFS-указателями.

РИД/заявка не готовилась по указанию пользователя.
