# Финальный отчёт № 117812 — цикл 9 (смена подхода: evidence-pointer + OOF CV)

Дата 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`, HEAD на старте `ceab5cc`. Воспроизведены v8-артефакты и guard; новый held-out **не запускался**, потому что ни одна конфигурация не прошла val/CV. Ручной разметки пользователем не было. РИД/заявка не готовилась.

## 1. Вердикт по договорным пунктам

| Пункт | Статус | Факт |
|---|---|---|
| Span F1 ≥ 0,50 @ FPR ≤ 0,40 | ❌ | Договорной held351 (v2): **0,3253 @ 0,3043**; лучший честный dev/CV в v9: **0,4712 @ 0,0000** (mFAVA OOF), но val RusHallu **0,3248** |
| Token F1 ≥ 0,50 @ FPR ≤ 0,40 | ⚠️ | held351 **0,4990** (−0,0010); v9 OOF **0,5394**, val **0,6455** |
| Pair F1 ≥ 0,50 @ FPR ≤ 0,40 | ✅ на границе | held351 **0,5000**; v9 OOF **0,8527** |
| IoU ≥ 0,5 | ✅ | во всех отчётах evaluate; `iou_threshold=0.5` |
| 1200 пар / doc_overlap = 0 | ✅ | `gold1200_v3.jsonl`, manifest, v1/v2 сохранены |
| 142 теста / пакет / скорость / CLI / HTTP / web / offline | ✅ | 142 passed; 523 986 B; 0,3798 с / 11,416 с; package SHA `6bbc5044…8ff7d73a` |
| ai_share шаг 0,25 | ⚠️ | lexical max\|Δ\|=0, LOO=0,25; model max\|Δ\|=0,7524, LOO=0,75 (partially) |
| `.exe` | ⛔ | нет Windows/Wine/PyInstaller; build script/инструкция остаются |

**Итог: цель не закрыта. Блокер-отчёт.** Никакой тюнинг по held-out не производился.

## 2. Воспроизведение v8 и аудит потолка

`HEAD ceab5cc`, отчёт `artifacts/final_report_117812_v8.md`, SHA `531a443d…738e61`. Воспроизведены: strict CV `guard_pass=false`, `dev_combined 0,3422/0,4658/0,5786 @ 0,1731`; cached sweep 525 конфигураций — лучший Span `0,3621`, про guard — 0.

Новый аудит `tools/audit_ceiling_v9.py` → `artifacts/audit_ceiling_v9.json` отвечает на вопрос «где именно теряется Span F1»:

| Сплит | Кандидаты | Oracle recall | Oracle Span F1 | Реальный Span F1 | Медианный ранг gold |
|---|---|---:|---:|---:|---:|
| train | baseline / v4 | 0,922 / 0,996 | 0,960 / 0,998 | 0,271 | 2 |
| val | baseline / v4 | 0,836 / **1,000** | 0,911 / **1,000** | 0,271 | 3–7 |
| mFAVA | baseline / v4 | 0,834 / **1,000** | 0,910 / **1,000** | 0,644 | 8–27 |

**Вывод:** генерация кандидатов и границы не являются потолком — при идеальном выборе Span F1 ≈ 1,0. Потолок — **ранжирование**: top-1 gold-кандидата = 0 на всех сплитах. Поэтому бессмысленно ещё tuning порогов/NMS/max_spans; нужна другая модель выбора границ.

## 3. Новые эксперименты (все отдельными артефактами)

### E1. Evidence-conditioned start/end pointer (rubert-tiny2)

`tools/train_neural_pointer_v9.py`, `tools/eval_neural_pointer_v9.py`.

Архитектура: `rubert-tiny2` (3 слоя, hidden 312) + head на конкатенации CLS-представления кандидата и CLS релевантного evidence-фрагмента документа. Обучение: 300 пар RAGTruth_ru + 549 RusHallu train + 148 mFAVA v3, 12 кандидатов на пару (gold + hard negatives по baseline logit), 2 эпохи, lr 2e−5, batch 16, max_len 160, pos_weight≈2,35, 11 920 строк / 4 799 позитивов. Чекпоинт `artifacts/neural_pointer_v9/best/model.safetensors` SHA `8b5cbf9fa413bb5a3e7bf420a735a498fb34e9590d4939bc82234a938a062bee`, `head.pt` SHA `a6a576d3…1d97`. В пакет не включён: он 117 МБ при лимите 0,6 МБ; рантайм остаётся stdlib.

Диагностика: при инференсе shortlist из baseline-ранкера сохраняет только **28%** gold-кандидатов на val и **5%** на mFAVA. Поэтому оценена версия на полном множестве `candidates_v4` (149 065 кандидатов на dev). Лучший конфиг `threshold=0,97, max_spans=5`:

| Набор | Span F1 | Token F1 | Pair F1 | FPR |
|---|---:|---:|---:|---:|
| val + mFAVA dev | 0,3328 | 0,4842 | 0,7081 | **0,4327** |
| RusHallu val | 0,2390 | 0,5034 | 0,3455 | 0,4327 |

Guard не пройден (FPR > 0,40, Span/Token/Pair ниже цели). **Отклонено.**

### E2. Честный 3-fold OOF CV вместо однодолевого CV v8

`tools/cv3_oof_v9.py` → `artifacts/cv3_oof_v9.json`, `artifacts/cv3_oof_predictions_v9.json`.

В v8 каждый фолд видел 55 mFAVA-пар, тогда как финальная модель — 148. Здесь каждый фолд обучается на 93/104/99 парах (2/3) и оценивается **только на своей доле**; OOF-предсказания агрегируются.

| Фолд | Train mFAVA | Span F1 | Token F1 | Pair F1 | FPR |
|---|---:|---:|---:|---:|---:|
| 0 | 93 | 0,4706 | 0,5215 | 0,8660 | 0,0000 |
| 1 | 104 | 0,4661 | 0,5854 | 0,8267 | 0,0000 |
| 2 | 99 | 0,4766 | 0,5159 | 0,8605 | 0,0000 |
| **OOF aggregate** | — | **0,4712** | **0,5394** | **0,8527** | **0,0000** |

Это лучший честный результат проекта по mFAVA, но **RusHallu val остаётся 0,3248/0,6455/0,4634 @ 0,1635** во всех трёх фолдах. Комбинированный guard «все три ≥0,50 на val/CV» не пройден из-за Span. **Отклонено как финальная конфигурация**; артефакты сохранены как доказательство прогресса.

### E3. GBDT с реальными фразовыми спанами RAGTruth_ru

`tools/train_ragtruth_aug_v9.py` → `artifacts/rag_aug_v9_report.json`, `artifacts/ranker_weights_rag_aug_v9.json`.

Источник: 600 пар RAGTruth_ru, 7 830 span, медиана 41 символ; 600 пар использованы в обучении; 29 131 строка кандидатов (3 342 позитива); held-документы исключены (doc_overlap=0). Отличается от v7: реальные ошибки в реальном контексте, а не тривиальные подмены токенов. Экспорт деревьев совпал со sklearn (`max|Δ| < 1e−6`).

Лучший конфиг `threshold=0,85, max_spans=3`: dev `Span 0,0164 / Token 0,0988 / Pair 0,0747 @ FPR 0,3654`. **Резко отклонено** — признаки/цель RAGTruth не переносятся на RusHallu-style границы, модель обучается различать слишком простой сигнал и не находит длинные clause-границы.

## 4. До / после по всем held-out версиям

Новый held-out прогон не делался. Ниже — зафиксированные цифры v1/v2 из `artifacts/final_report_117812_v8.md`.

| Набор · уровень | V1 P/R/F1 | V2 P/R/F1 | FPR V1→V2 |
|---|---|---|---|
| test301 · Span | 0,2880/0,2927/0,2903 | 0,2353/0,2927/0,2609 | 0,2995→0,3043 |
| test301 · Token | 0,3617/0,7817/0,4945 | 0,3540/0,7047/0,4713 | 0,2995→0,3043 |
| test301 · Pair | 0,3673/0,3830/0,3750 | 0,3636/0,3830/0,3731 | 0,2995→0,3043 |
| held501→351 · Span | 0,1424/0,0689/0,0928 | 0,3154/0,3357/**0,3253** | 0,2995→0,3043 |
| held501→351 · Token | 0,2881/0,8876/0,4351 | 0,3891/0,6954/**0,4990** | 0,2995→0,3043 |
| held501→351 · Pair | 0,4151/0,1497/0,2200 | 0,5227/0,4792/**0,5000** | 0,2995→0,3043 |
| extra200→50 · Span | 0,0435/0,0155/0,0229 | 0,4000/0,3694/0,3841 | 0→0 |
| extra200→50 · Token | 0,2656/0,9407/0,4143 | 0,4699/0,6800/0,5557 | 0→0 |
| extra200→50 · Pair | 1,0000/0,0400/0,0769 | 1,0000/0,6600/0,7952 | 0→0 |

## 5. Данные, лицензии, правила разметки

| Источник | Скрипт | Размер | SHA-256 | doc_overlap | Правило |
|---|---|---:|---|---:|---|
| `artifacts/gold.jsonl` train/val | `tools/build_gold1200_v3.py` | 699 пар | в v3 manifest | 0 | C1-гармонизация `tools/convert_labels_v3.py` |
| `artifacts/mfava_train_v2.jsonl` | тот же | 148 пар после исключения 2 документов | в v3 manifest | 0 | C1 + исключение по SHA документа |
| `artifacts/gold1200_v3.jsonl` | `tools/build_gold1200_v3.py` | 1 200 пар, 351 held | `910398513…a9e5` | 0 | 271 span, v1/v2 сохранены |
| `Dataset/RAGTruth_ru/ragtruth_ru.jsonl` | `tools/translate_ragtruth.py` (происхождение в `STATUS.md`) | 6 000 пар / 7 830 span / 2 635 документов | `4acb2988…8800ef` | 0 с held | готовые автоматические метки; оригинальный RAGTruth — MIT [1](https://github.com/ParticleMedia/RAGTruth) |
| `artifacts/synthetic_short_v7.jsonl` | `tools/build_synthetic_v7.py` | 6 308 пар | в v7 manifest | 0 | отклонено в v7, не использовалось |

Старые v1/v2/v3/v4/v5/v6/v7 manifests и данные сохранены рядом; ничего не удалено.

## 6. Артефакты и команды

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `artifacts/audit_ceiling_v9.json` | 2 740 | `36eefd06ea8916f7b4136b35976e72e73400e6ee63a3faa4f2fb4fe612aa2299` |
| `artifacts/neural_pointer_v9/report.json` | 10 361 | `55c27843ed48eb366c36e4f9631a77745887e4cc42fff1c526cdd1b553644502` |
| `artifacts/neural_pointer_v9/eval_full_dev.json` | 15 012 | `7b49264cf94ac8c4ad04539d6e445404ef177cfcaaee80a5de42d8e2ec1a5829` |
| `artifacts/neural_pointer_v9/best/head.pt` | 782 797 | `a6a576d312a392d6e8af79627da3bee73714f47deb425ad300482f6e3e1d1d97` |
| `artifacts/cv3_oof_v9.json` | 1 713 | `09c54b8f87d105e93471172371e88eb2ad9a8754d4ee520c63369390b6cb99e4` |
| `artifacts/cv3_oof_predictions_v9.json` | 416 589 | `ccc6fce4676035ede8c3a3b758da1d81813ade28914b4cffea0b24ba6ea8afc0` |
| `artifacts/rag_aug_v9_report.json` | 13 898 | `596266fe431012f5c89eb39bee4b087cc4c6bc8858bb38499ae8cf9c540e8a8d` |
| `artifacts/ranker_weights_rag_aug_v9.json` | 315 062 | `455e96df586165c5153d1e41ab96206ce65b9fc1dd4feb7d2507b1ff42f4a7c4` |

```
.venv/bin/python tools/audit_ceiling_v9.py
.venv/bin/python tools/train_neural_pointer_v9.py --rag-limit 300 --epochs 2 --batch 16
.venv/bin/python tools/eval_neural_pointer_v9.py
.venv/bin/python tools/cv3_oof_v9.py
.venv/bin/python tools/train_ragtruth_aug_v9.py --rag-pairs 600
.venv/bin/python -m pytest -q
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
```

## 7. Тесты / скорость / пакет / интерфейсы / ai_share

- `pytest -q`: **142 passed in 25.26 s**
- 50k: **0,3798 с**; 2M: **11,416 с**; 5,708 мс/1000 символов
- Package: **523 986 B**, SHA `6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a`
- CLI/HTTP/web/offline: без изменений, тесты проходят
- `ai_share`: lexical `max|Δ|=0`, `LOO=0,25`; model `max|Δ|=0,7524`, `LOO=0,75` → partially
- `.exe`: блокер по toolchain; build script/CI-инструкция сохранены в репозитории

## 8. Честный блокер

Потолок лексического ранкера объяснён: oracle-выбор кандидатов даёт Span F1 ≈ 1,0, реальный ранг gold-кандидата — 0 top-1. Смена модели на evidence-pointer и добавление реальных RAGTruth-спанов не дали guard: лучший честный OOF Span 0,4712 при Token 0,5394 / Pair 0,8527, но RusHallu val Span 0,3248. Для закрытия нужны либо метки clause-уровня в стиле RusHallu с большим объёмом (сейчас 55 val span), либо одобренная заказчиком гармонизация разметки, при которой held не должен смешивать whole/clause/short style. Без этого Success недостижим; цифры не фейкованы и held-out не тюнинговался.

[1]: https://github.com/ParticleMedia/RAGTruth — RAGTruth, MIT license.
