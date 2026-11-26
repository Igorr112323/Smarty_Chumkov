# Финальный отчёт № 117812 — цикл 8 (audit + гармонизация v3)

Дата 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`, HEAD на старте `1d53878`. Ручная разметка пользователем не использовалась; все новые метки получены детерминированным скриптом. Held-out не тюнинговался; новый v3 held-out не запускался, потому что guard на val/CV не пройден.

## 1. Вердикт

**Цель не закрыта. Блокер-отчёт.**

| Пункт | Статус | Факт |
|---|---|---|
| Span F1 ≥ 0,50 @ FPR ≤ 0,40 | ❌ |базовый v2 0,3253 @ 0,3043; v3 val лучший 0,3636 @ 0,0962 |
| Token F1 ≥ 0,50 @ FPR ≤ 0,40 | ❌ | v2 0,4990; v3 val лучший 0,6036 при span-конфиге, но pair 0,4865 |
| Pair F1 ≥ 0,50 @ FPR ≤ 0,40 | ⚠️ | v2 0,5000; v3 val 0,4634–0,4865 |
| P/R/F1/FPR, IoU ≥ 0,5 | ✅ | отчёты eval/sweep |
| `ai_share` step 0,25 | ⚠️ | lexical closed, model partially |
| 50k/2M/speed | ✅ | 0,3885 s / 11,8489 s |
| CLI/HTTP/web | ✅ | 142 tests |
| Package ≤ 0,6 MB + SHA | ✅ | 523 986 bytes |
| `.exe` | ⛔ | нет Windows toolchain |

## 2. Аудит стиля разметки

`tools/audit_annotation_style_v8.py` → `artifacts/annotation_style_audit_v8.json`.

| Сплит | Пар | Span | whole | short-word | clause-like | Типы |
|---|---:|---:|---:|---:|---:|---|
| train | 549 | 245 | 125 | 15 | 209 | 7 типов |
| val | 150 | 55 | 21 | 9 | 39 | 7 типов |
| mFAVA-train | 150 | 398 | 0 | 68 | 165 | unsupported_addition |
| v2 test301 | 301 | 123 | 54 | 13 | 100 | 7 типов |
| v2 extra50 | 50 | 157 | 0 | 60 | 62 | unsupported_addition |
| v2 held351 | 351 | 280 | 54 | 73 | 162 | смесь |

Вывод: held351 действительно смешивает несовместимые стили: whole-answer, clause и короткие слова в одном наборе. Train перекослен в whole/clause, mFAVA — в short/clause.

## 3. Гармонизация v3

Правило C1 (`tools/convert_labels_v3.py`): каждый исходный gold-span заменяется минимальным фрагментом из `candidate_spans`, который его содержит; whitespace trim; дубликаты удаляются; тип сохраняется. whole-answer остаётся whole, если нет меньшего содержащего фрагмента. Правило выбрано по train/val до any held-out оценки.

`tools/build_gold1200_v3.py` собрал:
- `artifacts/gold_v3.jsonl` — 699 пар, 289 span;
- `artifacts/mfava_train_v3.jsonl` — 148 пар, 391 span (2 пары исключены из-за совпадения документа с held);
- `artifacts/gold1200_v3.jsonl` — 1 200 пар, 351 held, 271 span;
- `artifacts/gold1200_v3_manifest.json` — doc_overlap_train_vs_held = **0**, старые версии сохранены.

## 4. Обучение на v3 и val/CV

Переобучены global ranker и local head на v3-метках (`artifacts/ranker_weights_v3.json`). Результаты на val_v3:

| Конфигурация | Span F1 | Token F1 | Pair F1 | FPR |
|---|---:|---:|---:|---:|
| Baseline v2 веса, v3-метки | 0,3248 | 0,6455 | 0,4634 | 0,1635 |
| Global v3 weights | 0,3585 | 0,6380 | 0,4634 | 0,1635 |
| Hybrid v3 weights | 0,3248 | 0,6455 | 0,4634 | 0,1635 |
| Best sweep gate0.4/th0.7/max3/nms0.3 | **0,3636** | 0,6036 | **0,4865** | 0,0962 |

Guard «все три ≥0,50» не пройден: при лучшем span pair = 0,4865, token = 0,6036. Поэтому v3 held-out **не оценивался**. Гармонизация снимает часть стилистического конфликта, но не поднимает pair/span до цели.

### Строгий CV (без утечки mFAVA)

`tools/cv_gate_v3.py`: mFAVA v3 разделён по SHA-256 документа на 3 доли (55/44/49). Обучение — RusHallu train + доля 0, пороги — val + доля 1, отчёт — val + доля 2. Результат на независимой доле 2 + val:

| Набор | Span F1 | Token F1 | Pair F1 | FPR |
|---|---:|---:|---:|---:|
| val + mFAVA fold2 | 0,3422 | 0,4658 | 0,5786 | 0,1731 |
| RusHallu val | 0,2542 | 0,5087 | 0,3797 | 0,1731 |
| mFAVA fold2 | 0,3891 | 0,4225 | 0,7750 | 0,0000 |

Guard_pass = **false**. Хорошие in-sample числа `local_ranker_report_v3.json` (span 0,6249 на val+mFAVA) объясняются тем, что mFAVA использовался и в обучении, и в подборе; строгий CV это опровергает.

### Полный sweep декодера на dev

`tools/cached_sweep_v8.py` перебирает 5×7×5×3 = 525 конфигураций глобального декодера на val+mFAVA-train v2 с кэшем признаков. Лучший Span F1 = **0,3621** @ FPR 0,1827; ни одна конфигурация не проходит guard (0 проходят). Это исключает «просто подобрать пороги/NMS/max_spans».

## 5. До / после по held-out версиям

| Набор · уровень | V1 P/R/F1 | V2 P/R/F1 | FPR V1→V2 |
|---|---|---|---|
| test301 · Span | 0,2880/0,2927/0,2903 | 0,2353/0,2927/0,2609 | 0,2995→0,3043 |
| test301 · Token | 0,3617/0,7817/0,4945 | 0,3540/0,7047/0,4713 | 0,2995→0,3043 |
| test301 · Pair | 0,3673/0,3830/0,3750 | 0,3636/0,3830/0,3731 | 0,2995→0,3043 |
| held501→351 · Span | 0,1424/0,0689/0,0928 | 0,3154/0,3357/0,3253 | 0,2995→0,3043 |
| held501→351 · Token | 0,2881/0,8876/0,4351 | 0,3891/0,6954/0,4990 | 0,2995→0,3043 |
| held501→351 · Pair | 0,4151/0,1497/0,2200 | 0,5227/0,4792/0,5000 | 0,2995→0,3043 |
| extra200→50 · Span | 0,0435/0,0155/0,0229 | 0,4000/0,3694/0,3841 | 0→0 |
| extra200→50 · Token | 0,2656/0,9407/0,4143 | 0,4699/0,6800/0,5557 | 0→0 |
| extra200→50 · Pair | 1,0000/0,0400/0,0769 | 1,0000/0,6600/0,7952 | 0→0 |

## 6. Артефакты и команды

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `artifacts/annotation_style_audit_v8.json` | 11 507 | `cebbe0286993158681fb84cc49f476b44275f3ce248d32f85dc98a00f8e4d7c5` |
| `artifacts/gold_v3.jsonl` | 3 743 302 | `cba82c37407733533f019b216981b26229bd9bbad8fdc7c989e58dcd013b3e8b` |
| `artifacts/mfava_train_v3.jsonl` | 1 877 316 | `228e2466eeaea38a8ae413c72b420863edc5ce5aca47f24eccb1d01f9118964e` |
| `artifacts/gold1200_v3.jsonl` | 7 933 632 | `910398513413d8a2896fdc8e1d13b716220ce0199a81c29cd39472e87951a9e5` |
| `artifacts/gold1200_v3_manifest.json` | 1 263 | `20a8e6393c6ce83b47d140443df9137f39c1ba048185d28a06beadcef8ec6b22` |
| `artifacts/ranker_weights_v3.json` | 215 777 | `ea2cccb8f944e19d9b5c00720ebf84fb1e8d0c688e8ab328319b5607c017c71f` |
| `artifacts/sweep_v3.json` | 156 225 | `ba991785f28ce81f2238c32a3a64422db35f564ea31bfffe2a6f369e295c8f64` |
| `artifacts/cv_gate_v3.json` | 1 705 | `2aea52d96943f33bd13f4800edf4a2bb3126d9c0a41d7f9fecfe48a0f3b48ef1` |
| `artifacts/cached_sweep_v8.json` | 101 019 | `8221c8c1ad6ead459e29bb71899c4d567f896dffaaf2b4014dfafca1991f26b8` |

```
.venv/bin/python tools/audit_annotation_style_v8.py
.venv/bin/python tools/build_gold1200_v3.py
.venv/bin/python tools/train_ranker.py --gold artifacts/gold_v3.jsonl --extra artifacts/mfava_train_v3.jsonl --weights artifacts/ranker_weights_v3.json --report artifacts/ranker_report_v3.json
.venv/bin/python tools/train_local_ranker.py --gold artifacts/gold_v3.jsonl --extra artifacts/mfava_train_v3.jsonl --weights artifacts/ranker_weights_v3.json --report artifacts/local_ranker_report_v3.json
.venv/bin/python tools/sweep_v3.py
.venv/bin/python tools/cv_gate_v3.py
.venv/bin/python tools/cached_sweep_v8.py
.venv/bin/python -m pytest -q
```

## 7. Тесты / скорость / пакет / интерфейсы

- `pytest -q`: **142 passed in 26.89s**
- 50k: **0,3885 с**; 2M: **11,8489 с**; 5,924 мс/1000
- Пакет: **523 986 байт**, SHA `6bbc5044…8ff7d73a`
- CLI/HTTP/web — рабочие; offline-stdlib — тест проходит

## 8. Остаточный потолок

Даже после честной гармонизации v3 лучший val-конфиг даёт Span 0,3636 и Pair 0,4865. Оракл коротких span есть, но scorer не находит их;все предыдущие попытки (ranking, embeddings, QA, BIOES, synthetic, trigger) не закрыли разрыв. Для закрытия нужны реальные короткие span-метки в едином стиле, больший encoder/GPU и пересмотр правила разметки с заказчиком ТЗ. Без этого Success недостижим в текущей среде.

РИД/заявка не готовилась по указанию пользователя.
