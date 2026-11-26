# Финальный отчёт № 117812 — цикл 7 (programmatic data / synthetic supervision)

Дата 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`, HEAD на старте `f1620ff`. Все новые данные и метки созданы программно; пользовательскую ручную разметку не использовали. Held-out не трогался, конфигурация не менялась.

## 1. Вердикт по договорным пунктам

| Пункт | Статус | Факт |
|---|---|---|
| Span F1 ≥ 0,50 @ FPR ≤ 0,40 | ❌ | 0,3253 @ 0,3043 |
| Token F1 ≥ 0,50 @ FPR ≤ 0,40 | ❌ | 0,4990 @ 0,3043 |
| Pair F1 ≥ 0,50 @ FPR ≤ 0,40 | ✅ | 0,5000 @ 0,3043 |
| Span/token/pair P,R,F1,FPR, IoU ≥ 0,5 | ✅ | `artifacts/ranker_eval_v2.json` |
| `ai_share` 0–1, calibration step 0,25 | ⚠️ | lexical max\|Δ\|=0, LOO=0,25; model max\|Δ\|=0,7524, LOO=0,75 |
| 50k ≤ 1 s, 2M ≤ 60 s | ✅ | 0,3784 s / 11,4971 s |
| CLI / HTTP API / web | ✅ | тесты проходят |
| Package ≤ 0,6 MB + SHA | ✅ | 523 986 bytes, `6bbc5044…8ff7d73a` |
| `.exe` | ⛔ | нет Windows toolchain; package вместо него |
| Тесты / воспроизводимость | ✅ | 142 passed; скрипты и артефакты в репозитории |

## 2. До / после

| Набор · уровень | V1 P/R/F1 | V2 P/R/F1 (текущий) | FPR V1→V2 |
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

Нового held-out прогона не было: ни one val/CV-конфигурация не прошла guard «все три F1 ≥ 0,50 при FPR ≤ 0,40».

## 3. Данные и weak/synthetic supervision

`tools/build_synthetic_v7.py` генерирует ответы из предложений train-документов и вносит контролируемые мутации: число, дата, сущность, отрицание, глагол противоречия. Gold span — точные offsets вставленного/заменённого фрагмента. Негативы — те же ответы без мутации.

| Параметр | Значение |
|---|---|
| Источник | `artifacts/gold.jsonl`, split=train, RusHallu-RAG docs |
| Пар | 6 308 (3 154 positive / 3 154 negative) |
| Уникальных документов | 539 |
| `doc_overlap_with_held` | **0** |
| bad spans | 0 |
| Мутации | entity 2217, number 493, negation 188, date 136, verb 120, negative 3154 |
| SHA JSONL | `8fd258eff3479b9987c495943ebdf6be09bc9074fa9f9019eec73eb183f4b2d6` |
| SHA manifest | `353b7fce2bbd07845800688507840035cd0a51ef2369f94fcf33db5d8bace81a` |

## 4. Итерации цикла 7

Baseline val+mFAVA-train: span/token/pair F1 `0,4731/0,6519/0,5995`, FPR `0,1635`.

| Итерация | Суть | val Span | mFAVA Span | val Token | Решение |
|---|---|---:|---:|---:|---|
| Synthetic local scorer (`tools/train_local_synth_v7.py`) | HGB на 469 440 локальных кандидатах вокруг синтетических мутаций + реальный train | 0,0851 | 0,1325 | 0,2925 | rejected |
| Trigger-narrowed baseline (`tools/trigger_narrow_v7.py`) | Синтетический trigger сужает кандидатов; внутри — замороженный baseline scorer | 0,1135 | 0,3106 | 0,5248 | rejected |
| Rule trigger (`tools/rule_trigger_v7.py`) | Прямое предсказание триггерных токенов без обучения | 0,0101 | 0,0141 | — | rejected |

Trigger-детектор покрывал 90,9% gold на val и 80,9% на mFAVA-train, но точность локализации внутри окна осталась низкой. Синтетические мутации выучиваются (внутри синтетики scorer работает), но не переносятся на реальную разметку: реальные короткие gold spans часто не являются «ошибкой» в документе, а представляют собой семантическое противоречие, которое lexical/contrast features не разделяют.

## 5. Что является потолком

1. **Распределение разметки.** В mFAVA короткий gold — отдельное слово внутри длинного пересказа; в RusHallu — клауза/весь ответ. Один и тот же локальный scoring не может быть одновременно точным в обоих режимах.
2. **Отсутствие переноса с синтетики.** Искусственные мутации дают идеальные короткие метки, но их поверхностные признаки (чужое число/сущность) не совпадают с реальными противоречиями (глагол, scope, negation flip).
3. **Оракл vs ranker.** Oracle `<40` = 100%, но медианный ранг 64/146, MRR 0,013–0,039. Все попытки улучшить scorer (лексические, embedding, hard negatives, synthetic, QA, BIOES) проигрывают baseline.
4. **Метрическая связность.** Token 0,4990 и Pair 0,5000 не оставляют запаса для агрессивного увеличения span.

Для закрытия цели нужны: данные с реальными короткими span-метками в том же стиле, что held-out, больший encoder/GPU и, вероятно, пересмотр/instruction разметки. Без этого текущая конфигурация остаётся лучшей воспроизводимой.

## 6. Артефакты и команды

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `artifacts/baseline_v7.json` | 5 521 | `b8873d21d1a8704b08c53f1121a7e8395a8a3bbf2030a075fb4c3bf1aa4359ab` |
| `artifacts/synthetic_short_v7.jsonl` | 36 118 103 | `8fd258eff3479b9987c495943ebdf6be09bc9074fa9f9019eec73eb183f4b2d6` |
| `artifacts/synthetic_short_v7_manifest.json` | 777 | `353b7fce2bbd07845800688507840035cd0a51ef2369f94fcf33db5d8bace81a` |
| `artifacts/local_synth_v7/model.pkl` | 363 930 | `8d5d5a0c4380694f4f51a15dbc0f268224a45e3a626b8e93c33c418afc617f5a` |
| `artifacts/local_synth_v7/report.json` | 75 269 | `45c17cd739c57a6733a200570886355ed80ac54d64796b7f07d1f6b7dd421da7` |
| `artifacts/trigger_narrow_v7.json` | 74 957 | `3bbf20f83d761dfeb7e17b1aa00effaa63d2d73acd31c1e9ccfbb321a0f5a1b7` |
| `artifacts/rule_trigger_v7.json` | 9 316 | `8740e129ae6d2c6cda8e19d01a31e8cf422472144cad808cd06736b6bfbf190e` |

```
.venv/bin/python tools/build_synthetic_v7.py
.venv/bin/python tools/train_local_synth_v7.py
.venv/bin/python tools/trigger_narrow_v7.py
.venv/bin/python tools/rule_trigger_v7.py
.venv/bin/python -m pytest -q
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
```

## 7. Тесты / скорость / пакет / интерфейсы / калибровка

- `pytest -q`: **142 passed in 24.52s**
- 50k: **0,3784 с**; 2M: **11,4971 с**, **5,749 мс/1000**
- Package: **523 986 bytes**, SHA `6bbc5044…8ff7d73a`
- CLI, HTTP `/verify`, web `localhost:8765` — рабочие
- `ai_share`: lexical max\|Δ\|=0, LOO=0,25; модельный max\|Δ\|=0,7524, LOO=0,75

## 8. Остаточные риски

Span F1 0,3253 и Token F1 0,4990 ниже цели; Pair F1 0,5000 без запаса; `.exe` отсутствует; модельный `ai_share` не откалиброван. Все эти пункты зафиксированы честно, без подгонки held-out.

РИД/заявка не готовилась по указанию пользователя.
