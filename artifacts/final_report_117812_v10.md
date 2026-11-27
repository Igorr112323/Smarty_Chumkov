# Финальный отчёт № 117812 — цикл 10 (фокус: RusHallu clause boundaries и ранжирование)

Дата 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`; HEAD на старте `38661d3`, промежуточный коммит `c92f2de`. Воспроизведение v9 выполнено. Held-out не запускался: guard на val/CV ни разу не пройден. Ручной разметки пользователем не было; РИД/заявка не готовилась.

## 1. Вердикт по договорным пунктам

| Пункт | Статус | Факт |
|---|---|---|
| Span F1 ≥ 0,50 @ FPR ≤ 0,40 | ❌ | held351 (v2) **0,3253 @ 0,3043**; лучший честный dev/CV в v10 — mFAVA OOF **0,4712** (v9), на русском val ни одна новая модель не превысила **0,3248** |
| Token F1 ≥ 0,50 @ FPR ≤ 0,40 | ⚠️ | held351 **0,4990** (−0,0010); v10 relative decode dev **0,5161** |
| Pair F1 ≥ 0,50 @ FPR ≤ 0,40 | ✅ на границе | held351 **0,5000**; mFAVA OOF **0,8527**, relative decode dev **0,8092** |
| IoU ≥ 0,5 | ✅ | все отчёты evaluate используют `iou_threshold=0.5` |
| 1200 пар / manifest / doc_overlap = 0 | ✅ | `gold1200_v3.jsonl`, старые версии сохранены |
| 142 теста / пакет / скорость / CLI / HTTP / web / offline | ✅ | 142 passed; 523 986 B; 0,3809 с / 11,2885 с; SHA `6bbc5044…8ff7d73a` |
| ai_share шаг 0,25 | ⚠️ | lexical max\|Δ\|=0, LOO=0,25; model max\|Δ\|=0,7524, LOO=0,75 (partially) |
| `.exe` | ⛔ | нет Windows/Wine/PyInstaller; build script/CI остаются |

**Итог: Success не достигнут. Блокер-отчёт.** Никакой подгонки по held-out не делалось.

## 2. Воспроизведение v9 и новая диагностика

Воспроизведены: `audit_ceiling_v9.json` (oracle Span F1 ≈ 1,0 при реальном 0,271), `cv3_oof_v9.json` (`0,4712/0,5394/0,8527 @ 0`, guard false), `neural_pointer_v9/eval_full_dev.json` (`0,3328/0,4842/0,7081 @ 0,4327`), `rag_aug_v9_report.json` (Span `0,0164`).

Новый разбор `tools/analyze_rushallu_val_v10.py` → `artifacts/rushallu_val_error_analysis_v10.json`:

| Тип gold | Число | Oracle-кандидат найден | Медианный ранг oracle | Попадание текущей модели | Промах |
|---|---:|---:|---:|---:|---:|
| whole-answer | 23 | 23/23 | **1** | 11/23 | 12 |
| clause | 27 | 27/27 | 12 | 7/27 | 20 |
| short | 4 | 4/4 | 17,5 | 0/4 | 4 |

Два конкретных механизма:

- **Mode-gate** на val при `cut=0.6` предсказал 9 «локальных» пар, из них 0 истинных, и пропустил 4 истиные. Из-за этого whole-кандидат (ранг 1, risk 0,90) удалялся local head — примеры `sber000008`, `sber000024`, `sber000089`.
- **Clause/short**-кандидаты имеют ранг 2–20, но низкий абсолютный risk (0,09–0,39); жёсткий порог 0,6 их отсекает. Для clause-золотых насчитывается 392 кандидата с IoU 0,3–0,5 и 147 с IoU 0,5–0,8 — то есть «почти правильные границы» доминируют.

## 3. Новые эксперименты v10

| Эксперимент | Идея | Результат на dev | Решение |
|---|---|---|---|
| `tools/relative_decode_v10.py` | softmax по candidate logits внутри пары вместо абсолютного порога | combined **0,4155/0,5161/0,8092 @ 0,1154**; val **0,2645/0,5817/0,4324** | val ниже guard, отклонён |
| `tools/train_pairwise_v10.py` + `eval_pairwise_v10.py` | pairwise logistic на разностях признаков gold − hard negative (сдвинутые на 1–3 токена границы, вложенные и почти совпадающие кандидаты), 39 060 diff-строк | лучший режим фактически baseline: combined **0,4091/0,5199/0,8023 @ 0,1154**; val **0,2931/0,6021/0,4533** | не превзошёл v3, отклонён |
| `tools/train_whole_gate_v10.py` | отдельный gate «есть whole-answer» на train-метках (137 позитивов), force whole / global fallback | combined **0,3910/0,5272/0,7803 @ 0,1635**; val **0,3115/0,6133/0,4634** | выигрыш на whole съедается FP, отклонён |
| Полный `rubert-tiny2` pointer, 3000 RAGTruth + 697 in-domain | 44 320 строк, 2 эпохи, lr 2e−5, batch 16, checkpoint `last`; eval по 149k кандидатов | dev **0,3674/0,4970/0,6888 @ 0,6058**; val **0,1638/0,4733/0,2969 @ 0,6058** | FPR > 0,40, отклонён |

Отдельно: попытка `rag-limit=6000` была убита OOM (RSS 3,78 GB при лимите ~4 GB) до сохранения чекпоинта; лог сохранён в `artifacts/neural_pointer_full3000_v10.log`. Масштабирование neural-данных с 300 до 3000 пар не улучшило результат, гипотеза «недостаток данных» опровергнута.

## 4. До / после по всем held-out версиям

Новый held-out прогон не выполнялся. Сводка v1/v2 из `artifacts/final_report_117812_v8.md`:

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

## 5. Данные, источники, лицензии

| Источник | Скрипт | Размер | SHA-256 | doc_overlap | Правило разметки |
|---|---|---:|---|---:|---|
| `artifacts/gold.jsonl` train/val | `tools/build_gold1200_v3.py` | 699 пар | в v3 manifest | 0 | C1 (`tools/convert_labels_v3.py`) |
| `artifacts/mfava_train_v3.jsonl` | тот же | 148 пар | в v3 manifest | 0 | C1 + исключение 2 пар по SHA документа |
| `artifacts/gold1200_v3.jsonl` | тот же | 1 200 пар / 351 held | `910398513…a9e5` | 0 | 271 span; v1/v2 сохранены |
| `Dataset/RAGTruth_ru/ragtruth_ru.jsonl` | `tools/translate_ragtruth.py` | 6 000 пар / 7 830 span | `4acb2988…8800ef` | 0 | готовые автоматические span; оригинальный RAGTruth — MIT [1](https://github.com/ParticleMedia/RAGTruth) |

Старые версии v1–v9 не удалялись; все новые метки получены программно.

## 6. Модель и декодирование

- Базовый runtime: `spanver/ranker.py`, linear candidate scorer + pair gate + optional local HGB; конфиг v3 зафиксирован в `artifacts/ranker_weights_v3.json`.
- Исследованные в v10 варианты: pair-relative softmax (`T=0,1`, `relative=0,01`, gate `0,5–0,6`, max_spans 5), pairwise logistic на 50 признаках (37 baseline + 13 boundary), отдельный whole-gate (14 признаков), evidence pointer `rubert-tiny2` (3 слоя, hidden 312, candidate/evidence CLS, max_len 160, 44 320 строк).
- Все пороги и режимы подбирались **только** на val + mFAVA-train; cfg с FPR > 0,40 или val F1 < 0,50 не принимался.

## 7. Артефакты и команды

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `artifacts/rushallu_val_error_analysis_v10.json` | 13 127 | `09d849ab261cf3754ab0b2d9b61fd97472d67c6a60d298fe713e5603bbb3d91f` |
| `artifacts/relative_decode_v10.json` | 157 494 | `92a1cf85fd603572eaa0b3a70e290c3e5841fedac81eb035c2f18e640ec6f3e9` |
| `artifacts/pairwise_weights_v10.json` | 2 071 | `d23d7352203262cc5eef5a7e639d8d4a9d703c4abd07a8bb7a6f6daeec227ee1` |
| `artifacts/pairwise_eval_v10.json` | 75 880 | `49a9f484442d8898f84377adca3ccc17d10b434ec5dc249def1171d6abbdccf5` |
| `artifacts/whole_gate_v10.json` | 83 411 | `1ea8471e5bcd88f15d1f7381edeb642a2d2bb26b936574e047da903ae1d4c91d` |
| `artifacts/neural_pointer_full3000b_v10/report_full.json` | 762 | `013f520e57078c052dc9d98bf3d5f266df6661d1d0270e2ad500ec67cdd77188` |
| `artifacts/neural_pointer_full3000b_v10/eval_full_dev.json` | 14 972 | `2c7347256af2afa8a9fd85212c5bc598e30cadb87716fd8e5ceccd8765d8f32a` |
| `artifacts/neural_pointer_full3000b_v10/last/head.pt` | 782 797 | `5a7ac4522c4954633d822e15bdecd60ec5ef71ebf0e3698871fa59282230bb98` |
| `artifacts/neural_pointer_full3000b_v10/last/model.safetensors` | 116 781 184 | `206b8567ec6dd71998c1a0bd1cb3760cb23818630e2c7ec66e459d8ea23dbfc0` |

```
.venv/bin/python tools/analyze_rushallu_val_v10.py
.venv/bin/python tools/relative_decode_v10.py
.venv/bin/python tools/train_pairwise_v10.py && .venv/bin/python tools/eval_pairwise_v10.py
.venv/bin/python tools/train_whole_gate_v10.py
.venv/bin/python tools/train_neural_pointer_v9.py --rag-limit 3000 --epochs 2 --batch 16 --no-dev-eval --out artifacts/neural_pointer_full3000b_v10
.venv/bin/python tools/eval_neural_pointer_v9.py --ckpt artifacts/neural_pointer_full3000b_v10/last --output artifacts/neural_pointer_full3000b_v10/eval_full_dev.json
.venv/bin/python -m pytest -q
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
```

## 8. Тесты / скорость / пакет / интерфейсы / ai_share

- `pytest -q`: **142 passed in 24.07 s**
- 50k **0,3809 с**, 2M **11,2885 с**, 5,644 мс/1000 символов
- Package **523 986 B**, SHA `6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a`
- CLI/HTTP/web/offline: без изменений, тесты проходят
- `ai_share`: lexical `max|Δ|=0`, `LOO=0,25`; model `max|Δ|=0,7524`, `LOO=0,75` — partially
- `.exe`: блокер по Windows toolchain; build script/CI-инструкция сохранены

## 9. Честный блокер

Диагностика доказала, что потолок — не offsets и не candidate recall (oracle Span F1 ≈ 1,0), а выбор границ: на val у whole-answer oracle-ранг 1, но старый mode gate удаляет whole-кандидата; у clause gold медианный ранг 12 при множестве near-miss кандидатов. Четыре новых рычага (relative decode, pairwise hard negatives, whole gate, полномасштабный neural pointer) не дали val/CV guard: лучший val Span во всех новых экспериментах — `0,3115`, лучший честный OOF mFAVA — `0,4712` (v9).

Для Success нужны либо clause-метки в стиле RusHallu с объёмом, сопоставимым с mFAVA, либо формально согласованное с заказчиком правило разметки, при котором held351 не смешивает whole/clause/short style. В текущих условиях цель недостижима без подгонки под held-out, что запрещено. Цифры получены реальными прогонами, JSON вручную не правился.

[1]: https://github.com/ParticleMedia/RAGTruth — RAGTruth, MIT license.
