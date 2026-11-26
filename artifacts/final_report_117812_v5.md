# Финальный отчёт № 117812 — цикл 5 (ranking analysis + control scoring)

Дата 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`, HEAD на старте `e305232`. Все гипотезы проверялись только на train/val/mFAVA-train. Held-out не использовался; конфигурация не менялась, поэтому новый held-out прогон не выполнялся.

## 1. Вердикт

**Цель не достигнута. Блокер-отчёт.**

| Уровень | P | R | F1 | FPR | Цель | Вердикт |
|---|---:|---:|---:|---:|---|---|
| Span | 0,3154 | 0,3357 | **0,3253** | 0,3043 | ≥ 0,50 | ❌ |
| Token | 0,3891 | 0,6954 | **0,4990** | 0,3043 | ≥ 0,50 | ❌ (-0,0010) |
| Pair | 0,5227 | 0,4792 | **0,5000** | 0,3043 | ≥ 0,50 | ✅ на границе |

IoU ≥ 0,5. FPR ≤ 0,40 соблюдён. `ai_share`: lexical max\|Δ\|=0, LOO=0,25; модельный max\|Δ\|=0,7524, LOO=0,75 — partially. `.exe` не собран из-за отсутствия Windows toolchain.

## 2. Ranking analysis (главный результат цикла)

`tools/ranking_analysis_v5.py` → `artifacts/ranking_analysis_v5.json`. Правильный кандидат = IoU ≥ 0,5 с gold; ранг считается среди всех v4-кандидатов пары по активному score baseline-модели.

| Сплит | short gold | top1 | top10 | median rank | MRR |
|---|---:|---:|---:|---:|---:|
| train | 25 | 2 | 50 | 84 | 0,0471 |
| val | 11 | 0 | 20 | 64 | 0,0391 |
| mFAVA-train | 93 | 0 | 16 | 146 | 0,0129 |

Пороги по коротким кандидатам на mFAVA: precision 0,114@0,2; 0,205@0,3; **0,314@0,4**, recall 0,0126, candidate FPR 0,0003. На val precision коротких кандидатов ≤0,033 при любом пороге. Вывод: oracle-кандидаты есть, но scorer их не находит — feature/ranking problem, не пороговая.

## 3. Что пробовал и что отклонено

Baseline на `val×6 + mFAVA-train`: span/token/pair F1 `0,4731/0,6519/0,5995`, FPR `0,1635`.

| Итерация | Чем отличается от отклонённых v4 | val span | mFAVA span | val token | Решение |
|---|---|---:|---:|---:|---|
| Context embeddings (`tools/embedding_score_v5.py`, `embedding_rank_v5.py`) | новый признак answer-token↔doc-token cosine; условный locator при замороженном gate | 0,1705 | 0,6421 | — | rejected: embedding AUC 0,37–0,40 |
| Conditional add/replace (`tools/conditional_add_v5.py`) |замороженный pair gate, добавление/замена только внутри уже с флагом пар | 0,2689 | 0,6452 | — | rejected: хуже baseline по val |
| Hard-negative global head (`tools/hard_negative_v5.py`) | negatives 0,2≤IoU<0,5 с весом 6; gate/mode заморожены | 0,2712 | 0,6443 | — | rejected: нет изменений |
| Mode_cut sweep (`tools/mode_cut_v5.py`) | калибровка только выбора global/local, head’ы не меняются | 0,3519 @0,9 | 0,4931 @0,9 | 0,6405 | rejected: token проседает на 0,0026 |
| Performance-labelled mode gate (`tools/mode_gate_v5.py`) | метка = argmax per-pair F1 двух head’ов, а не эвристика v2 | 0,3364 @0,9 | 0,2517 @0,9 | — | rejected |

Лучший по weighted span mode_cut=0,75: val span 0,3273, mFAVA 0,5938, weighted span 0,3654 (baseline 0,3245), но weighted token 0,6312 против 0,6412 у baseline. Поскольку Token F1 на held уже 0,4990, это неприемлемо. Ни одна итерация не дала одновременно рост span/token/pair при FPR ≤ 0,40, поэтому прод-конфиг не менялся.

## 4. До / после

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

## 5. Изменения в репозитории

Скрипты (все в `tools/`, без `/tmp`):
- `build_candidate_cache_v5.py`, `ranking_analysis_v5.py`
- `embedding_score_v5.py`, `embedding_rank_v5.py`
- `conditional_add_v5.py`, `hard_negative_v5.py`
- `mode_cut_v5.py`, `mode_gate_v5.py`

Артефакты (все в `artifacts/`): `ranking_analysis_v5.json`, `embedding_rank_v5.json`, `conditional_add_v5.json`, `hard_negative_v5.json`, `mode_cut_v5.json`, `mode_gate_v5.json`, плюс компактные кэши `embedding_v5_cache.pkl`, `conditional_scores_v5.pkl`, `baseline_flags_v5.pkl`. Большой derived cache `candidate_cache_v5.pkl` (110 МБ) не коммитится: он воспроизводится `build_candidate_cache_v5.py`.

Модель, веса, пороги, сплиты, package — без изменений.

## 6. Артефакты и SHA-256

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `artifacts/ranking_analysis_v5.json` | 62 916 | `c749173365efe7bd701b3557db6e4ac8de7127841df2f7d3f83e2946d10aa10d` |
| `artifacts/embedding_rank_v5.json` | 33 822 | `d50ff38a0dd1750d53bf82a76fb00a950e9b70dab66fe75b4ca5e17039fd636c` |
| `artifacts/conditional_add_v5.json` | 221 543 | `23e88c58dcdc6f8fdcccb6134da16de12eb9f57bcd7c0def3b7b0895365c56af` |
| `artifacts/hard_negative_v5.json` | 31 160 | `3b771da8df8bda634a2d9d99616fb6ceb2c54be47adf9fbf4cd01defc440d7bc` |
| `artifacts/mode_cut_v5.json` | 19 152 | `771289f11be6592475a9834972bb26cec7843b9af81d26e3670da839d299ff30` |
| `artifacts/mode_gate_v5.json` | 15 632 | `aa5c8021e1ebd9493be61dbef44acafdc8dc1b8e2fd511d897d9a61eafebc347` |
| `artifacts/oracle_recall_v4.json` | 9 208 | `c19d7f434d6b248f97b2a4f884dfb705a9961bf277ea1ad11735de62045a532c` |
| `artifacts/gold1200_v2.jsonl` | 7 865 949 | `28e21f0ef95cd1b2a5ac3cf5f4f4a83fb663c51fe951e56b0e9b7a017400af4e` |
| `spanver/ranker_weights.json` | 227 338 | `853902cb78ae27b9b0f1c8e94b0f6bd2c5af7ffb26455f0de7caad6babc12eac` |
| `artifacts/spanver-0.2.0.zip` | 523 986 | `6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a` |

Команды воспроизведения:

```
.venv/bin/python tools/build_candidate_cache_v5.py --out artifacts/candidate_cache_v5.pkl
.venv/bin/python tools/ranking_analysis_v5.py --cache artifacts/candidate_cache_v5.pkl
.venv/bin/python tools/embedding_score_v5.py
.venv/bin/python tools/embedding_rank_v5.py --cache artifacts/candidate_cache_v5.pkl
.venv/bin/python tools/conditional_add_v5.py
.venv/bin/python tools/hard_negative_v5.py
.venv/bin/python tools/mode_cut_v5.py
.venv/bin/python tools/mode_gate_v5.py
.venv/bin/python -m pytest -q
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
```

## 7. Тесты / скорость / пакет / интерфейсы

- `pytest -q`: **142 passed in 25.39s**
- 50k: **0,3807 с**; 2M: **11,5612 с**; **5,781 мс/1000**
- Пакет: **523 986 байт**, SHA `6bbc5044…8ff7d73a`, совпадает с `package.sha256`
- CLI `python -m spanver verify`, HTTP `/verify`, web `localhost:8765` — рабочие и покрытые тестами
- Офлайн-рантайм на stdlib — тест проходит

## 8. Остаточные риски

1. Главный: правильные короткие кандидаты находятся на медианном ранге 64–146; существующие признаки не отделяют их отoтысяч негативных кандидатов.
2. Token F1 0,4990 без запаса; mode_cut улучшает span, но снижает token.
3. Pair F1 ровно 0,5000.
4. Модельный `ai_share` не откалиброван; `.exe` и `artifacts/model_eval_report.json` отсутствуют.
5. Обученные в этом цикле head’ы не сохранялись в прод: все проиграли baseline на val/mFAVA-train.

РИД/заявка не готовилась по указанию пользователя.
