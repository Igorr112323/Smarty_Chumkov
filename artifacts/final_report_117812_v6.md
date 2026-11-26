# Финальный отчёт № 117812 — цикл 6 (sequence labeling / QA-style)

Дата 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`, HEAD на старте `c369f9a`. Все гипотезы проверены только на train/val/mFAVA-train. Held-out не тронут; прод-конфигурация не менялась.

## 1. Вердикт

**Цель не достигнута. Блокер-отчёт.**

| Уровень | P | R | F1 | FPR | Цель | Вердикт |
|---|---:|---:|---:|---:|---|---|
| Span | 0,3154 | 0,3357 | **0,3253** | 0,3043 | ≥ 0,50 | ❌ |
| Token | 0,3891 | 0,6954 | **0,4990** | 0,3043 | ≥ 0,50 | ❌ −0,0010 |
| Pair | 0,5227 | 0,4792 | **0,5000** | 0,3043 | ≥ 0,50 | ✅ на границе |

IoU ≥ 0,5, FPR ≤ 0,40 соблюдён. `ai_share`: lexical max\|Δ\|=0, LOO=0,25; модельный max\|Δ\|=0,7524, LOO=0,75 — partially.

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

## 3. Новые итерации и почему они отклонены

Baseline на `val×6 + mFAVA-train`: span/token/pair F1 `0,4731/0,6519/0,5995`, FPR `0,1635`.

| Итерация | Чем отличается от предыдущих | val Span | mFAVA Span | Решение |
|---|---|---:|---:|---|
| QA start/end (`tools/train_qa_span_v6.py`) | нет кандидатов/BIO; RuBERT с двумя head’ами и multi-label BCE по границам; pair gate заморожен | 0,0569 | 0,0602 | rejected |
| QA constrained decoding (`tools/eval_qa_span_v6.py`) | length prior из train, отдельные пороги start/end, пары start→end вместо top×top | 0,0985 | 0,1494 | rejected |
| BIOES + Viterbi (`tools/bioes_v6.py`) | последовательностная разметка с переходной матрицей, а не candidate ranking | 0,0000 | 0,0000 | rejected |

Все три попытки сохранили FPR на уровне baseline за счёт замороженного gate, но не дали ни одного TP-выигрыша. Ни один вес не принят в прод.

## 4. Что является потолком

1. **Разметка/данные.** В mFAVA короткие gold span’ы — это отдельные слова внутри длинного пересказа (`обнимать`, `упала`, `признан`), тогда как в RusHallu gold часто вся клауза или весь ответ. Одна и та же модель должна предсказывать границы в двух несовместимых масштабах. При IoU ≥ 0,5 одно смещение на токен для 5–10-символьного span’а уничтожает TP.
2. **Feature problem.** Oracle <40 = 100%, но медианный ранг правильного кандидата 64 (val) / 146 (mFAVA), MRR 0,013–0,039. Ни lexical, ни contextual embeddings, ни QA-логиты не отделяют короткий gold от тысяч правдоподобных негативных n-грамм.
3. **Compute/model capacity.** Доступен только CPU и `rubert-tiny2` (3 слоя, 312 hidden). Обученный QA head сходится к почти пустым предсказаниям; BIOES-теггер доминирует класс O. Для закрытия цели нужны: больший encoder (base/large) или GPU, больше размеченных примеров с короткими span’ами, и, вероятно, пересмотр инструкции разметки и целевых типов.
4. **Метрическая связность.** Token F1 почти закрыт (0,4990), но любые агрессивные правила повышения span роняют token или pair._pair F1 ровно 0,5000 не оставляет запаса.

## 5. Артефакты и SHA-256

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `artifacts/qa_span_v6/head.pt` | 3 917 | `c6a211477b99dcc67f7b1f11f3858788df7f6c07792c00b84ab3125d58fde42f` |
| `artifacts/qa_span_v6/report.json` | 3 287 | `182818804624aedb52b2e78feeed4aa99923f2f95b5f7914a65dbdfda14b4e67` |
| `artifacts/qa_span_v6/decode_report.json` | 436 685 | `d598fc1a1de9f659ab6afc5976620a3265ac6fdc9f9c5a4aa2561c49dab702a0` |
| `artifacts/bioes_v6/tagger.pkl` | 1 006 779 | `98c9a8a0ba5238a7cf8dddf468460cdad49d66c57651f58f106c4baf9bae50d2` |
| `artifacts/bioes_v6/report.json` | 28 687 | `77a645c297449874fb053cccb00d5f89b4cccd64320aa2d1e01c1d14ea3ce395` |
| `artifacts/ranking_analysis_v5.json` | 62 916 | `c749173365efe7bd701b3557db6e4ac8de7127841df2f7d3f83e2946d10aa10d` |
| `artifacts/oracle_recall_v4.json` | 9 208 | `c19d7f434d6b248f97b2a4f884dfb705a9961bf277ea1ad11735de62045a532c` |
| `artifacts/gold1200_v2.jsonl` | 7 865 949 | `28e21f0ef95cd1b2a5ac3cf5f4f4a83fb663c51fe951e56b0e9b7a017400af4e` |
| `spanver/ranker_weights.json` | 227 338 | `853902cb78ae27b9b0f1c8e94b0f6bd2c5af7ffb26455f0de7caad6babc12eac` |
| `artifacts/spanver-0.2.0.zip` | 523 986 | `6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a` |

Полный checkpoint `artifacts/qa_span_v6/model.pt` (117 МБ) является производным и в Git не коммитится: сохраняются только head-веса `head.pt`, базовый encoder берётся из `models/rubert-tiny2`.

Команды воспроизведения:

```
.venv/bin/python tools/train_qa_span_v6.py --epochs 2 --lr 3e-5
.venv/bin/python tools/eval_qa_span_v6.py
.venv/bin/python tools/bioes_v6.py
.venv/bin/python tools/ranking_analysis_v5.py --cache artifacts/candidate_cache_v5.pkl
.venv/bin/python -m pytest -q
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
```

## 6. Тесты / скорость / пакет / интерфейсы

- `pytest -q`: **142 passed in 25.45s**
- 50k: **0,3925 с**; 2M: **11,8307 с**; **5,915 мс/1000**
- Пакет: **523 986 байт**, SHA `6bbc5044…8ff7d73a`
- CLI `python -m spanver verify`, HTTP `/verify`, web `localhost:8765` — рабочие
- Офлайн-рантайм на stdlib — тест проходит

## 7. Остаточные риски / что нужно для закрытия

1. Главный блокер — короткие span’ы при длинном ответе; oracle есть, нет сигнала для выбора гранулярности.
2. Token F1 0,4990 и Pair F1 0,5000 без запаса.
3. Модельный `ai_share` не откалиброван; `.exe` и `model_eval_report.json` отсутствуют.
4. Что нужно человеку/ресурсам: больше ручной разметки коротких span’ов в длинных ответах, согласованная инструкция по типам и границам, GPU или больший encoder, 그리고 повторный CV-подбор под новую разметку. Без этого текущая конфигурация остаётся лучшей доступной.

РИД/заявка не готовилась по указанию пользователя.
