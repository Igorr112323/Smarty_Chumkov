# V16 — веса и данные через GitHub, калибровка в бюджете FPR и честный вердикт

## Итог одной строкой

Через **GitHub** получено и то, и другое: веса (`rubert-tiny2`, int8-ONNX, dequant → fp32,
сверен с onnxruntime) и **новый русский датасет с `doc_overlap = 0`** —
`DFKonGH/ru_hallucination_detector` (49 395 пар, SberChallenge/SberQuAD, `parquet`, 9 074
документа, ни один не совпал с нашими train/val/held-out). Весовой режим §4.1 работает:
LM-инференс в int8-ONNX укладывается в лимиты скорости (1k 0.007 с, 50k 0.992 с, 2M 30.23 с).

Калибровка порогов в разрешённом бюджете FPR ≤ 0.40 закрывает **Pair** (`0.5122`) и **Token**
(`0.6247`) при FPR `0.1442`, но **Span остаётся `0.3621` при требовании 0.50** →
**reject**. Диагноз точный и проверенный: у **всех 54** gold-спанов val есть кандидат с
IoU = 1.0 (генерация кандидатов идеальна), при идеальном ранжировании внутри пары и том же
pair-гейте Span F1 = `0.6165` — значит блокирует **качество ранжирования кандидатов**
(AUC `0.7425`), а не пороги, не кандидаты и не отсутствие весов. Ни один переобученный скор
кандидата (37 признаков на новых данных — `0.3226`; относительные признаки внутри пары —
`0.2832`) не обошёл калиброванный v11. Held-out не читался, рантайм не менялся, 142 теста
проходят, пакет 523 986 B с прежним SHA.

## ШАГ 1. Веса через GitHub — `artifacts/weights_v16.json`

| источник | файл | байт | SHA-256 | формат | роль |
|---|---|---|---|---|---|
| `pyramidheadshark/rubert-turbo-payments` | `models/onnx_model/model_quantized.onnx` | 29 385 865 | `f04067da…534420` | ONNX QOperator int8/uint8 | **основной**: rubert-tiny2 (312/3/12, vocab 83 828) |
| `models/rubert-tiny2-deq/pytorch_model.bin` | dequant fp32 | 116 786 340 | `46ed1562…acebb` | torch state_dict | рабочая модель для transformers |
| `Gregory-mcbit/avitomodel` | `spacer_model/checkpoint-113/model.safetensors` | 46 754 816 | `e04f7a83…7f9268` | safetensors fp32 | не используется (vocab 29 564) |
| `dinis-a/android-ru-stylometry` | `ru_detector/model.onnx` | 36 371 513 | — | ONNX (чужой fine-tune) | найден, не нужен (нет чистого энкодера) |

**`rubert-base` (~530 МБ) на GitHub нет.** Журнал поиска в артефакте: repository search по
9 запросам (`rubert-base`, `rubert-large`, `ruroberta`, `sbert russian`, `rubert-base-cased`,
`russian bert onnx`, `rubert russian`, `rubert-tiny2`, `russian hallucination`), code search по
4 запросам (`filename:pytorch_model.bin` / `model.safetensors` → total_count 0: GitHub не
индексирует бинарники), обход деревьев 10 репозиториев-кандидатов — файлов весов > 1 МБ нет
ни в одном, кроме чужих fine-tuned ONNX/классификаторов. Скачивание — `git clone --depth 1`
в `tools/external/` (каталог вне git), веса вне пакета.

## ШАГ 2. Новые русские данные через GitHub — `artifacts/ru_data_v16.json`

| репозиторий | файл | байт | лицензия | объём | doc_overlap |
|---|---|---|---|---|---|
| `DFKonGH/ru_hallucination_detector` | `data/ds_v0.parquet` | 30 461 228 | не указана в репо | 50 364 строки, 9 078 документов | **0** ✅ |
| `sber-summer-practice/hybrid-rag-hallucination-verification` | `data/optimized_rag_marking.xlsx` | 88 588 | не указана | 88 KB ручной разметки без span-аннотаций | не используется |

Перевод в наш формат (`tools/build_sber_v16.py`): `document_text = context`,
`model_output = evidence_aug`, gold-спан = **предложение `evidence_aug`, затронутое правкой**
(char-diff `evidence → evidence_aug`), у `clean` спанов нет. Итог: **49 395 пар, 41 981 со
спанами**, типы `_negation_flip` 21 619 / `_entity_swap` 18 355 / `clean` 7 406 / `_num_flip`
2 609, средняя длина ответа 146.6, документа 754.6. `doc_overlap` = 0 против `gold_v3`,
`mfava_train_v3` и `gold1200_v3` (held-out) по SHA-256 документов — жёсткое требование
выполнено. Стиль разметки — `unsupported_addition` по границам предложения (как наш C1).

## ШАГ 3. LM-признаки §4.1 — `artifacts/lm_features_v16.json`

5 признаков на токен ответа (энтропия внимания к документу, доля внимания на документ, пик
внимания, косинус к среднему документу, косинус к ближайшему токену документа), `mean`+`max`
по кандидату = 10 колонок к 37 штатным. Вход `[CLS] документ (≤380) [SEP] ответ (≤128) [SEP]`,
модель в `eval`, без дообучения. Посчитаны на 116 487 кандидатах (real/synth/pseudo/ragtruth/
val/mFAVA), тайминги 0.023–0.043 с на 1k знаков.

Полезность на val: AUC по отдельным признакам `0.50–0.60` (`attn_doc_peak_mean` 0.5982,
`cos_doc_mean_mean` 0.5904), штатный риск v11 на тех же кандидатах — `0.7425`. Вывод: сами по
себе LM-признаки слабее лексических и не поднимают ранжирование до нужного уровня (в v15
добавление всех 10 колонок меняло AUC на +0.007 по смеси и −0.03 по русским парам).

## ШАГ 4. Дообучение — только GBDT/инференс (GPU нет)

Переобучался **только скор кандидата** (HGB, экспорт деревьев в JSON без sklearn в рантайме);
mode-гейт, pair-гейт, `local_hgb` и boundary-логика v11 не тронуты. Три варианта:

| вариант | данные | признаков | val Span | Token | Pair | FPR |
|---|---|---|---|---|---|---|
| GBDT plain-37 | sber+synth+pseudo+real | 37 | 0.3226 | 0.5040 | 0.4348 | 0.2500 |
| GBDT plain-37 | sber+synth+pseudo | 37 | 0.2105 | 0.4751 | 0.4156 | 0.1442 |
| GBDT relative | sber+synth+pseudo+real | 37 + 40 относительных | 0.2645 | 0.4089 | 0.3721 | 0.2308 |
| GBDT relative | sber+real | 37 + 40 относительных | 0.2832 | 0.5606 | 0.4267 | 0.1250 |
| **калибровка v11 (без переобучения)** | — | 37 | **0.3621** | **0.6247** | **0.5122** | **0.1442** |

Новый датасет помогает относительно «без него» (`0.2105 → 0.3226`), но всё равно ниже
калиброванного v11: синтетические правки уровня предложения (number/entity/negation flip) не
переносятся на реальные клаузовые галлюцинации RusHallu-RAG.

## ШАГ 5. Калибровка + OOF — `artifacts/span_oof_v16.json`

Сетка: `mode_cut` × `gate` × `threshold` × `max_spans` × `local_config` = 2 160 конфигураций,
выбор по val в бюджете FPR ≤ 0.40 (скор и гейты v11 не переобучаются, симуляция по заранее
посчитанным таблицам скоров).

* лучший конфиг: `mode_cut 0.6, gate 0.4, threshold 0.6, max_spans 1, local_threshold 0.5,
  local_max_spans 3`;
* val: **Span 0.3621** (P 0.3387 / R 0.3889), **Token 0.6247**, **Pair 0.5122**, **FPR 0.1442**;
* mFAVA (не метрика договора): span `0.6694` — падения нет, `artifacts/mfava_v16.json`;
* 3-fold **OOF по документам** на реальных парах: Span `0.2247`, Token `0.4819`, Pair `0.3126`,
  FPR `0.2945` — ниже baseline OOF v14 (`0.2437`), калибровка на 150 val-парах не переносится
  → OOF-проверка тоже не пройдена.

### Почему Span не закрывается (измерено, не предположение)

| факт | значение |
|---|---|
| gold-спанов на val | 54 |
| имеют кандидата с IoU = 1.0 | **54 (100 %)** |
| ранг 1 по риску v11 | 20 |
| ранг ≤ 2 / ≤ 3 / ≤ 5 | 26 / 30 / 37 |
| Span F1 при идеальном ранжировании внутри пары (тот же гейт, `max_spans=1`) | **0.6165** |
| Span F1 при идеальном ранжировании, `max_spans=2` | 0.4689 |

То есть кандидаты содержат истину всегда, а декодер-сетка и пороги позволяют дойти до 0.62 —
не хватает **качества порядка кандидатов внутри пары** (AUC 0.7425 против ≳0.9, нужных для
P≈0.45 при флаге 84 пар). Ни признаки LM, ни 49k новых пар этого не дают.

## ШАГ 6. Скорость с весами — `artifacts/speed_v16.json`

| реализация | 1k (≤ 0.03 с) | 50k (≤ 1 с) | 2M (≤ 60 с) |
|---|---|---|---|
| torch fp32 (transformers) | 0.0149 ✅ | 1.9935 ❌ | 85.2813 ❌ |
| **int8-ONNX (onnxruntime, + выходы Softmax/LayerNorm)** | **0.0070 ✅** | **0.9920 ✅** | **30.2318 ✅** |

Рантайм-часть без LM (37 признаков + деревья) осталась на уровне v14: 1k 0.0089 с,
50k 0.1695 с, 2M 8.8141 с — лимиты договора выполняются и с весами, и без.

## ШАГ 7. Инженерия

* 142 теста — passed (`python -m pytest -q`);
* пакет `artifacts/spanver-0.2.0.zip` — 523 986 B (511.7 KB), SHA
  `6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a` — **не изменился**;
* рантайм без torch/sklearn; CLI/HTTP/JSON и `ai_share` не менялись (кода рантайма v16 не
  касался, все изменения — в `tools/` и `artifacts/`);
* веса и данные лежат вне пакета и вне git (`tools/external/` и `models/*` в `.gitignore`).

## Вердикт

| метрика договора | требование | v16 (val) | статус |
|---|---|---|---|
| Span F1 | ≥ 0.50 | **0.3621** | ❌ |
| Token F1 | ≥ 0.50 | 0.6247 | ✅ |
| Pair F1 | ≥ 0.50 | 0.5122 | ✅ |
| FPR | ≤ 0.40 | 0.1442 | ✅ |
| скорость 1k / 50k / 2M | 0.03 с / 1 с / 60 с | 0.0070 / 0.9920 / 30.2318 | ✅ |
| пакет | ≤ 0.6 МБ | 511.7 KB | ✅ |

**Reject по Span.** Честный блокер: нужен русский span-датасет с реальными (не синтетическими)
галлюцинациями уровня клаузы и `doc_overlap = 0` либо модель-ранжировщик с AUC ≳ 0.9 по нашим
кандидатам; на доступных данных и весах (tiny2, int8) потолок ранжирования — 0.36.

## Воспроизведение

```bash
python tools/download_weights_v16.py          # шаг 1: проверка весов + журнал поиска GitHub (~12 с)
python tools/build_sber_v16.py                # шаг 2: parquet → наш формат + doc_overlap (~12 с)
python tools/lm_stats_v16.py                  # шаг 3: статистика 5 LM-признаков (~5 с)
python tools/sweep_v16.py                     # шаг 5a: калибровка v11-декодера (таблицы ~40 с)
python tools/train_v16.py --sber-limit 6000   # шаг 4a: GBDT на 37 признаках (~7 мин)
python tools/train_v16_rank.py                # шаг 4b: относительный GBDT (~1.5 мин)
python tools/eval_v16.py                      # шаг 5b: val + OOF + потолки + вердикт (~20 с)
python tools/speed_v15.py --report artifacts/speed_v16.json   # шаг 6 (~2 мин)
python -m pytest -q && python tools/build_package.py
```

## Артефакты

| файл | что внутри |
|---|---|
| `artifacts/weights_v16.json` | источники весов на GitHub, URL, размер, SHA-256, формат, журнал поиска (repository/code/tree), почему нет `rubert-base` |
| `artifacts/ru_data_v16.json` | новые русские данные: репозиторий, путь, лицензия, SHA-256, статистика, конвертация, `doc_overlap = 0` |
| `artifacts/lm_features_v16.json` | 5 LM-признаков: колонки, шаблон входа, тайминги и статистика по группам, AUC по признакам на val |
| `artifacts/span_oof_v16.json` | калибровка (2 160 конфигураций), 3-fold OOF по документам, GBDT-варианты, потолки ранжирования, guard и вердикт |
| `artifacts/calibration_v16.json` | полный срез калибровки (top-конфиги, FPR-фронт) |
| `artifacts/speed_v16.json` | скорость LM в torch и int8-ONNX на 1k/50k/2M |
| `artifacts/mfava_v16.json` | mFAVA (не метрика договора) по всем вариантам |
| `artifacts/span_model_v16.json` | деревья относительного GBDT (для аудита, в рантайм не подключены) |
| `tools/download_weights_v16.py`, `build_sber_v16.py`, `lm_stats_v16.py`, `sweep_v16.py`, `train_v16.py`, `train_v16_rank.py`, `eval_v16.py` | скрипты шагов 1–6 |
