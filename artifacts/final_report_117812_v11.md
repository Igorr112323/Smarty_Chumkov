# Финальный отчёт № 117812 — цикл 11 (RusHallu-style данные + listwise выбор границ)

Дата 2026-11-25. Ветка `arena/963b6d7a-smarty-chumkov`. Старт: `93f690c` (v10), коммиты v11: `3fb87f3`, `c0f2a7f`, `fc2aba4`. Held-out **не запускался**: guard на val/OOF не пройден. Ручной разметки не было, РИД/заявки/патенты не готовились.

## 1. Вердикт по договорным пунктам

| Пункт | Статус | Факт (val / mFAVA / combined) |
|---|---|---|
| Span F1 ≥ 0,50 @ FPR ≤ 0,40 | ❌ | лучший честный v11: val **0,3363**, mFAVA 0,6895, combined 0,6438 |
| Token F1 ≥ 0,50 @ FPR ≤ 0,40 | ✅ | val **0,6191**, mFAVA 0,7067, combined 0,6854 |
| Pair F1 ≥ 0,50 @ FPR ≤ 0,40 | ⚠️ | val **0,4872** (недобор 0,0128), mFAVA 0,9197, combined 0,8239 |
| FPR ≤ 0,40 | ✅ | 0,125 (val), 0,0 (mFAVA) |
| IoU ≥ 0,5 | ✅ | `spanver.evaluate.metrics` |
| 1200 пар / manifest / doc_overlap = 0 | ✅ | старые версии сохранены, новые данные с манифестами |
| 142 теста / пакет / скорость / CLI / HTTP / web / offline | ✅ | 142 passed; 523 986 B; 0,0456 / 0,3222 / 10,2932 с |
| ai_share шаг 0,25 | ✅ | max\|Δ\| = 0,0 на 5 точках микста, монотонно, регенерация байт-в-байт |
| `.exe` | ⛔ | нет Windows/Wine/PyInstaller в песочнице |

**Success не достигнут. Span на RusHallu val вырос с 0,3186 до 0,3363, но до 0,50 не дошёл. Блокер-отчёт.**

Восстановление v10 в чистой песочнице: `artifacts/gold.jsonl` пересобран через `spanver.build` и совпал с манифестным SHA `fddd62ce05baf028b64cb6db81b89053bdc5444a540c6e18e34e318577c8b32b`; окружение python 3.11.2 / numpy 1.26.4 / sklearn 1.9.1 / lightgbm 4.7.0 / transformers 4.38.2 / torch 2.15.0+cu132, `pytest -q` → 142 passed.

## 2. Блок 1 — данные в стиле RusHallu

| Датасет | Скрипт | Пар | SHA-256 | doc_overlap |
|---|---|---:|---|---:|
| `clause_pseudo_v11.jsonl` (1.1) | `tools/build_clause_pseudo_v11.py` | 84 | `edba48e1825fc8ddf26c0134a85f531958fcd503f37912372971f03cbba80030` | val 0 / held1200 0 / mFAVA 0 |
| `rushallu_synth_v11.jsonl` (1.2) | `tools/build_rushallu_synth_v11.py` | 3 563 | `64b4633ccf8be219cc4c36ce7d25463d0711c293aad140fb76550e9b916a3840` | val 0 / held1200 0 |

**1.1 Pseudo-clause.** Из 133 train-пар с whole-меткой 84 дали отделимый низко-поддержанный clause (`min_support < 0,75 × median`), 29 отброшены как неотделимые, 436 пар не имели whole-метки. Средняя длина ответа 523,9 симв., pseudo-спан 45,8 симв. (13,2 % ответа). Манифест: `5e8906bb8bb2682d8fc4f3e6a823bbbda70af5583727879f75197f36033ddd65`.

**1.2 Синтетика с точными границами.** Мутации: `swap` 3 170, `entity` 348, `number` 45; ровно одна ошибка на ответ; ответы ≥165 симв. (среднее 493,6, p50 445), спан 70,2 симв. (p50 62), доля спана 14,2 %. Числовых мутаций мало объективно: только 186 из 549 train-ответов содержат цифры, а 218 clause — числа; отсюда 45 пар. Манифест: `b6481b7ea3363cc5f5bf9f6e77341e2ca7abc15b6d5c54f5dd3bdd6e971549ec`. Лицензия: производная от `Dataset/1_rushallu_rag_gold` (cc), только train-документы.

**Найденный конфликт разметки (важно).** Pseudo-clause пара строится из того же ответа, что и родительская whole-пара: одинаковый вход, противоположные метки. Поэтому pseudo нельзя смешивать с whole-надзором без потери. Измерено (listwise, val / mFAVA / combined span): без pseudo **0,1228 / 0,5565 / 0,4988**, с pseudo **0,1607 / 0,5339 / 0,4855**. Pseudo помогает val (+0,038) и вредит mFAVA (−0,023), guard не двигает. В основной конфигурации pseudo исключён.

## 3. Блок 2 — выбор границ

**2.1 Listwise-ранкер (LightGBM `lambdarank`, 300 деревьев, leaves 15, lr 0,05).** Позитив = gold-граница, негативы = сдвиги ±1..3 слова, расширение/сжатие на clause, соседние clause, топ-1 baseline, случайные кандидаты; экспорт деревьев в чистый JSON, расхождение с LightGBM `max|Δ| = 0,0`.

| Конфигурация | val span | mFAVA span | combined span/token/pair @FPR |
|---|---:|---:|---|
| baseline v3 (runtime, mode gate) | 0,3186 | 0,6895 | 0,6415 / 0,6853 / 0,8205 @ 0,125 |
| listwise, full features (clause-only) | 0,1379 | 0,5598 | 0,5040 / 0,5570 / 0,8242 @ 0,0962 |
| listwise, full + whole-пары | 0,1228 | 0,5565 | 0,4988 / 0,5659 / 0,8211 @ 0,0673 |
| ablation: только document-support | 0,1429 | 0,4357 | 0,3914 / 0,5373 / 0,7576 @ 0,1058 |
| ablation: только boundary-признаки | 0,0962 | 0,0623 | 0,0690 / 0,1866 / 0,2846 @ 0,1635 |
| ablation: без document-support | 0,1622 | 0,5118 | 0,4674 / 0,5471 / 0,8187 @ 0,0769 |

Вывод: document-support даёт mFAVA +0,045 (0,5118 → 0,5565) и это единственный работающий блок признаков; boundary-признаки без support бесполезны. **На val listwise проигрывает baseline во всех конфигурациях → reject как замена декодера.**

**2.2 Document-support rerank / stacker.** Причина провала вскрыта ablation'ом scorer'ов (`artifacts/score_compare_v11.json`, top-1 точность на val):

| Scorer | whole | clause | short | val span (лучший декод) | mFAVA span |
|---|---:|---:|---:|---:|---:|
| baseline v3 global | **0,87** | 0,222 | 0,0 | 0,2275 | 0,4048 |
| local HGB (mFAVA) | 0,261 | **0,50** | 0,20 | 0,2057 | 0,6563 |
| document-support | 0,217 | 0,167 | 0,40 | 0,1159 | 0,1653 |
| fusion base+local | 0,609 | 0,389 | 0,0 | 0,2057 | 0,6060 |

Головы **комплементарны**: base находит whole, local — clause; ни одна не решает обе задачи. Stacker (lambdarank на реальных gold train+mFAVA, скоры обеих голов как признаки): mFAVA вырос до **0,7850**, val упал до 0,1758; балансировка whole-групп (вес 1/3/6) даёт val 0,1758–0,1875. **Reject по val.**

**2.3 Boundary adjuster (±3 слова + привязка к пунктуации, логистический классификатор на train).**

| Декод | val span | val token | val pair | mFAVA span |
|---|---:|---:|---:|---:|
| global, без adjuster | 0,3168 | 0,5689 | 0,4267 | 0,2056 |
| global + adjuster | **0,3366** | 0,5713 | 0,4474 | 0,1944 |
| mode, adjuster для всех спанов | 0,3363 | 0,6269 | 0,4872 | 0,6132 (−0,076) |
| **hybrid: adjuster только на global-пути** | **0,3363** | **0,6191** | **0,4872** | **0,6895** |

KEEP в hybrid-виде: единственный механизм v11, который поднял val span без потери mFAVA (+0,0177 span, +0,0205 pair, +0,0009 token).

## 4. Блок 3 — router whole/clause/short

* **Router v1** (25 pair-level признаков): на dev whole-пары уходят в clause 22 из 27, recall whole 5/27, лучший val span 0,146 → reject.
* **Router v2** (head-relative признаки: base_whole vs base_clause, local_whole vs local_clause, gate, support): confusion на dev — whole→whole 10, whole→clause 17, clause→clause 153, clause→whole 4, neg→whole 79. Спаново это дало mFAVA 0,6943 и val 0,2414, но **90 пар отправлены в whole-декодер, из них истинно whole только 10 → 88,9 % ложных whole**. По правилу плана (>15 %) конфигурация **отклонена, не натягивалась**.
* **Dual decode** (обе гипотезы сразу: top-1 base + top-1..2 local): лучший val 0,2542, mFAVA 0,5872, combined 0,5374 → reject (precision падает быстрее, чем растёт recall).

## 5. Блок 4 — честный 3-fold OOF по документам

Пул: RusHallu train 549 + mFAVA 148 + synth 3 563; fold = SHA-256 документа (синтетика не утекает к родителю), stacker-ранкер на 250 деревьях.

| Срез OOF | span | token | pair | FPR |
|---|---:|---:|---:|---:|
| fold 0 | 0,6563 | 0,6850 | 0,7451 | 0,0 |
| fold 1 | 0,7196 | 0,7400 | 0,7854 | 0,0 |
| fold 2 | 0,6888 | 0,7193 | 0,7816 | 0,0 |
| **все пары пула** | 0,6871 | 0,7140 | 0,7694 | 0,0 |
| **только реальные (RusHallu train + mFAVA)** | **0,4714** | 0,6507 | 0,7083 | 0,0 |
| — из них RusHallu train | 0,3200 | 0,6859 | 0,4683 | 0,0 |
| — из них mFAVA train | 0,5536 | 0,6097 | 0,9275 | 0,0 |
| только синтетика | 0,7241 | 0,7390 | 0,7750 | 0,0 |
| val (среднее по 3 фолдам) | 0,1029 | 0,3026 | 0,1779 | 0,1154 |

Guard: `oof_all` — True, `oof_real` — **False**, `val_mean` — **False**, `both_real_and_val` — **False**. Общий OOF 0,6871 — артефакт состава пула (60 % синтетики), реальный OOF 0,4714 и val 0,1029 показывают истину. Held-out не запускался.

## 6. Лучший честный конфиг v11 (что предлагается как кандидат)

`v3 mode gate + boundary adjuster на global-пути` (артефакт `artifacts/boundary_adjust_hybrid_v11.json`):

| Набор | Span | Token | Pair | FPR |
|---|---:|---:|---:|---:|
| val (150) | 0,3363 | 0,6191 | 0,4872 | 0,125 |
| mFAVA train (148) | 0,6895 | 0,7067 | 0,9197 | 0,0 |
| combined | 0,6438 | 0,6854 | 0,8239 | 0,125 |

Прирост к v10 baseline: val span +0,0177, pair +0,0205, token +0,0009, mFAVA без изменений. Договорной guard (0,50/0,50/0,50 при FPR ≤ 0,40) **не пройден**.

## 7. Артефакты и SHA-256

| Артефакт | Байты | SHA-256 |
|---|---:|---|
| `clause_pseudo_v11.jsonl` | 713 039 | `edba48e1825fc8ddf26c0134a85f531958fcd503f37912372971f03cbba80030` |
| `clause_pseudo_v11_manifest.json` | — | `5e8906bb8bb2682d8fc4f3e6a823bbbda70af5583727879f75197f36033ddd65` |
| `rushallu_synth_v11.jsonl` | 29 741 107 | `64b4633ccf8be219cc4c36ce7d25463d0711c293aad140fb76550e9b916a3840` |
| `rushallu_synth_v11_manifest.json` | — | `b6481b7ea3363cc5f5bf9f6e77341e2ca7abc15b6d5c54f5dd3bdd6e971549ec` |
| `listwise_full_v11.json` | 52 817 | `04b9dc8ee61a9718e72e679bbddeec529f38fc771bbf8ff42fce2de7ce78efa3` |
| `listwise_full_model_v11.json` | 524 256 | `13bccce73b3cadd34f5622232176f5f8e89c5a8b97e50c502e4ec0bbe5fe1953` |
| `listwise_pseudo_v11.json` | — | `43db69cf9c5682ce9da1fba8c3c84cf06f00b8e2ab631ae89b9adca3332556d5` |
| `score_compare_v11.json` | 74 247 | `ab95e839195b692b57145ece80025a4cbda171833b32c401d5762a5f7bf62b56` |
| `router_v2_v11.json` | 578 131 | `46bc9e129e16bc729cd3374cccadca06d52ca98cfc5093177bff6dd7966b665b` |
| `stacker_w6_v11.json` | 67 089 | `827b558261b9dd8e026a6a4e33b6becd6676f2c6df04f0696e5db7f374af66ec` |
| `stacker_w6_model_v11.json` | — | `ca7fb2ed53b555632c04cbfdfc2c75d9523420f5455b29f37c45db5518b6a0ae` |
| `boundary_adjust_hybrid_v11.json` | 3 957 | `e5164d64ec970265155f716d61a6cace22dc8ce2551df3b4683ead183209d28c` |
| `cv3_oof_v11.json` | 2 937 | `ad0d51c9d8f72171b04dc965b932b4db6f09153116a693f881c5c20bc5e48625` |
| `dual_decode_v11.json` | 205 543 | `0324761deb87843340c43a348a4e83f678fed3fba98ffbbad46a7b9e28e0dc2b` |

## 8. Команды воспроизведения

```
.venv/bin/python tools/build_clause_pseudo_v11.py
.venv/bin/python tools/build_rushallu_synth_v11.py
.venv/bin/python tools/train_listwise_v11.py --features full --report artifacts/listwise_full_v11.json --model artifacts/listwise_full_model_v11.json
.venv/bin/python tools/train_listwise_v11.py --features support|boundary|no_support --report artifacts/listwise_<set>_v11.json --model artifacts/listwise_<set>_model_v11.json
.venv/bin/python tools/train_listwise_v11.py --with-pseudo --report artifacts/listwise_pseudo_v11.json --model artifacts/listwise_pseudo_model_v11.json
.venv/bin/python tools/score_compare_v11.py
.venv/bin/python tools/router_v11.py
.venv/bin/python tools/router_v2_v11.py
.venv/bin/python tools/train_stacker_v11.py --whole-weight 6
.venv/bin/python tools/decode_v11.py
.venv/bin/python tools/dual_decode_v11.py
.venv/bin/python tools/boundary_adjust_v11.py --decode global|mode|hybrid
.venv/bin/python tools/cv3_oof_v11.py
.venv/bin/python -m pytest tests -q
.venv/bin/python tools/speed_ranker.py
.venv/bin/python tools/build_package.py
```

## 9. Инженерия

* `pytest tests -q`: **142 passed in 22,43 s** (регрессий нет; рантайм не менялся, все v11-скрипты вне пакета).
* Скорость: 10k — 0,0456 с, 50k — 0,3222 с, 2M — 10,2932 с (5,147 мс/1000 симв.).
* Пакет: `spanver-0.2.0.zip`, **523 986 B**, under_limit True, SHA `6bbc50448aeb9deadfb44df35e98023fc0132ac3e59e18c590ace9698ff7d73a` (не менялся).
* CLI: `python -m spanver verify --answer ... --document ... --json` → `{"spans": [], "ai_share": 0.2064, "meta": {...}}`.
* HTTP: `/verify` → 200 JSON, `/` → 200 (5 076 B), offline-режим сохранён.
* ai_share: шаг 0,25 (0 / 0,25 / 0,5 / 0,75 / 1,0), max\|Δ\| = 0,0, монотонно, регенерация микста байт-в-байт (LOO-эквивалент).
* `.exe`: блокер — в песочнице нет Windows/Wine/PyInstaller; build script/CI-инструкция остаются в репозитории.

## 10. Честный блокер и что именно не сработало

1. **Гранулярность gold в RusHallu val не выводится из ответа/документа.** val: 23 whole-пары, 18 clause, 5 short, 104 негатива. Сигнатура «равномерно неподдержанный ответ» (whole) против «один провал» (clause) на train слабая (ratio min/mean 0,461 vs 0,382), на val она сильнее (0,465 vs 0,257) — то есть обучение на train систематически недоучивает именно тот режим, который нужен val.
2. **Router по правилу плана отклонён:** 88,9 % ложных whole при cut 0,6 (нужно ≤15 %). Натягивать не стал.
3. **Document support не селективен на val clause-спанах:** gold clause лишь на 43-м процентиле по support среди clause того же ответа (средний support-score 0,449), поэтому support-правило даёт ≤40 % попаданий в clause-половине — этого мало для Span 0,50.
4. **Головы комплементарны, но их объединение ухудшает precision:** base whole 0,87 / clause 0,222; local clause 0,50 / whole 0,261; fusion и dual decode проигрывают.
5. **Синтетика учит mFAVA-стиль, не RusHallu:** listwise/stacker дают mFAVA 0,78 (лучше baseline 0,6895), но val 0,12–0,19. OOF на объединённом пуле (0,6871) — артефакт доли синтетики; реальный OOF 0,4714.
6. **Единственный реальный плюс v11** — boundary adjuster на global-пути: val span 0,3186 → 0,3363 при сохранении mFAVA. Этого мало: нужно ещё +0,14 span и +0,013 pair.

Что даст прорыв (требует решения заказчика, не инженерии): либо формально согласованное правило разметки, устраняющее смесь whole/clause/short в held-out, либо RusHallu-подобные **ручные** clause-метки объёмом, сопоставимым с mFAVA (≥150 пар), которых в открытом доступе нет. Все цифры получены реальными прогонами; JSON и метрики вручную не правились.
