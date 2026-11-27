# V18 — финальная охота за реальными русскими клаузовыми span-данными: данных нет, блокер

## Итог одной строкой

Проверены **все** доступные снаружи источники и весь GitHub: 15 внешних хостов (Kaggle, Zenodo,
figshare, Mendeley, GitLab, Bitbucket, SourceForge, arXiv, data.world, Yandex Cloud, Sber,
HuggingFace, hf-mirror, ModelScope, текстовый прокси) недоступны из песочницы, 16 repository- и
5 code-запросов по GitHub плюс обход деревьев 10 репозиториев дали 8 кандидатов — и **ни один**
не содержит реальных русских клаузовых span-аннотаций с `doc_overlap = 0`. Реальные
человеческие span-аннотации есть только у RAGTruth (английский, уже проверен в v14: ухудшает
метрики) и у нашего же RusHallu-RAG (`doc_overlap = 100 %`).

Поэтому честный итог по договору: **закрыто Token 0.6247, Pair 0.5122, FPR 0.1442, скорость,
пакет, тесты; не закрыт Span 0.3621 против требуемых 0.50 → финальный reject.**

## ШАГ 1. Все попытки поиска — `artifacts/ru_clause_search_v18.json`

### Внешние хосты (реальные HTTP-ответы, 8 с таймаут)

| хост | ответ | что искали |
|---|---|---|
| `kaggle.com/api/v1/datasets/list?search=russian hallucination` | `TLS/SSL connection has been closed (EOF)` | ru hallucination / ru rag datasets |
| `zenodo.org/api/records?q=russian hallucination` | `TLS EOF` | научные дополнения (span-level RU) |
| `figshare.com/api/v2/articles?search_for=russian+hallucination` | `TLS EOF` | датасеты статей |
| `data.mendeley.com/public-api/datasets` | `TLS EOF` | дополнения к статьям |
| `gitlab.com/api/v4/projects?search=russian hallucination` | `TLS EOF` | зеркала/форки |
| `api.bitbucket.org/2.0/repositories` | `TLS EOF` | зеркала |
| `sourceforge.net/directory/?q=russian dataset` | `TLS EOF` | архивы |
| `export.arxiv.org/api/query` | `TLS EOF` | приложения к статьям |
| `api.data.world/v0/search` | `TLS EOF` | открытые датасеты |
| `cloud.yandex.ru/docs/datasets/` | `TLS EOF` | Yandex datasets |
| `sber.ru` | `TLS EOF` | Sber datasets |
| `huggingface.co/api/datasets?search=russian hallucination` | `TLS EOF` | HF-датасеты |
| `hf-mirror.com/api/datasets?search=ru rag` | `TLS EOF` | зеркало HF |
| `modelscope.cn/api/v1/datasets` | `TLS EOF` | ModelScope |
| `r.jina.ai/https://huggingface.co/datasets?...` | `TLS EOF` | текстовый прокси к HF |
| `pypi.org/pypi/ruhallu/json` | `404` (хост доступен) | датасет как пакет на PyPI |
| `registry.npmjs.org/russian-hallucination-dataset` | `404` (хост доступен) | датасет как npm-пакет |

Доступны в песочнице только `github.com`, `codeload.github.com`, `api.github.com`, `pypi.org`,
`files.pythonhosted.org`, `registry.npmjs.org` — на PyPI и npm датасетов нет (404).

### GitHub: что искали и что нашли

16 repository-запросов (`russian hallucination dataset`, `RuHalu`, `ru hallucination`,
`russian factuality dataset`, `russian fact verification dataset`, `ru fact checking dataset`,
`RAGTruth russian`, `russian RAG benchmark`, `ru rag dataset`, `clause hallucination annotation`,
`hallucination spans dataset`, `ru span annotation dataset`, `Russian NLI dataset`,
`russe fact checking`, `CheRRI russian`, `russian claim verification`), 5 code-запросов
(`ру галлюцинации разметка spans`, `hallucination russian jsonl spans`, `ru_hallucination filename:data`,
`RAGTruth filename:ru`, `factuality russian filename:jsonl`), обход деревьев 10 репозиториев —
всё в артефакте с реальными ответами API.

## ШАГ 2. Проверка стиля найденных кандидатов — `artifacts/ru_clause_data_v18.json`

| репозиторий | звёзд / размер | что внутри | стиль разметки | вердикт |
|---|---|---|---|---|
| `s-nlp/PsiloQA` | 16 / 1.3 MB | пайплайн мультиязычного span-level датасета (14 языков), в репо только CSV аннотаторов и `evaluation/gpt.csv` | span-level, но **автоматическая LLM-аннотация** и **данные на HF** (недоступен) | reject: данных в репозитории нет |
| `sb-ai-lab/TOHA` | 2 / 4.8 MB | CoQA (en) + TruthfulQA + BLEURT-скоры, topological hallucination detection | span-аннотаций нет, русских данных нет | reject |
| `slivka83/ru_rag_test_dataset` | 18 / 41 MB | RuBQ 2.0 paragraphs/test | retrieval-данные, span-аннотаций нет | reject |
| `AndreisLavrov/LegalBenchRu-RAG` | 1 / 44 MB | юридический RAG (RAG_data.csv, cleaned_dataset.csv) | ответы без span-координат | reject |
| `TheNarhawl/RAG-Scope-RU` | 10 / 77 MB | метрики абляций RAG (JSON-логи экспериментов) | аннотаций нет | reject |
| `DFKonGH/ru_hallucination_detector` | 0 / 23 MB | 50 364 пар, `ds_v0.parquet` | **синтетические флипы предложений** (number/entity/negation) | уже использован в v16–v17: Group-CV 0.98 in-domain, перенос на реальные клаузы отрицательный (val AUC 0.60–0.69) |
| `sber-summer-practice/hybrid-rag-hallucination-verification` | 0 / 0.15 MB | 88 КБ xlsx ручной разметки | метки ответа без span-координат | reject |
| `ParticleMedia/RAGTruth` | — / 37 MB | `dataset/response.jsonl`, `source_info.jsonl` | **реальные человеческие span-аннотации, английский** | уже использован в v14 как en-проекция: Span 0.30 < baseline 0.3363 |

Сравнение стиля с нашим val (RusHallu-RAG, 46 позитивов): **23 whole / 18 clause / 5 short**.
У доступных русских кандидатов либо 100 % синтетических флипов предложения, либо вообще нет
span-координат — то есть клаузового стиля, как в нашем датасете, в открытом доступе нет.
`doc_overlap` проверять не пришлось: ни один кандидат не содержит span-разметки неподтверждённых
фрагментов (для RusHallu-RAG он 100 %, для `ru_hallucination_detector` — 0, но стиль синтетический).

## ШАГ 3. Почему это и есть блокер (измерено в v9–v17)

| факт | значение | источник |
|---|---|---|
| кандидат-генерация | 54/54 gold-спанов val имеют кандидат с IoU = 1.0 | v16 |
| Span F1 при идеальном ранжировании в паре (текущий гейт) | 0.6165 | v16 |
| AUC штатного риска v11 на val | 0.7425 | v17 |
| лучший переобученный ранжировщик (pairwise-логит на 49k+наших) | AUC 0.6921, val Span 0.1837 | v17 |
| Group-CV по документам на новых парах (in-domain) | **AUC 0.9814** | v17 |
| те же модели на наших реальных парах | AUC 0.60–0.69 | v16–v17 |
| OOF по документам (hgb_61) | Span 0.0426 против baseline 0.2437 | v17 |
| обучение cross-encoder на CPU | 2.36 с на батч 16×384 при **2 ядрах** → ~50 мин/эпоху; инференс 0.068 с/кандидат ≈ 0.6 с/пара (в 200× выше лимита) | v17 |

Вывод: чтобы закрыть Span ≥ 0.50, нужен ранжировщик с AUC ≳ 0.9 **в кросс-домене** (на реальных
клаузовых галлюцинациях). Доступные данные дают 0.98 in-domain и 0.65 cross-domain; GPU нет,
реальных русских клаузовых span-данных с `doc_overlap = 0` в открытом доступе — нет.

## ШАГ 5. Итог по договору

| метрика договора | требование | факт (лучший честный результат) | статус |
|---|---|---|---|
| Span F1 | ≥ 0.50 | **0.3621** | ❌ |
| Token F1 | ≥ 0.50 | 0.6247 | ✅ |
| Pair F1 | ≥ 0.50 | 0.5122 | ✅ |
| FPR | ≤ 0.40 | 0.1442 | ✅ |
| ai_share | 0–1, шаг 0.25 | не менялся (калибровка `spanver.module.calibrate_share`) | ✅ |
| скорость 1k / 50k / 2M | 0.03 с / 1 с / 60 с | 0.0089 / 0.1695 / 8.8141 (рантайм) · 0.0070 / 0.9920 / 30.2318 (LM int8) | ✅ |
| пакет | ≤ 0.6 МБ | 523 986 B, SHA `6bbc5044…d73a` | ✅ |
| тесты / CLI / HTTP / web | 142 passed, localhost:8765 | 142 passed, `/` и `/verify` 200 | ✅ |
| held-out 1200 пар | не трогали | не читался ни в одном цикле v9–v18 | ✅ |

**Финальный вердикт: договор закрыт на 6 из 7 пунктов, Span остаётся открытым. Reject по Span,
без натяжек.**

## История циклов (что пробовали и почему reject)

| цикл | что делали | результат |
|---|---|---|
| v1–v8 | признаки, калибровка, декодеры, переранжирование | Span рос до 0.30, Pair до 0.48; зафиксировано, что кандидаты хороши |
| v9 | аудит потолка (oracle Span 0.91), neural pointer | neural-pointer на CPU недостаточен, reject |
| v10 | pairwise/relative decode, whole-gate | без выигрыша, reject |
| v11 | mode-гейт + boundary adjuster + local_hgb | **baseline: Span 0.3363 / Token 0.6191 / Pair 0.4872 @ FPR 0.125** |
| v12 | каскад pair + span, экспорт деревьев | метрики не выросли, reject |
| v13 | чужой код (union-merge, sentence-snap) | ухудшение, reject |
| v14 | чужие данные (RAGTruth en-проекция, 958 пар) | Span 0.3030, reject |
| v15 | веса rubert-tiny2 из int8-ONNX + 10 LM-признаков | Span 0.2549 (кандидатный декод), reject |
| v16 | новые русские данные (49k синтетических флипов) + калибровка в FPR ≤ 0.40 | **Span 0.3621 / Token 0.6247 / Pair 0.5122 / FPR 0.1442** — Pair/Token закрыты |
| v17 | 14 span-level признаков кросс-внимания + pairwise/listwise | AUC 0.6921 < 0.7425, OOF Span 0.0426, reject |
| v18 | последняя охота за реальными русскими клаузовыми span-данными | данных нет, финальный блокер |

## Что нужно, чтобы закрыть Span (без иллюзий)

1. **Реальные русские клаузовые span-аннотации с `doc_overlap = 0`** (аналог RAGTruth/RusHallu на
   другом корпусе). Такого открытого датасета нет: проверены GitHub (16+5 запросов, 10 деревьев),
   Kaggle, Zenodo, figshare, Mendeley, GitLab, Bitbucket, SourceForge, arXiv, data.world,
   Яндекс, Sber, HF/зеркала, PyPI, npm.
2. **GPU + cross-encoder**: обучение `rubert-base`/`rubert-tiny2` как cross-encoder на имеющихся
   данных даёт шанс поднять кросс-доменный AUC, но на 2 ядрах это ~50 мин/эпоху и 0.6 с/пара на
   инференсе (лимит скорости §4.1 нарушается в 200×; понадобится квантование + каскад).
3. **Смягчение требования по Span в договоре** (например, Span ≥ 0.35 при Token/Pair ≥ 0.50) —
   решение не техническое, а продуктовое: рантайм уже отдаёт корректные фрагменты в 36 % случаев
   при Pair F1 0.51 и FPR 0.14.

## Артефакты

| файл | что внутри |
|---|---|
| `artifacts/ru_clause_search_v18.json` | все попытки: 15 хостов с реальными HTTP-ответами, 16 repository- и 5 code-запросов GitHub, 10 деревьев, 10 локальных файлов данных с SHA-256 |
| `artifacts/ru_clause_data_v18.json` | проверка 8 найденных кандидатов: лицензия, размер, файлы данных, стиль разметки, вердикт и причина отказа |
| `artifacts/span_oof_v17.json`, `ranker_train_v17.json`, `ranker_features_v17.json` | последние измеренные метрики ранжирования (AUC, Group-CV, OOF) |
| `artifacts/span_oof_v16.json`, `calibration_v16.json` | лучший честный результат по договору (Span 0.3621 / Pair 0.5122) |
| `artifacts/speed_v16.json`, `speed_v17.json` | скорость с весами (int8-ONNX) и с span-level вниманием |
| `tools/download_data_v18.py`, `check_candidates_v18.py` | скрипты поиска и проверки кандидатов |
