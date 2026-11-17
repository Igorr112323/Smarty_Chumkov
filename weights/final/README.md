# Ансамбль В5, внешний фолд 1

Пять checkpoint сидов 0–4 обучены для внешнего фолда 1 и внутреннего фолда 2. Это **один ансамбль**, а не все пять ансамблей CV: macro F1 0,5014017611077594 и сводный F1 0,4998734625702283 описывают пять разных ансамблей на 699 парах, а не эту модель. В5 выбран при равенстве сводного F1 с В4. Порог start 0,50; клаузный expand сохранён. SHA-256 частей и собранных файлов — в манифестах; части не более 80 МиБ. Независимое выполнение договора не установлено. Остальные CV-ансамбли находятся в `weights/ablation47/v5/`. Исторический ансамбль сохранён в истории Git и `weights/frozen/`.

Работать из корня репозитория. В входном JSONL нужны уникальные `pair_id`, непустой `model_output` и `document_text`.

```bash
python -m pip install torch 'transformers>=4.50,<5'
python tools/assemble_weights.py weights/rubert-tiny2 models/rubert-tiny2/pytorch_model.bin
python tools/predict_frozen38.py --weights weights/final --input ваши_пары.jsonl --output предсказания.jsonl
```

Загрузчик проверяет SHA-256 каждой части и каждого checkpoint. Результат повторной проверки всего ансамбля на его внешнем фолде: `artifacts/final47_restore_verification.json`.
