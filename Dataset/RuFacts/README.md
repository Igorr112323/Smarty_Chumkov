# RuFacts (span-разметка)

Источник: Kaggle, соревнование internal-fact-checking-for-the-russian-language.
Файлы: train_with_spans.json (4677 записей), val_with_spans.json (1559 записей).
Формат: JSONL, одна запись на строку. Поля: idx, evidence, claim, evidence_facts, claim_facts, label.
Координаты range = [start, end), конец НЕ включается.
Спаны только в claim_facts (в ответе). Длина спанов: min=1, медиана=2, максимум=15 символов.

Права принадлежат организаторам соревнования. Файлы размещены для воспроизводимости обучения.
