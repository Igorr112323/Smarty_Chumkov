# Сохранённый ансамбль tiny2

Пять checkpoint соответствуют заранее выбранному внешнему фолду 1, а не пяти внешним моделям CV. Порог start 0,52, extend 0,26. Сборка весов и воспроизведение метрик этого фолда проверены в GitHub Actions; итоговые CV-метрики не являются метриками этой одной модели. Test 301 повторно не оценивался.

Работать из корня репозитория в PowerShell, Python 3.10 или новее. В `ваши_пары.jsonl` должны быть реальные входные пары с уникальными `pair_id`, непустым `model_output` и текстом документа в `document_text`. Файл test 301 не использовать.

```powershell
chcp 65001
```
ДОЛЖНО БЫТЬ В ВЫВОДЕ: `65001`.

```powershell
python -m pip install torch 'transformers>=4.50,<5'
```
ДОЛЖНО БЫТЬ В ВЫВОДЕ: успешная установка или сообщение, что зависимости уже установлены.

```powershell
python tools/assemble_weights.py weights/rubert-tiny2 models/rubert-tiny2/pytorch_model.bin
```
ДОЛЖНО БЫТЬ В ВЫВОДЕ: `assembled` и `official_match True`.

```powershell
python tools/predict_frozen38.py --input ваши_пары.jsonl --output предсказания.jsonl
```
ДОЛЖНО БЫТЬ В ВЫВОДЕ: `готово` и число обработанных пар. Загрузчик проверяет SHA-256 каждой части и каждого checkpoint до инференса.
