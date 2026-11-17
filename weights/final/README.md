# Точка A: сохранённый ансамбль

Здесь сохранены **исходные** пять checkpoint сидов 0–4 из nested38 для внешнего фолда 1 и внутреннего фолда 2. Это один ансамбль, а не пять моделей внешней CV. Восстановление исходных частей и метрик фолда 1 уже проверено: `artifacts/frozen38_verification.json`. Средний CV F1 0,4506 относится к пяти разным ансамблям; он не является метрикой этих пяти весов. Порог start 0,52 и extend 0,26. SHA-256 частей и собранных файлов — в `manifest.json`; части не более 80 МиБ.

Работать из корня репозитория, Python 3.10 или новее. В `ваши_пары.jsonl` нужны реальные входные пары с уникальными `pair_id`, непустым `model_output` и текстом документа в `document_text`. Test 301 не использовать.

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
python tools/predict_frozen38.py --weights weights/final --input ваши_пары.jsonl --output предсказания.jsonl
```
ДОЛЖНО БЫТЬ В ВЫВОДЕ: `готово` и число обработанных пар. Загрузчик проверяет SHA-256 каждой части и каждого checkpoint перед инференсом.
