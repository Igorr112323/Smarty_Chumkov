# Поставка SpanVer 0.2.0

## ВЕСА ИСТЕКАЮТ 17.12.2026

Истекает **Actions-артефакт** `weights47-v5-outer1` (17.12.2026 в 10:11:53 UTC), а не все копии весов. Найдены и проверены обе части V5 seed 0 в ранее существовавшем Git-коммите `f1487214edf239db2e99c6748ac8926927ba0465`: общий SHA-256 `04eb7932a33a7ed1a2f4e166f536b002ba3ad4a16e2a410a21e292e1558d747a`, 116 806 133 байта. Для восстановления нужны доступ к репозиторию и сохранность этого коммита; сохраните проверенную копию отдельно до истечения артефакта. Прямая загрузка Actions-артефакта в этой среде завершилась EOF; его содержимое повторно не проверено. Ниже — команды получения проверенной Git-копии. Лексический режим работает и без весов.

## Назначение и состав

Локальный модуль проверяет русскоязычный ответ по документу и возвращает подозрительные фрагменты (`spans` с границами, риском, типом и уверенностью), `ai_share` и служебные сведения `meta`. Концы фрагментов включительные. Интерфейсы: CLI, HTTP `POST /verify` и локальная веб-страница; все работают с одним выбранным backend. Это не гарантия истинности ответа.

**Архив кода** `spanver-0.2.0.zip`: 338 409 байт (330,5 КиБ), SHA-256

```
a741988762b7b447f6cd50b98924fa3b81b91f3ee66a4cd54589b69c6ac1c3ad
```

В архив входят Python- и JSON-файлы `spanver/` и `README.txt`. Наборы данных, тесты, инструменты сборки, `artifacts/tiny_head.json`, базовая модель, checkpoint и зависимости в него **не входят**. Сам ZIP в Git не хранится: ниже приведена команда сборки из исходного дерева. Архив проверен в Actions; нормируемый кодовый архив — не полная установка.

- **Лексический режим** (`lexical`, по умолчанию): Python ≥3.10, стандартная библиотека; работает без весов и torch.
- **Поставляемый модельный путь Tiny V5 seed 0 без вопроса**: дополнительно нужны `torch`, `transformers`, каталог базовой модели `models/rubert-tiny2`, профиль `artifacts/tiny_head.json`, манифест `weights/final/manifest.json` и все части `weights/final/seed_0/`. Checkpoint — 116 806 133 байта, SHA-256 `04eb7932a33a7ed1a2f4e166f536b002ba3ad4a16e2a410a21e292e1558d747a`; базовая модель и зависимости занимают дополнительное место. Tiny проверяет манифест, размер и SHA каждой части, общий SHA и загружает состояние строго; при отсутствии или повреждении checkpoint он не подменяет его случайной моделью.
- Профиль: `gate=0,45`, `start=0,49`, `extend=0,245`, `top_chunks=4`, максимум 3 span, clause expansion. Аргумент CLI `--weights` задаёт **каталог базовой модели**, а не checkpoint; checkpoint по умолчанию берётся из `weights/final`. При нестандартном профиле `--head` может задать другой путь checkpoint, но тогда происхождение и метрики V5 не гарантированы.
- Режим `model` также присутствует в коде, но не является восстановленным V5. Результаты эксперимента B с вопросом не являются интерфейсом поставки.

## Сборка и запуск

Ниже команды для Linux и Bash с GNU `sha256sum` и `base64`, из корня полученного репозитория этой поставки. Сборка не обучает модель и не читает Dataset. Для сверяемой сборки исходники `spanver/` и `tools/build_package.py` должны соответствовать зафиксированным ниже хешам. Требуются Python ≥3.10 и `unzip`.

```bash
python3 tools/build_package.py --out artifacts
printf '%s  %s\n' \
  'a741988762b7b447f6cd50b98924fa3b81b91f3ee66a4cd54589b69c6ac1c3ad' \
  'artifacts/spanver-0.2.0.zip' | sha256sum -c -
mkdir -p delivery-run
unzip -q artifacts/spanver-0.2.0.zip -d delivery-run
cd delivery-run
```

Если контрольная сумма не совпала, не считать архив этой проверенной поставкой. В распакованном каталоге можно сразу запустить **лексический** режим:

```bash
python3 -m spanver verify --answer 'Париж находится во Франции.' \
  --document 'Париж — столица Франции.' --json
python3 -m spanver serve --host 127.0.0.1 --port 8765
```

В другом терминале:

```bash
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/verify -H 'Content-Type: application/json' \
  -d '{"answer":"Париж находится во Франции.","document":"Париж — столица Франции."}'
```

Открыть `http://127.0.0.1:8765/`, ввести ответ и документ, нажать «Проверить». Сервер не содержит аутентификации и TLS; для локального использования указывайте `--host 127.0.0.1` (по умолчанию он слушает все интерфейсы). Остановка — Ctrl+C.

### Подготовка Tiny V5 seed 0

Нужны исходное дерево репозитория, его файлы `models/rubert-tiny2/` и `weights/rubert-tiny2/`, доступ к GitHub-репозиторию, GitHub CLI `gh` с авторизацией и свободное место для весов. Следующие команды выполняются **из корня репозитория**, после распаковки архива как выше; они не запускают обучение или оценку. Создание окружения и загрузка зависимостей требуют сети; сам инференс использует локальные файлы.

```bash
set -euo pipefail
python3 -m venv delivery-run/.venv
delivery-run/.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
delivery-run/.venv/bin/python -m pip install transformers huggingface_hub
mkdir -p delivery-run/artifacts delivery-run/models delivery-run/weights/final
cp artifacts/tiny_head.json delivery-run/artifacts/tiny_head.json
cp -R models/rubert-tiny2 delivery-run/models/
python3 tools/assemble_weights.py weights/rubert-tiny2 delivery-run/models/rubert-tiny2/pytorch_model.bin

# Проверенная Git-копия seed_0; SHA-1 ниже — идентификаторы Git blob, не SHA-256.
mkdir -p delivery-run/weights/final/seed_0
gh api -H 'Accept: application/vnd.github.raw+json' \
  repos/Igorr112323/Smarty_Chumkov/git/blobs/89fc7bfac684af13f4f8a9cc9a5f6c7b330c78f0 \
  > delivery-run/weights/final/seed_0/part-000.bin
gh api -H 'Accept: application/vnd.github.raw+json' \
  repos/Igorr112323/Smarty_Chumkov/git/blobs/169eeb3ac9872ac4185181d52ef7ce88b8ec540d \
  > delivery-run/weights/final/seed_0/part-001.bin
printf '%s  %s\n' \
  '3eb21ac365ab9f2807fd6db872579c671138be124c26aafa6e05a88e03577cd6' \
  'delivery-run/weights/final/seed_0/part-000.bin' \
  '8702e663a46a0d4739191d0ee207d48fb88a76d7500aee8384be92780f59bbba' \
  'delivery-run/weights/final/seed_0/part-001.bin' | sha256sum -c -
total_sha=$(cat delivery-run/weights/final/seed_0/part-000.bin \
                delivery-run/weights/final/seed_0/part-001.bin | sha256sum | cut -d' ' -f1)
[ "$total_sha" = '04eb7932a33a7ed1a2f4e166f536b002ba3ad4a16e2a410a21e292e1558d747a' ]
gh api 'repos/Igorr112323/Smarty_Chumkov/contents/weights/final/manifest.json?ref=f1487214edf239db2e99c6748ac8926927ba0465' \
  --jq .content | base64 -d > delivery-run/weights/final/manifest.json
printf '%s  %s\n' \
  '51ecf33f34ac54481d71d0d5e54f849e33ce8e1399120be49e71aeb174c74709' \
  'delivery-run/weights/final/manifest.json' | sha256sum -c -
```

Команды получают Git-копию из существующего коммита, а не истекающий Actions-артефакт; проверенные файлы не входят в ZIP или текущую ветку. Если доступа к Git-копии и сохранённому `seed_0` нет, Tiny V5 не запустить; лексический режим остаётся доступен. Не заменять checkpoint произвольными весами. Сборщик базовых весов сверяет части и полный SHA `137fa2b1d944dae19c74456dfe8fac2f780d9acf34e037f5d1e37acba1157768`.

Запуск из `delivery-run`:

```bash
cd delivery-run
.venv/bin/python -m spanver verify --backend tiny \
  --answer 'Париж находится во Франции.' --document 'Париж — столица Франции.' --json
.venv/bin/python -m spanver serve --backend tiny --host 127.0.0.1 --port 8765
```

CLI также принимает UTF-8 файлы вместо текстовых аргументов. HTTP и веб при таком запуске используют Tiny. Пути профиля и весов относительны текущему каталогу. Для Tiny `ai_share` — доля символов ответа, покрытых найденными span, без лексической калибровки; не следует смешивать её с `ai_share` лексического режима.

## Что означают результаты

Разные наборы, backend и определения не следует сравнивать как один эксперимент.

| Срез | Числа и статус |
|---|---|
| Текущий Tiny V5 seed 0, 140 внешних train/val, без вопроса | Token F1 **0,479979**, precision 0,390929, recall 0,621566, token FPR 0,393635; exact span F1 0,144330, IoU span F1 0,226804; 41 предсказанный и 56 gold span. Клиентский путь и восстановленный V5 совпали **140/140**, а 12 проверенных агрегатов согласованного `cli52.json` совпали точно. Это не независимый test. |
| B с вопросом, 699 train/val, nested CV | Token F1 **0,516522**, precision 0,436775, recall 0,631894, FPR **0,407631**; baseline «пометить всё» token F1 0,500132, точечная разность +0,016390. Парный 95% bootstrap CI разности **[−0,033882; +0,063053]**, 10 000 повторов, seed 67: включает ноль, **значимый отрыв не подтверждён**. Это не доказывает нулевой истинный эффект. В 6/10 outer FPR >0,40, максимум 0,575071; скорость B не измерена. B закрыт и не является поставляемым интерфейсом. |
| A, 699 train/val | Token F1 **0,508888**, FPR **0,378691**, baseline 0,500132. Checkpoint A утрачен; его нельзя отождествить с восстановленным V5. |

### B: 10 внешних фолдов (699 train/val)

Это сохранённые **outer**-результаты B с вопросом; пороги выбраны на inner. FPR >0,40 у фолдов 3, 5, 6, 7, 8 и 9.

| Outer | Порог start | Token FPR | Token F1 |
|---:|---:|---:|---:|
| 1 | 0,49 | 0,255319 | 0,459237 |
| 2 | 0,53 | 0,243730 | 0,531137 |
| 3 | 0,51 | 0,547570 | 0,311664 |
| 4 | 0,49 | 0,294538 | 0,617693 |
| 5 | 0,50 | 0,434851 | 0,522263 |
| 6 | 0,50 | 0,412236 | 0,493470 |
| 7 | 0,49 | 0,460105 | 0,592267 |
| 8 | 0,43 | 0,575071 | 0,511760 |
| 9 | 0,45 | 0,429204 | 0,597535 |
| 10 | 0,46 | 0,378360 | 0,508197 |

Три уровня B и **whole-answer** baseline на одних 699 парах, одним `spanver.evaluate.metrics`, soft совпадение при IoU ≥0,5:

| Уровень F1 | B | baseline | B − baseline |
|---|---:|---:|---:|
| Token | 0,5165 | 0,5001 | +0,0164 |
| Pair, soft hit | 0,3696 | 0,3572 | +0,0124 |
| Span, soft IoU ≥0,5 | 0,2944 | 0,3043 | −0,0099 |

В pair F1 положительная пара считается TP только при совпавшем span; это не обычный бинарный pair all-positive F1. Exact span F1 B = 0,2581, exact pair F1 = 0,3316. Разности — точечные оценки, а не доказанные улучшения.

### Пункты 6–9

| Пункт | Факт и статус |
|---|---|
| **6 — качество** | A на 699: token F1 0,508888 и FPR 0,378691; пользователь считает эту CV-строку формальным выполнением «по букве». Независимый test не подтверждён. В историческом `compliance52.json` п.6 отмечен **«не выполнено»** по другому span-определению: F1 0,226804, pair FPR 0,208333. Tiny V5 на 140: token F1 0,479979, token FPR 0,393635. Это разные определения и срезы. |
| **7 — скорость** | Лексический модуль: 463 мс на 50 000 знаков = 9,26 мс/1000. Восстановленный V5: **28,84 мс/1000 знаков документа**, при норме 30 — проходит, **без резерва**; измерено на раннере без загрузки модели. На другом раннере текущий путь V5 20,9559 мс/1000; разность раннеров не означает ускорения модели. Скорость B с вопросом не установлена. |
| **8 — интерфейсы** | CLI, HTTP и веб присутствуют. Изолированный pytest: **263 passed, 1 failed, 2 skipped**, всего 266; провал калибровки фактический, пользователь принимает ограничение как не блокирующее сдачу. Полный штатный CI с общим gold не запускался. |
| **9 — размер** | Исторический архив **287,3 КБ (294 176 байт)**; новый архив кода **338 409 байт (330,5 КиБ)**. Это разные снимки, оба меньше 600 000 байт и лимита сборщика 614 400 байт. Веса и зависимости отдельно; полная установка существенно больше архива. |

## Ограничения, тесты и изоляция

- Test 301 исторически **не полностью слепой**; приведённые оценки основаны на train/val. При случайном push-run `cli52` [45151353019](https://github.com/Igorr112323/Smarty_Chumkov/actions/runs/45151353019) был полный checkout, шаг data завершился ошибкой; логи недоступны, поэтому **возможно чтение test 301**, установить это нельзя. Результаты инцидента не использовались, обучение и подбор по test не проводились. Впоследствии push-триггеры проверены: в текущей опубликованной ветке ни один из 33 workflow не запускается от push; ручные и другие события не запрещены.
- Checkpoint A утрачен. Восстановленный V5 проверен отдельно; тождества A и V5 не доказано. Разметка выполнена одним человеком, независимого подтверждения нет.
- В лексической калибровке при истинной доле 0,25 получено 0,1999: ошибка **0,0501 при допуске 0,05**, превышение 0,0001. Тест остаётся failed; константы и допуск не подгонялись. Два теста split64 пропущены, поскольку требуют test 301; исходные проверки не менялись. Проверка вопроса на 699 train/val не подтверждает импорт исходного CSV.
- Работа по метрикам закрыта. CI с нулём не доказывает отсутствия эффекта. Отдельно утерянный исторический поэлементный эталон заменён пользователем на агрегаты `cli52.json`.

## Файлы-основания и SHA-256

Ниже SHA-256 **содержимого файлов**, не git blob SHA. Они фиксируют источники приведённых сведений; основные условия и результаты уже изложены выше. Команда проверки: `sha256sum <путь>` из корня репозитория.

| Файл | SHA-256 | Основание |
|---|---|---|
| `artifacts/review67.md` | `1d63d8c147d0d39623f04405dc7cdc7d82f06894ce5cdc37914ddc950b39562a` | Итог, границы и п.6–9 |
| `artifacts/regression67_fixed_verify.json` | `5cf39ac3199505f37d9fcce77f0257ea7b6ff91c306e45df9f36e0386e63b282` | V5, 140/140 и агрегаты |
| `artifacts/regression67_fixed_tests.json` | `e79a8927570f638bf682b94be444d0cf8b8203fb1df81045a150bb60fb6fbae1` | Тесты, калибровка и skip |
| `artifacts/regression67_fixed_package.json` | `cc961b9a5efb1e2a842f8aad29606ede1e27f031a5fe2e85c819a2d2e384cf35` | Текущий архив и SHA |
| `artifacts/recalculated67_b.json` | `98a6d861605be94206e089685f291498ce5364c17d31ef92ee117fcca3546d32` | B, bootstrap и три уровня |
| `artifacts/v5_weights_provenance.json` | `6ef6628689f2cd580a6b313d2fde07ba01828c1ee386bce309eacbc243d9197b` | Доступность, проверка и происхождение V5 |
| `artifacts/review69.md` | `ef8503531c162ef6728fab678810b66173310c4cfb2016cc0c48d4ed0fe017c1` | Таблица сверки результатов и ограничения аудита |
| `artifacts/question59.json` | `49b9b7ac2db84f76434ae92fcd113c210a65bf08086d24d3c3d1d46bbd7ef2f0` | Результаты A на 699 train/val |
| `artifacts/regression67_repeat_verify.json` | `c65f0e219c351dd8e5554fc93007e0157b2f989a77c82fce5c55aed70b75d550` | Историческая скорость V5 без загрузки |
| `artifacts/cli52.json` | `ef7c4d7c329ac3436dc777c4f6a588d7bb4f3874af82f244a5dda8065a9d57f0` | Согласованный агрегированный эталон V5 |
| `artifacts/compliance52.json` | `76acca9cbad187311d67949c7a0438eeb37636a1b64e5a2a3d80cfe2a60f27d3` | Исторические статусы и размер архива |
| `artifacts/regression67_recheck.md` | `9761e449314bcdaf0a8d6921e74b0d78b359d4507af4daa9200a5e4aeac54dec` | Исторический размер и замер V5; промежуточный отчёт, не итоговый статус |
| `artifacts/module52_report.json` | `841384e8657262d4a4fcefef89f13827cf63f13f9476a6239d2f8cc8acc29dcc` | Историческая лексическая скорость |
| `artifacts/tiny_head.json` | `41d550268b0ab3367b899ea351b990f167e11a46e8a036cc38df37a77ce0112a` | Профиль Tiny |
| `docs/incident67_cli52.md` | `ecb0cc479ecc8f67f6b0346c15a47636b3201672f741f7b51b9c3543bf058b2d` | Инцидент изоляции |
| `docs/tiny67_delivery.md` | `aa95eba4ccb96541811fe2be95bad202203aa64061f5ed01b800bf1a800d56b8` | Происхождение и загрузка V5 |
| `package.sha256` | `1ed73837efccc4dcaa5d5b3bde83d4af62394024a6083f4ade3928d1f1d73857` | Файл записи SHA, не SHA самого ZIP |
| `tools/build_package.py` | `e3333828355b3accb7ee8c89f3a756c802a1e7a21c61df3df784e4fb6c282695` | Детерминированный сборщик |
| `spanver/cli.py` | `d58854358d195a3289186286b04c8b959538e61478d5e3f3827b4a1dee761f6a` | CLI |
| `spanver/api.py` | `9fcd094975161f3e32f684d3bfa86f09e4c774ffad3055931ce7869e40a9a23b` | HTTP и веб |
| `spanver/tiny_backend.py` | `d6668433869ec8a6e79011498270d41b75a5d1948c2848b6d8371707252f0cea` | Проверка и загрузка V5 |
| `weights/rubert-tiny2/manifest.json` | `30e63b67ba9861eb943ea0bc5f934cf90fa246a94b1ae8c211df078fff0a6ccb` | Части базовой модели |
| Внешний `weights/final/manifest.json` | `51ecf33f34ac54481d71d0d5e54f849e33ce8e1399120be49e71aeb174c74709` | Манифест V5 из зафиксированного commit `f1487214edf239db2e99c6748ac8926927ba0465` |

Результаты проверки V5, тестов и пакета: [Actions 45151350705](https://github.com/Igorr112323/Smarty_Chumkov/actions/runs/45151350705). Расчёт B: [Actions 45152394711](https://github.com/Igorr112323/Smarty_Chumkov/actions/runs/45152394711). Документ подготовлен без новых обучений, тестовых запусков или пересчёта метрик.
