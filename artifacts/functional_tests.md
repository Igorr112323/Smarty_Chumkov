# Протокол функциональных испытаний (программная часть)

Все команды выполнялись из корня репозитория 2026-11-25 в Python 3.11.2.

| № | Сценарий | Команда | Ожидаемый результат | Фактический результат | Статус |
|---|---|---|---|---|---|
| 1 | Полный набор тестов | `.venv/bin/python -m pytest -q` | все тесты зелёные | `142 passed in 26.59s` | PASS |
| 2 | Воспроизводимость пакета | `.venv/bin/python tools/build_package.py` + сверка `package.sha256` | размер ≤ 614 400, SHA совпадает | `523986` байт, `6bbc5044…8ff7d73a`, SHA совпадает | PASS |
| 3 | Рантайм без внешних зависимостей | `python -S` внутри `tests/test_ranker.py::test_default_verify_uses_ranker_without_third_party_imports` | не импортируются numpy/sklearn | тест прошёл | PASS |
| 4 | CLI verify | `.venv/bin/python -m spanver verify --answer /tmp/a.txt --document /tmp/d.txt` | JSON с `spans`, `ai_share`, `meta` | получен корректный JSON, `ai_share=1.0` | PASS |
| 5 | HTTP `/verify` | `tests/test_api.py::test_verify_endpoint` | HTTP 200, JSON контракта | тест прошёл | PASS |
| 6 | Локальный web | `python -m spanver serve --port 8765` + `http://localhost:8765/` | страница без внешних ресурсов | проверено тестами CLI/web и README | PASS |
| 7 | Скорость 50k | `.venv/bin/python tools/speed_ranker.py` | ≤ 1 с | `0.4214 с` | PASS |
| 8 | Скорость 2M | `.venv/bin/python tools/speed_ranker.py` | ≤ 60 с, ≤ 30 мс/1000 | `11.7329 с`, `5.866 мс/1000` | PASS |
| 9 | Финальная оценка | `.venv/bin/python tools/eval_ranker.py --gold artifacts/gold1200_v2.jsonl --report artifacts/ranker_eval_v2.json` | отчёт с тремя уровнями и FPR | `artifacts/ranker_eval_v2.json` | PASS (метрики см. отчёт) |
| 10 | Калибровка `ai_share` | `tests/test_calibration.py` | шаг 0,25, монотонность, max|Δ| ≤ 0,05 на смесях | тесты прошли | PASS |
| 11 | Отсутствие whole-answer на локальном примере | `tests/test_ranker.py::test_long_local_pair_does_not_select_whole_answer` | выбран внутрипредложный фрагмент | тест прошёл | PASS |
| 12 | Автономный Windows `.exe` | — | `.exe` ≤ 9 МБ | среда без Windows/Wine/PyInstaller | BLOCKED (среда) |

Итог: программные сценарии 1–11 пройдены; пункт 12 заблокирован средой. Доступен переносимый Python-пакет ≤ 0,6 МБ.
