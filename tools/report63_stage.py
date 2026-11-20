"""Печать стадии, на которой остановился фолд (для разбора падений в Actions)."""
import json
import sys
from pathlib import Path


def main():
    outer, inner, seed = sys.argv[1], sys.argv[2], sys.argv[3]
    path = Path(f"artifacts/seg63_outer{outer}_inner{inner}_seed{seed}.json")
    if not path.is_file():
        print(f"отчёта нет: {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        print(f"отчёт не читается: {error}")
        return
    print(f"стадия: {data.get('stage')}")
    print(f"строк обучения: {data.get('training_rows')}, положительных: {data.get('positive_rows')}")
    print(f"статус: {data.get('status')}")


if __name__ == "__main__":
    main()
