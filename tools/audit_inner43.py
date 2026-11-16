import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def search_inner_prediction_keys(value, path=""):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = f"{path}.{key}" if path else key
            if re.search("inner", next_path, re.I) and re.search("prediction|probabilit|piece|cache", key, re.I):
                result.append(next_path)
            result.extend(search_inner_prediction_keys(child, next_path))
    elif isinstance(value, list):
        for child in value[:1]:
            result.extend(search_inner_prediction_keys(child, path + "[]"))
    return sorted(set(result))


def inspect(path, outer, inner):
    raw = path.read_bytes()
    record = json.loads(raw)
    if record.get("status") != "готово" or record.get("outer_fold") != outer or record.get("inner_fold") != inner:
        raise ValueError("неверный или неполный исходный артефакт")
    grid = record.get("inner_grid") or []
    if not grid:
        raise ValueError("нет сохранённой внутренней сводки")
    nested_keys = search_inner_prediction_keys(record)
    return {"outer_fold": outer, "inner_fold": inner, "sha256": hashlib.sha256(raw).hexdigest(), "root_keys": sorted(record), "inner_grid_rows": len(grid), "inner_grid_columns": sorted(grid[0]), "candidate_inner_prediction_keys": nested_keys, "outer_predictions_present": "outer_predictions" in record, "outer_prediction_pairs": len(record.get("outer_predictions") or {}), "seed_count": len(record.get("seeds") or [])}


def main():
    results = []
    for outer in range(1, 6):
        for inner in range(1, 6):
            if inner != outer:
                results.append(inspect(ROOT / f"artifacts/nested38_outer_{outer}_inner_{inner}.json", outer, inner))
    candidates = [row for row in results if row["candidate_inner_prediction_keys"]]
    output = {"source_run": 43914581442, "jobs_checked": len(results), "jobs_with_candidate_inner_prediction_keys": len(candidates), "jobs_with_outer_predictions": sum(row["outer_predictions_present"] for row in results), "scope": "только структура 20 ранее сохранённых JSON-файлов; без данных test, обучения и инференса", "limitation": "Отсутствие полей вероятностей внутренних фолдов в этих артефактах не доказывает, что они не сохранены где-то вне репозитория и run.", "records": results}
    target = ROOT / "artifacts/inner43_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
