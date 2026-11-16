import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nested_cv35 import predict


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    tested = 0
    for fold in range(1, 6):
        source = read(ROOT / f"artifacts/predictions_fold{fold}.jsonl")
        outputs = {mode: {row["pair_id"]: row for row in read(ROOT / f"artifacts/snap42_fold{fold}_mode{mode}.jsonl")} for mode in ("B1", "B2", "B3", "B4")}
        for row in source:
            cached = {"answer": row["answer"], "pieces": row["pieces"]}
            baseline = predict(cached, row["start_threshold"])
            if baseline != row["predicted_spans"]:
                raise ValueError("декодер не воспроизвёл baseline")
            for mode in outputs:
                result = predict(cached, row["start_threshold"], mode)
                if result != outputs[mode][row["pair_id"]]["predicted_spans"]:
                    raise ValueError(f"несовпадение интеграции {mode}")
                tested += 1
    if tested != 699 * 4:
        raise ValueError("неверное число проверенных предсказаний")
    print(f"совпали baseline и {tested} результатов B1/B2/B3/B4; без инференса", flush=True)


if __name__ == "__main__":
    main()
