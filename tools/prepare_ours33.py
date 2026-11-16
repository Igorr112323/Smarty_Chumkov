import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.preflight_13 import id_digest, our_rows

EXPECTED_IDS = "22fd4f1dc99bb96c59ed644725690be9d7d7f5fbeac7cb1189b7bc237c088da1"


def main():
    pairs = our_rows()
    if len(pairs) != 699 or id_digest(pairs) != EXPECTED_IDS or sum(len(row["spans"]) for row in pairs) != 300:
        raise ValueError("наши train/val не совпадают с проверенными 699 парами")
    for row in pairs:
        answer = row["model_output"]
        if not answer or any(not 0 <= span["start"] <= span["end"] < len(answer) for span in row["spans"]):
            raise ValueError("некорректная пара или координаты")
    target = ROOT / "artifacts/ours33.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pairs).encode("utf-8")
    target.write_bytes(payload)
    print(f"train/val: {len(pairs)} пар, 300 спанов")
    print(f"pair_id sha256 {id_digest(pairs)}")
    print(f"ours33 sha256 {hashlib.sha256(payload).hexdigest()}")
    print("Полный gold.jsonl не строился, чтобы не обрабатывать test")


if __name__ == "__main__":
    main()
