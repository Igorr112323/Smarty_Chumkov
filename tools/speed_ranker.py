"""Замер скорости ранкера на длинных документах без torch."""

import argparse
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from spanver.module import verify

LENGTHS = (1000, 10000, 50000, 2_000_000)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--report", default="artifacts/speed_ranker.json")
    args = parser.parse_args(argv)
    gold = read_jsonl(Path(args.gold))
    rows = []
    for length in LENGTHS:
        document = ""
        index = 0
        while len(document) < length:
            document += (gold[index % len(gold)].get("document_text") or "") + "\n"
            index += 1
        document = document[:length]
        answer = gold[index % len(gold)].get("model_output") or ""
        started = time.perf_counter()
        for _ in range(3 if length < 100000 else 1):
            result = verify(answer, document)
        seconds = (time.perf_counter() - started) / (3 if length < 100000 else 1)
        rows.append(
            {
                "document_chars": length,
                "seconds": round(seconds, 4),
                "ms_per_1000_chars": round(seconds * 1_000_000 / length, 3),
                "spans": len(result["spans"]),
                "answer_chars": len(answer),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {
        "python": platform.python_version(),
        "mode": "ranker, stdlib only",
        "target_ms_per_1000_chars": 30.0,
        "target_2m_seconds": 60.0,
        "rows": rows,
        "meets_target": all(row["ms_per_1000_chars"] <= 30.0 for row in rows),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
