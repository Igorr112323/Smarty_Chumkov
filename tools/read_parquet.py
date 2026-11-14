import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spanver.io_parquet import LfsPointerError, ensure_readable
from spanver.pairs import doc_ids, ordered

TRAIN_DIR = "2_rushallu_rag_pairs"
TRAIN_FILE = "sberquad-train-rag.parquet"
TRAIN_LFS = "ruscibench-train-rag.parquet"


def make_pair(row, number):
    document_text = row.get("joined_reference") or ""
    return ordered(
        {
            "pair_id": f"sbert{number:06d}",
            "source": f"{TRAIN_DIR}/{TRAIN_FILE}",
            "domain": "sberquad",
            "query_id": str(row.get("query_id")),
            "query_text": row.get("query_text") or "",
            "document_text": document_text,
            "doc_ids": doc_ids(document_text),
            "model_output": row.get("model_output") or "",
            "spans": [],
            "clean": None,
            "label_source": "unlabeled",
            "perturb": {
                "rel_position_mod": int(row.get("rel_position_mod")),
                "score_mod": float(row.get("score_mod")),
            },
            "split": None,
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Dataset")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lfs_path = dataset / TRAIN_DIR / TRAIN_LFS
    try:
        ensure_readable(lfs_path)
        print(f"{TRAIN_LFS} -> readable")
    except LfsPointerError as exc:
        print(f"{TRAIN_LFS} -> LFS pointer, data not fetched")
        print(f"reason: {exc}")
    path = dataset / TRAIN_DIR / TRAIN_FILE
    size = ensure_readable(path)
    table = pq.read_table(path)
    rows = table.to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    target = out_dir / "pairs_unlabeled.jsonl"
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        for number, row in enumerate(rows):
            handle.write(json.dumps(make_pair(row, number), ensure_ascii=False) + "\n")
    print(f"records = {len(rows)}")
    print(f"written {target} {len(rows)} lines")
    print(f"file bytes = {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
