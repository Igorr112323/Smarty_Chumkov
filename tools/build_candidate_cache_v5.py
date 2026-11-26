"""Воспроизводимый кэш признаков кандидатов v4 для скриптов v5.

Сохраняет список (pair_id, start, end, fragment, features, matched) в
artifacts/candidate_cache_v5.pkl.Held-out не читается.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from oracle_recall_v4 import candidates_v4
from spanver.evaluate import iou, load_split, read_jsonl
from spanver.module import document_context
from spanver import ranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/candidate_cache_v5.pkl")
    args = parser.parse_args()
    pairs = (
        load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
        + load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
        + read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    )
    rows = []
    for pair in pairs:
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        dtok, dstems, dgrams = document_context(document)
        for start, end, fragment in candidates_v4(answer):
            x = ranker.features(fragment, start, end, answer, document, dtok, dgrams)
            matched = int(any(
                iou(start, end, g["start"], g["end"]) >= 0.5
                for g in (pair.get("spans") or [])
            ))
            rows.append((pair["pair_id"], start, end, fragment, x, matched))
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(target, "wb") as handle:
        pickle.dump(rows, handle, protocol=4)
    print("saved", target, len(rows), target.stat().st_size)


if __name__ == "__main__":
    main()
