import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import decode
from spanver.evaluate import read_jsonl
from spanver.snap import snapped
from spanver.splits import load_splits

TOKEN_THRESHOLD = 0.25
PAIR_GATE = 0.45
MIN_CHARS = 2


def load_oof(folder):
    dumps = {}
    for path in sorted(Path(folder).glob("oof_*.json")):
        dumps.update(json.loads(path.read_text(encoding="utf-8")))
    return dumps


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="artifacts/cv/base")
    parser.add_argument("--gold", default="artifacts/gold.jsonl")
    parser.add_argument("--out", default="artifacts/hard_negatives.jsonl")
    parser.add_argument("--report", default="artifacts/hard_report.json")
    args = parser.parse_args(argv)

    splits = load_splits()
    gold = read_jsonl(Path(args.gold))
    by_id = {pair.get("pair_id"): pair for pair in gold}
    ours = [by_id[pair_id] for name in ("train", "val") for pair_id in splits[name] if pair_id in by_id]
    dumps = load_oof(args.oof)
    rows = []
    positives = 0
    total_negatives = 0
    for pair in ours:
        if pair.get("spans"):
            positives += 1
            continue
        if not pair.get("model_output"):
            continue
        total_negatives += 1
        dumped = dumps.get(pair.get("pair_id"))
        if not dumped:
            continue
        top = max((value for _, _, value in dumped["pieces"]), default=0.0)
        if top < PAIR_GATE:
            continue
        flags = decode.flags_for(dumped["pieces"], TOKEN_THRESHOLD, None)
        spans = decode.spans_from_flags(dumped["answer"], dumped["pieces"], flags, MIN_CHARS)
        if not spans:
            continue
        rows.append(
            {
                "pair_id": f"hard-{pair.get('pair_id')}",
                "model_output": pair.get("model_output"),
                "document_text": pair.get("document_text"),
                "doc_id": (pair.get("doc_ids") or [pair.get("pair_id")])[0],
                "spans": [],
                "label": 0,
                "source": "hard_negative",
                "origin_pair": pair.get("pair_id"),
            }
        )
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "oof": args.oof,
        "pairs": len(ours),
        "positives": positives,
        "negatives": total_negatives,
        "hard_negatives": len(rows),
        "share_of_negatives": round(len(rows) / total_negatives, 4) if total_negatives else 0.0,
        "decode": {"token_threshold": TOKEN_THRESHOLD, "pair_gate": PAIR_GATE, "min_chars": MIN_CHARS},
        "file": str(target),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
