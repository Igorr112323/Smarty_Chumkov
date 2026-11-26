"""Аудит стиля разметки span-ов по train/val/held/mFAVA.

Только read-only. Результат: artifacts/annotation_style_audit_v8.json.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import load_split, read_jsonl

SENT = re.compile(r"[^.!?\n]+[.!?]*")
CLAUSE = re.compile(r"[^,;:()—–\n]+[,;:()—–]*")
TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-’'][0-9A-Za-zА-Яа-яЁё]+)*|[^\s]", re.U)


def bucket(length):
    if length < 20:
        return "<20"
    if length < 40:
        return "20-40"
    if length < 80:
        return "40-80"
    if length < 200:
        return "80-200"
    return "200+"


def style(answer, gold):
    text = answer or ""
    start, end = gold["start"], gold["end"]
    length = end - start + 1
    fragment = text[start:end + 1]
    sentences = [m.group(0) for m in SENT.finditer(text) if m.group(0).strip()]
    clauses = [m.group(0) for m in CLAUSE.finditer(text) if m.group(0).strip()]
    sent_cov = max((len(set(range(start, end + 1)) & set(range(m.start(), m.end()))) / max(1, len(m.group(0)))
                    for m in SENT.finditer(text)), default=0.0)
    clause_cov = max((len(set(range(start, end + 1)) & set(range(m.start(), m.end()))) / max(1, len(m.group(0)))
                      for m in CLAUSE.finditer(text)), default=0.0)
    token_starts = {m.start() for m in TOKEN.finditer(text)}
    token_ends = {m.end() - 1 for m in TOKEN.finditer(text)}
    return {
        "length": length,
        "bucket": bucket(length),
        "tokens": len(TOKEN.findall(fragment)),
        "whole_answer": start <= 2 and end >= len(text) - 3,
        "sentence_like": sent_cov >= 0.8,
        "clause_like": clause_cov >= 0.8,
        "short_word": length < 40 and len(TOKEN.findall(fragment)) <= 4,
        "punct_start": bool(fragment[:1] in ",;:()—–.!? "),
        "punct_end": bool(fragment[-1:] in ",;:()—–.!? "),
        "token_boundary": start in token_starts and end in token_ends,
        "type": gold.get("type") or "?",
        "fragment": fragment[:160],
    }


def analyse(pairs, examples=5):
    spans = []
    for pair in pairs:
        for gold in pair.get("spans") or []:
            spans.append(style(pair.get("model_output") or "", gold))
    result = {
        "pairs": len(pairs),
        "pairs_with_spans": sum(1 for p in pairs if p.get("spans")),
        "spans": len(spans),
        "spans_per_positive_pair": round(len(spans) / max(1, sum(1 for p in pairs if p.get("spans"))), 3),
        "multi_span_pairs": sum(1 for p in pairs if len(p.get("spans") or []) > 1),
        "length_buckets": dict(collections.Counter(s["bucket"] for s in spans)),
        "types": dict(collections.Counter(s["type"] for s in spans)),
        "whole_answer": sum(s["whole_answer"] for s in spans),
        "sentence_like": sum(s["sentence_like"] for s in spans),
        "clause_like": sum(s["clause_like"] for s in spans),
        "short_word": sum(s["short_word"] for s in spans),
        "punct_start": sum(s["punct_start"] for s in spans),
        "punct_end": sum(s["punct_end"] for s in spans),
        "token_boundary": sum(s["token_boundary"] for s in spans),
        "examples_whole": [s["fragment"] for s in spans if s["whole_answer"]][:examples],
        "examples_short": [s["fragment"] for s in spans if s["short_word"]][:examples],
    }
    return result


def main():
    gold = str(ROOT / "artifacts/gold.jsonl")
    v2 = str(ROOT / "artifacts/gold1200_v2.jsonl")
    mfava = str(ROOT / "artifacts/mfava_train_v2.jsonl")
    v2rows = read_jsonl(v2)
    slices = {
        "train": load_split(gold, "train"),
        "val": load_split(gold, "val"),
        "mfava_train": read_jsonl(mfava),
        "v2_test301": [p for p in v2rows if p.get("split") == "test"],
        "v2_extra50": [p for p in v2rows if p.get("split") == "extra"],
        "v2_held351": [p for p in v2rows if p.get("split") in ("test", "extra")],
    }
    report = {
        "rule": "read-only audit; boundary style may differ between sources",
        "slices": {name: analyse(pairs) for name, pairs in slices.items()},
        "manifest": {
            "gold_jsonl_sha256": hashlib.sha256(Path(gold).read_bytes()).hexdigest(),
            "gold1200_v2_sha256": hashlib.sha256(Path(v2).read_bytes()).hexdigest(),
            "mfava_train_v2_sha256": hashlib.sha256(Path(mfava).read_bytes()).hexdigest(),
        },
    }
    target = ROOT / "artifacts/annotation_style_audit_v8.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, value in report["slices"].items():
        print(name, "pairs", value["pairs"], "spans", value["spans"],
              "whole", value["whole_answer"], "short", value["short_word"],
              "clause", value["clause_like"], "types", value["types"], flush=True)
    print("report", target, target.stat().st_size)


if __name__ == "__main__":
    main()
