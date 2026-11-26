"""Воспроизводимый oracle-candidate recall v4 (только train/val/mFAVA-train).

Гейт: кандидат считается покрывающим gold, если IoU >= 0.5, как в eval.
Held-out/test не читаются. Результат сохраняется в artifacts/oracle_recall_v4.json.
"""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from spanver.evaluate import iou, load_split, read_jsonl
from spanver.module import candidate_spans

TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-’'][0-9A-Za-zА-Яа-яЁё]+)*|[^\s]", re.U)
PUNCT_RE = re.compile(r"[^,;:()\[\]«»\"\n—–-]+")
MAX_TOKENS = 6


def bucket(length):
    if length < 20:
        return "<20"
    if length < 40:
        return "20-40"
    if length < 80:
        return "40-80"
    return "80+"


def candidates_v4(answer):
    """Базовые кандидаты + токенные n-граммы + фрагменты по пунктуации."""
    text = answer or ""
    base = list(candidate_spans(text))
    seen = {(start, end) for start, end, _ in base}
    out = list(base)
    tokens = list(TOKEN_RE.finditer(text))
    for index, match in enumerate(tokens):
        for size in range(1, MAX_TOKENS + 1):
            if index + size > len(tokens):
                break
            start = match.start()
            end = tokens[index + size - 1].end() - 1
            fragment = text[start:end + 1].strip()
            if not (3 <= len(fragment) <= 160):
                continue
            if (start, end) not in seen:
                seen.add((start, end))
                out.append((start, end, fragment))
    for match in PUNCT_RE.finditer(text):
        fragment = match.group(0).strip()
        if not (3 <= len(fragment) <= 160):
            continue
        start = match.start() + match.group(0).find(fragment)
        end = start + len(fragment) - 1
        if (start, end) not in seen:
            seen.add((start, end))
            out.append((start, end, fragment))
    return sorted(out)


def covered(candidates, gold):
    start, end = gold["start"], gold["end"]
    return any(iou(c[0], c[1], start, end) >= 0.5 for c in candidates)


def analyse(pairs):
    by_bucket = collections.defaultdict(lambda: {"base": 0, "new": 0, "n": 0})
    by_type = collections.defaultdict(lambda: {"base": 0, "new": 0, "n": 0})
    total = {"base": 0, "new": 0, "n": 0}
    candidate_sum = 0
    for pair in pairs:
        answer = pair.get("model_output") or ""
        base = candidate_spans(answer)
        new = candidates_v4(answer)
        candidate_sum += len(new)
        for gold in pair.get("spans") or []:
            length = gold["end"] - gold["start"] + 1
            hit_base = covered(base, gold)
            hit_new = covered(new, gold)
            for table, key in ((by_bucket, bucket(length)), (by_type, gold.get("type") or "?")):
                table[key]["base"] += int(hit_base)
                table[key]["new"] += int(hit_new)
                table[key]["n"] += 1
            total["base"] += int(hit_base)
            total["new"] += int(hit_new)
            total["n"] += 1
    def render(table):
        out = {}
        for key, values in table.items():
            count = values["n"]
            out[key] = {
                "gold": count,
                "baseline_ok": values["base"],
                "baseline_recall": round(values["base"] / count, 4) if count else 0.0,
                "v4_ok": values["new"],
                "v4_recall": round(values["new"] / count, 4) if count else 0.0,
            }
        return out
    return {
        "pairs": len(pairs),
        "gold_spans": total["n"],
        "avg_candidates_v4": round(candidate_sum / max(1, len(pairs)), 1),
        "baseline_recall": round(total["base"] / total["n"], 4) if total["n"] else 0.0,
        "v4_recall": round(total["new"] / total["n"], 4) if total["n"] else 0.0,
        "by_length": render(by_bucket),
        "by_type": render(by_type),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/oracle_recall_v4.json")
    args = parser.parse_args()
    splits = {
        "train": load_split(str(ROOT / "artifacts/gold.jsonl"), "train"),
        "val": load_split(str(ROOT / "artifacts/gold.jsonl"), "val"),
        "mfava_train": read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl")),
    }
    datasets = {
        "rus_hallu_train_val": splits["train"] + splits["val"],
        "mfava_train": splits["mfava_train"],
    }
    report = {
        "rule": "IoU >= 0.5; только train/val/mFAVA-train, held-out не используется",
        "splits": {name: analyse(pairs) for name, pairs in splits.items()},
        "datasets": {name: analyse(pairs) for name, pairs in datasets.items()},
        "manifest": {
            "gold_jsonl_sha256": hashlib.sha256((ROOT / "artifacts/gold.jsonl").read_bytes()).hexdigest(),
            "mfava_train_v2_sha256": hashlib.sha256((ROOT / "artifacts/mfava_train_v2.jsonl").read_bytes()).hexdigest(),
        },
    }
    short_keys = ("<20", "20-40")
    for name in ("train", "val", "mfava_train"):
        section = report["splits"][name]["by_length"]
        before = sum(section[k]["baseline_ok"] for k in short_keys)
        after = sum(section[k]["v4_ok"] for k in short_keys)
        total = sum(section[k]["gold"] for k in short_keys)
        report["splits"][name]["short_under40_gate"] = {
            "gold": total, "baseline": before, "v4": after,
            "baseline_recall": round(before / total, 4) if total else 0.0,
            "v4_recall": round(after / total, 4) if total else 0.0,
        }
    target = ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
