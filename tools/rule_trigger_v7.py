"""Простой rule-based trigger decoder v7 (без обучения).

Для каждого токена-триггера (число, дата, сущность, отрицание, глагол
противоречия), отсутствующего в документе, предсказываем короткий span
вокруг триггера. Pair gate заморожен. Проверяется на val/mFAVA-train.
"""
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import document_context, stem
from train_local_synth_v7 import trigger_positions

TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-’'][0-9A-Za-zА-Яа-яЁё]+)*|[^\s]")


def rule_spans(answer, doc, dtok, dstems, use_neg, max_len):
    out = []
    for s, e in trigger_positions(answer, doc, dtok, dstems, 40):
        token = answer[s:e + 1]
        low = token.lower()
        supported = low in dtok or stem(low) in dstems
        if supported and not use_neg:
            continue
        if not use_neg and not (token[0].isdigit() or token[:1].isupper()):
            continue
        left = max(0, s - (max_len // 2))
        right = min(len(answer) - 1, e + (max_len // 2))
        while left <= s and left < len(answer) and answer[left].isspace():
            left += 1
        while right >= e and answer[right].isspace():
            right -= 1
        out.append({"start": left, "end": right, "risk": 1.0,
                    "type": "unsupported_addition", "confidence": 1.0})
    return out


def main():
    val = load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    ex = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    flags = pickle.load(open(ROOT / "artifacts/baseline_flags_v5.pkl", "rb"))
    sweep = {}
    for use_neg in (False, True):
        for max_len in (1, 3, 5, 9):
            def predict(pair):
                if not flags.get(pair["pair_id"]):
                    return []
                answer = pair.get("model_output") or ""
                doc = pair.get("document_text") or ""
                dtok, dstems, _ = document_context(doc)
                return rule_spans(answer, doc, dtok, dstems, use_neg, max_len)
            mv = metrics(val, predict)
            me = metrics(ex, predict)
            sweep[f"neg{use_neg}_len{max_len}"] = {"val": mv, "mfava": me}
            print(f"neg{use_neg}_len{max_len}", mv["span_level"]["f1"], me["span_level"]["f1"], mv["fpr"])
    target = ROOT / "artifacts/rule_trigger_v7.json"
    target.write_text(json.dumps(sweep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
