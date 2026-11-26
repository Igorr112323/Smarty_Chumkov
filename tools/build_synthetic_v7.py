"""Генератор синтетических коротких span-примеров v7 (без ручной разметки).

Источник: документы train-сплита artifacts/gold.jsonl. Из предложений документа
строится чистый ответ, затем программно вносится одна контролируемая мутация:
число/дата/сущность/отрицание/глагол/неподдержанная фраза. Gold span — точные
символьные offsets вставленного/заменённого фрагмента. Негативы — тот же ответ
без мутации. Held-out документы исключаются по SHA-256 текста.

Выход: artifacts/synthetic_short_v7.jsonl + manifest.
"""
import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spanver.evaluate import load_split, read_jsonl

SENT = re.compile(r"[^.!?\n]+[.!?]*")
TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-’'][0-9A-Za-zА-Яа-яЁё]+)*|[^\s]", re.U)
YEAR = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")
NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
CAP = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b")
STARTERS = {"как", "в", "это", "при", "по", "на", "так", "но", "и", "о", "с", "для", "если", "такой", "также", "однако", "например", "впрочем", "зато", "даже", "притом"}
VERB_MAP = {
    "увеличилась": "снизилась", "увеличился": "снизился", "вырос": "упал",
    "выросла": "упала", "возрос": "упал", "поддержал": "отверг",
    "поддержала": "отвергла", "подтвердил": "опроверг", "подтвердила": "опровергла",
    "помог": "помешал", "помогла": "помешала", "признал": "отверг",
    "соответствует": "противоречит", "открыл": "закрыл", "начал": "прекратил",
    "началась": "прекратилась", "имеет": "не имеет", "является": "не является",
}
INJECT = [
    "при этом проект финансировался частным фондом",
    "и также активно обсуждался в международной прессе",
    "что было официально подтверждено правительством",
    "при этом все расходы были полностью покрыты спонсорами",
    "и впоследствии компания получила государственную награду",
    "по данным независимых экспертов это былоargest достижением",
]
NEGATIONS = ("не", "никогда", "нигде")


def sentences(text):
    return [m.group(0).strip() for m in SENT.finditer(text or "") if len(m.group(0).strip()) > 30]


def shade_answer(sents, rng):
    if len(sents) < 2:
        return None
    start = rng.randrange(0, max(1, len(sents) - 2))
    count = rng.randint(2, min(5, len(sents) - start))
    answer = " ".join(sents[start:start + count])
    if not (80 <= len(answer) <= 1200):
        return None
    return answer


def replace_span(answer, start, end, replacement):
    return answer[:start] + replacement + answer[end + 1:], (start, start + len(replacement) - 1)


def mutate(answer, doc, rng):
    candidates = []
    for match in YEAR.finditer(answer):
        candidates.append(("date", match.start(), match.end() - 1))
    for match in NUM.finditer(answer):
        if not YEAR.fullmatch(answer[match.start():match.end()]):
            candidates.append(("number", match.start(), match.end() - 1))
    for match in TOKEN.finditer(answer):
        low = match.group(0).lower()
        if low in VERB_MAP:
            candidates.append(("verb", match.start(), match.end() - 1))
    for match in CAP.finditer(answer):
        if match.start() > 0 and match.group(0).lower() not in STARTERS:
            candidates.append(("entity", match.start(), match.end() - 1))
    for match in re.finditer(r"\b" + "|".join(VERB_MAP) + r"\b", answer, re.I):
        candidates.append(("negation", match.start(), match.end() - 1))
    if not candidates:
        return None
    kind, start, end = rng.choice(candidates)
    original = answer[start:end + 1]
    if kind == "date":
        year = int(original)
        replacement = str(year + rng.choice([-7, -3, 2, 5, 11]))
    elif kind == "number":
        try:
            value = float(original.replace(",", "."))
            replacement = str(round(value * rng.choice([0.5, 1.5, 2.0, 3.0]), 2)).rstrip("0").rstrip(".")
        except ValueError:
            return None
    elif kind == "verb":
        replacement = VERB_MAP[original.lower()]
    elif kind == "entity":
        entities = list(dict.fromkeys(CAP.findall(doc or "")))
        pool = [e for e in entities if e.lower() not in STARTERS and e.lower() != original.lower()]
        if not pool:
            return None
        replacement = rng.choice(pool[:20])
    elif kind == "negation":
        replacement = "не " + original
    else:
        return None
    if replacement == original:
        return None
    mutated, span = replace_span(answer, start, end, replacement)
    if len(mutated) > 1300:
        return None
    return mutated, span, kind, original, replacement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/synthetic_short_v7.jsonl")
    parser.add_argument("--manifest", default="artifacts/synthetic_short_v7_manifest.json")
    parser.add_argument("--per-doc", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    train = load_split(str(ROOT / "artifacts/gold.jsonl"), "train")
    held = [p for p in read_jsonl(str(ROOT / "artifacts/gold1200_v2.jsonl"))
            if p.get("split") in ("test", "extra")]
    held_hashes = {hashlib.sha256((p.get("document_text") or "").encode()).hexdigest() for p in held}
    used_doc_hashes = set()
    rows = []
    stats = collections.Counter()
    for pair in train:
        doc = pair.get("document_text") or ""
        doc_hash = hashlib.sha256(doc.encode()).hexdigest()
        if doc_hash in held_hashes or doc_hash in used_doc_hashes or not doc:
            continue
        used_doc_hashes.add(doc_hash)
        sents = sentences(doc)
        made = 0
        for _ in range(args.per_doc * 4):
            if made >= args.per_doc:
                break
            answer = shade_answer(sents, rng)
            if not answer:
                continue
            result = mutate(answer, doc, rng)
            if not result:
                continue
            mutated, span, kind, original, replacement = result
            pid = f"synth{doc_hash[:10]}_{made:02d}"
            rows.append({
                "pair_id": pid,
                "split": "train_synthetic",
                "source_pair_id": pair["pair_id"],
                "document_text": doc,
                "model_output": mutated,
                "spans": [{"start": span[0], "end": span[1],
                           "type": "unsupported_addition", "risk": 1.0, "confidence": 1.0}],
                "mutation": {"type": kind, "original": original, "replacement": replacement},
                "doc_sha256": doc_hash,
            })
            rows.append({
                "pair_id": pid + "_n",
                "split": "train_synthetic",
                "source_pair_id": pair["pair_id"],
                "document_text": doc,
                "model_output": answer,
                "spans": [],
                "mutation": {"type": "negative", "original": "", "replacement": ""},
                "doc_sha256": doc_hash,
            })
            stats[kind] += 1
            stats["negative"] += 1
            made += 1
    bad = []
    for row in rows:
        answer = row["model_output"]
        for gold in row["spans"]:
            if not (0 <= gold["start"] <= gold["end"] < len(answer)):
                bad.append(row["pair_id"])
            elif answer[gold["start"]:gold["end"] + 1] != row["mutation"]["replacement"]:
                bad.append(row["pair_id"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": "artifacts/gold.jsonl train documents (RusHallu-RAG)",
        "license": "исследовательское использование; исходные документы из локального набора репозитория",
        "generator": "tools/build_synthetic_v7.py",
        "seed": args.seed,
        "pairs": len(rows),
        "positive_pairs": sum(1 for r in rows if r["spans"]),
        "negative_pairs": sum(1 for r in rows if not r["spans"]),
        "unique_docs": len(used_doc_hashes),
        "doc_overlap_with_held": len(used_doc_hashes & held_hashes),
        "mutation_counts": dict(stats),
        "bad_spans": len(bad),
        "synthetic_jsonl_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "held_dataset_sha256": hashlib.sha256((ROOT / "artifacts/gold1200_v2.jsonl").read_bytes()).hexdigest(),
    }
    (ROOT / args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
