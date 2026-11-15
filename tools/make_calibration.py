import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
SENTENCES_PER_MIXTURE = 40
MIN_SENTENCE = 40
MAX_SENTENCE = 220
SEED = 0
DOCUMENT_CHARS = 12000


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sentences(text):
    found = []
    for match in SENTENCE_RE.finditer(text or ""):
        piece = match.group(0).strip()
        if MIN_SENTENCE <= len(piece) <= MAX_SENTENCE:
            found.append(piece)
    return found


def build_document(pairs):
    parts = []
    total = 0
    for pair in pairs:
        document = pair.get("document_text") or ""
        if not document:
            continue
        parts.append(document)
        total += len(document) + 1
        if total >= DOCUMENT_CHARS:
            break
    return " ".join(parts)[:DOCUMENT_CHARS]


def pick(pool, count, rng):
    if count >= len(pool):
        return list(pool)
    indexes = list(range(len(pool)))
    rng.shuffle(indexes)
    return [pool[index] for index in sorted(indexes[:count])]


def interleave(human, ai):
    total = len(human) + len(ai)
    share = len(ai) / total if total else 0.0
    result = []
    human_index = 0
    ai_index = 0
    for position in range(total):
        target_before = round(position * share)
        target_after = round((position + 1) * share)
        if target_after > target_before and ai_index < len(ai):
            result.append(("ai", ai[ai_index]))
            ai_index += 1
        elif human_index < len(human):
            result.append(("human", human[human_index]))
            human_index += 1
        elif ai_index < len(ai):
            result.append(("ai", ai[ai_index]))
            ai_index += 1
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/gold.jsonl")
    parser.add_argument("--out", default="artifacts/calib_mixture.jsonl")
    args = parser.parse_args(argv)
    pairs = read_jsonl(Path(args.data))
    document = build_document(pairs)
    human_pool = sentences(document)
    ai_pool = []
    for pair in pairs:
        ai_pool.extend(sentences(pair.get("model_output") or ""))
    rows = []
    for share in SHARES:
        rng = random.Random(SEED)
        human_count = int(round(SENTENCES_PER_MIXTURE * (1.0 - share)))
        ai_count = SENTENCES_PER_MIXTURE - human_count
        human = pick(human_pool, human_count, rng)
        ai = pick(ai_pool, ai_count, rng)
        mixed = interleave(human, ai)
        text = " ".join(piece for _, piece in mixed)
        rows.append(
            {
                "share": share,
                "text": text,
                "document": document,
                "n_human": human_count,
                "n_ai": ai_count,
                "chars": len(text),
            }
        )
    target = Path(args.out)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    for row in rows:
        print(f"share {row['share']:.2f}: human {row['n_human']}, ai {row['n_ai']}, chars {row['chars']}")
    print(f"written {target} {len(rows)} lines")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
