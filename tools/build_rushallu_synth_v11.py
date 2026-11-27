"""V11 блок 1.2: RusHallu-style синтетика clause-уровня с точными границами.

Источник — train-пары v3 без спанов (чистые ответы) и документы train-сплита.
Три программные мутации, ровно одна ошибка на ответ, границы известны по построению:

  number  — число внутри одного clause заменяется другим числом, которого нет в документе;
  entity  — имя/название внутри clause заменяется именем из ДРУГОГО документа той же тематики;
  swap    — clause с низким document support заменяется clause из другого документа.

Ответы остаются длинными (>= MIN_ANSWER символов, >= MIN_CLAUSES кусков), поэтому
синтетика повторяет стиль RusHallu (длинный ответ, локальная ошибка), а не mFAVA.
Выход: artifacts/rushallu_synth_v11.jsonl + манифест (SHA, статистика, doc_overlap=0).
"""
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import v11_lab as L
from spanver.evaluate import read_jsonl

SEED = 11
TARGET = 4000
QUOTA = {"number": 2, "entity": 6, "swap": 20}
MIN_ANSWER = 165
MIN_CLAUSES = 3
OUT = ROOT / "artifacts/rushallu_synth_v11.jsonl"
MANIFEST = ROOT / "artifacts/rushallu_synth_v11_manifest.json"
NUM_TOKEN = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)?(?![\d])")
WORD_TOKEN = re.compile(r"[А-ЯЁA-Z][а-яёa-z\-]{2,}")


def doc_words(document):
    return set(w.lower() for w in L.WORD_RE.findall(document or ""))


def donor_pool(pairs):
    """Числа и имена из документов train, сгруппированные по домену."""
    pool = {}
    for pair in pairs:
        domain = pair.get("domain") or "unknown"
        document = pair.get("document_text") or ""
        bucket = pool.setdefault(domain, {"numbers": set(), "names": set()})
        bucket["numbers"].update(NUM_TOKEN.findall(document))
        bucket["names"].update(WORD_TOKEN.findall(document))
    return {key: {"numbers": sorted(value["numbers"])[:4000],
                  "names": sorted(value["names"])[:8000]}
            for key, value in pool.items()}


def clamp_number(value, rng):
    if re.fullmatch(r"\d{4}", value):
        return str(int(value) + rng.choice([-3, -2, -1, 1, 2, 3, 7, 11]))
    if re.fullmatch(r"\d{1,3}", value):
        return str(max(2, int(value) + rng.choice([-7, -3, -2, -1, 1, 2, 3, 5, 13])))
    return value + "0"


def mutate_number(answer, clause, document, pool, rng):
    start, end, text = clause
    numbers = list(NUM_TOKEN.finditer(text))
    if not numbers:
        return None
    donors = [value for value in (pool.get("numbers") or [])
              if not re.search(rf"(?<![\d.,]){re.escape(value)}(?![\d])", document or "")]
    for _ in range(8):
        target = rng.choice(numbers)
        original = target.group(0)
        replacement = None
        for _ in range(6):
            option = clamp_number(original, rng) if not donors else rng.choice(donors)
            if option != original and not re.search(
                    rf"(?<![\d.,]){re.escape(option)}(?![\d])", document or ""):
                replacement = option
                break
        if replacement is None:
            continue
        local_start = start + target.start()
        local_end = start + target.end() - 1
        new_answer = answer[:local_start] + replacement + answer[local_end + 1:]
        shift = len(replacement) - (local_end - local_start + 1)
        return {
            "answer": new_answer,
            "start": start,
            "end": end + shift,
            "type": "number",
            "replacement": f"{original} -> {replacement}",
        }
    return None


def mutate_entity(answer, clause, document, pool, rng, own_words):
    start, end, text = clause
    names = [match for match in WORD_TOKEN.finditer(text) if match.start() > 0]
    if not names:
        return None
    candidates = pool.get("names") or []
    if not candidates:
        return None
    for _ in range(8):
        target = rng.choice(names)
        original = target.group(0)
        replacement = rng.choice(candidates)
        if replacement == original or replacement.lower() in own_words:
            continue
        if len(replacement) < 3:
            continue
        local_start = start + target.start()
        local_end = start + target.end() - 1
        new_answer = answer[:local_start] + replacement + answer[local_end + 1:]
        shift = len(replacement) - (local_end - local_start + 1)
        return {
            "answer": new_answer,
            "start": start,
            "end": end + shift,
            "type": "entity",
            "replacement": f"{original} -> {replacement}",
        }
    return None


def mutate_swap(answer, clause, document, donors, rng):
    start, end, text = clause
    if len(donors) < 2:
        return None
    for _ in range(10):
        replacement = rng.choice(donors)
        if not replacement or replacement.strip().lower() in (document or "").lower():
            continue
        if abs(len(replacement) - len(text)) > 120:
            continue
        new_answer = answer[:start] + replacement + answer[end + 1:]
        return {
            "answer": new_answer,
            "start": start,
            "end": start + len(replacement) - 1,
            "type": "unsupported_addition",
            "replacement": f"clause_swap(len={len(replacement)})",
        }
    return None


def main():
    rng = random.Random(SEED)
    gold = read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
    train = [p for p in gold if p.get("split") == "train"]
    pool = donor_pool(train)
    donor_answers = {}
    for pair in train:
        donor_answers.setdefault(pair.get("domain") or "unknown", []).append(
            (pair.get("model_output") or "", pair.get("document_text") or ""))

    output = []
    successes = {"number": 0, "entity": 0, "swap": 0}
    failures = {"number": 0, "entity": 0, "swap": 0}
    seen = set()
    order = list(train)
    rng.shuffle(order)
    for pair in order:
        if len(output) >= TARGET:
            break
        answer = pair.get("model_output") or ""
        document = pair.get("document_text") or ""
        domain = pair.get("domain") or "unknown"
        if len(answer) < MIN_ANSWER:
            continue
        clauses = L.clause_spans(answer)
        if len(clauses) < MIN_CLAUSES:
            continue
        gold_spans = pair.get("spans") or []
        own = doc_words(document)
        domain_pool = pool.get(domain) or {}
        quota = dict(QUOTA)
        clause_order = list(clauses)
        rng.shuffle(clause_order)
        donor_cache = None
        for clause in clause_order:
            if len(output) >= TARGET or not any(value > 0 for value in quota.values()):
                break
            if any(clause[0] <= span["end"] and span["start"] <= clause[1] for span in gold_spans):
                continue
            for kind in ("number", "entity", "swap"):
                repeats = 1 if kind == "number" else 3
                for _ in range(repeats):
                  if quota[kind] <= 0 or len(output) >= TARGET:
                    continue
                  mutated = None
                  if kind == "number":
                      mutated = mutate_number(answer, clause, document, domain_pool, rng)
                  elif kind == "entity":
                      mutated = mutate_entity(answer, clause, document, domain_pool, rng, own)
                  else:
                      if donor_cache is None:
                          donor_cache = []
                          own_key = L.doc_key(pair)
                          for donor_answer, donor_document in donor_answers.get(domain, []):
                              if L.doc_key({"document_text": donor_document}) == own_key:
                                  continue
                              for _, _, piece in L.clause_spans(donor_answer):
                                  if len(piece) < 20 or piece.lower() in (document or "").lower():
                                      continue
                                  donor_cache.append(piece)
                              if len(donor_cache) > 400:
                                  break
                      mutated = mutate_swap(answer, clause, document, donor_cache, rng)
                  if not mutated:
                      failures[kind] += 1
                      continue
                  new_answer = mutated["answer"]
                  span_text = new_answer[mutated["start"]:mutated["end"] + 1]
                  key = (pair["pair_id"], mutated["start"], span_text)
                  if len(span_text) < 8 or key in seen:
                      failures[kind] += 1
                      continue
                  seen.add(key)
                  successes[kind] += 1
                  quota[kind] -= 1
                  result = dict(pair)
                  result["pair_id"] = f"synthv11{len(output):06d}"
                  result["model_output"] = new_answer
                  result["spans"] = [{
                      "text": span_text,
                      "start": mutated["start"],
                      "end": mutated["end"],
                      "type": mutated["type"],
                      "type_src": "synthetic",
                      "conversion": "v11_synth_exact_boundary",
                  }]
                  result["split"] = "train"
                  result["label_source"] = "synthetic_v11"
                  result["synthetic"] = {
                      "parent": pair["pair_id"],
                      "mutation": kind,
                      "replacement": mutated["replacement"],
                      "one_error_per_answer": True,
                  }
                  output.append(result)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
                   encoding="utf-8")
    sha_out = hashlib.sha256(OUT.read_bytes()).hexdigest()
    docs_out = {L.doc_key(p) for p in output}
    docs_val = {L.doc_key(p) for p in gold if p.get("split") == "val"}
    docs_held = {L.doc_key(p) for p in read_jsonl(str(ROOT / "artifacts/gold1200_v3.jsonl"))
                 if p.get("split") in ("test", "extra")}
    lengths = [len(p["model_output"]) for p in output]
    span_lengths = [s["end"] - s["start"] + 1 for p in output for s in p["spans"]]
    by_type = {}
    for item in output:
        by_type[item["synthetic"]["mutation"]] = by_type.get(item["synthetic"]["mutation"], 0) + 1
    manifest = {
        "rule": "number/entity mutation и clause swap внутри train-ответа; одна ошибка на ответ; границы точные",
        "seed": SEED, "target": TARGET, "quota_per_parent": QUOTA,
        "sources": {
            "gold_v3.jsonl": hashlib.sha256((ROOT / "artifacts/gold_v3.jsonl").read_bytes()).hexdigest(),
            "gold1200_v3.jsonl": hashlib.sha256((ROOT / "artifacts/gold1200_v3.jsonl").read_bytes()).hexdigest(),
        },
        "outputs": {"rushallu_synth_v11.jsonl": sha_out},
        "counts": {"pairs": len(output), "by_mutation": by_type, "failures": failures,
                   "parents_used": len({p["synthetic"]["parent"] for p in output})},
        "stats": {
            "answer_chars_mean": round(sum(lengths) / len(lengths), 1),
            "answer_chars_p50": statistics.median(lengths),
            "span_chars_mean": round(sum(span_lengths) / len(span_lengths), 1),
            "span_chars_p50": statistics.median(span_lengths),
            "span_share_mean": round(sum(s["end"] - s["start"] + 1 for p in output for s in p["spans"])
                                     / sum(lengths), 4),
            "spans_per_answer": 1.0,
        },
        "doc_overlap": {
            "synth_vs_val": len(docs_out & docs_val),
            "synth_vs_held1200": len(docs_out & docs_held),
        },
        "license": "derived from Dataset/1_rushallu_rag_gold (repo license cc) — только train-документы",
        "old_versions_kept": True,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(json.dumps(manifest["stats"], ensure_ascii=False))
    print(json.dumps(manifest["doc_overlap"], ensure_ascii=False))
    print("sha", sha_out, "manifest_sha", hashlib.sha256(MANIFEST.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
