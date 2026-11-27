"""V11 блок 1.1: pseudo-clause метки из whole-спанов RusHallu train.

Правило (программное, без ручной разметки и без held-out):
  берём train-пары, у которых gold = whole-answer; режем ответ на синтаксические
  куски по пунктуации/союзам (tools/v11_lab.clause_spans); для каждого куска
  считаем document support (token coverage, TF-IDF cosine к лучшему предложению,
  отсутствие чисел/имён); pseudo-clause = кусок с минимальным support.
  Если минимум не отделён от медианы (>= 0.75 медианы), пара пропускается как
  шумная — сигнала «локальная ошибка» в ней нет.

Выход: artifacts/clause_pseudo_v11.jsonl + манифест с SHA и doc_overlap=0.
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import v11_lab as L
from spanver.evaluate import read_jsonl

MIN_CLAUSES = 2
SEPARATION = 0.75
OUT = ROOT / "artifacts/clause_pseudo_v11.jsonl"
MANIFEST = ROOT / "artifacts/clause_pseudo_v11_manifest.json"


def support_score(features):
    return 0.6 * features["token_cov"] + 0.4 * features["cos"]


def build_pair(pair):
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    whole = [s for s in (pair.get("spans") or []) if s["start"] == 0 and s["end"] == len(answer) - 1]
    if not whole:
        return None
    clauses = L.clause_spans(answer)
    if len(clauses) < MIN_CLAUSES:
        return None
    scored = []
    for start, end, text in clauses:
        features = L.support_features(text, document)
        scored.append({"start": start, "end": end, "text": text,
                       "support": round(support_score(features), 4),
                       "token_cov": round(features["token_cov"], 4),
                       "cos": round(features["cos"], 4)})
    median = statistics.median(item["support"] for item in scored)
    worst = min(scored, key=lambda item: (item["support"], item["start"]))
    if median <= 0 or worst["support"] >= SEPARATION * median:
        return {"skipped": True, "reason": "no_separation", "median": round(median, 4),
                "worst": worst["support"], "pair_id": pair["pair_id"]}
    result = dict(pair)
    result["pair_id"] = f"pseudo{pair['pair_id']}"
    result["spans"] = [{
        "text": worst["text"],
        "start": worst["start"],
        "end": worst["end"],
        "type": "unsupported_addition",
        "type_src": whole[0].get("type_src"),
        "conversion": "v11_pseudo_clause_lowest_support",
        "support": worst["support"],
    }]
    result["split"] = "train"
    result["label_source"] = "pseudo_clause_v11"
    result["pseudo"] = {
        "parent": pair["pair_id"],
        "rule": "lowest_document_support_clause",
        "median_support": round(median, 4),
        "clauses": len(scored),
    }
    return {"pair": result, "median": median, "worst": worst["support"]}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    train = [p for p in read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl")) if p.get("split") == "train"]
    held = [p for p in read_jsonl(str(ROOT / "artifacts/gold1200_v3.jsonl"))
            if p.get("split") in ("test", "extra")]
    output, skipped = [], []
    for pair in train:
        built = build_pair(pair)
        if built is None:
            skipped.append({"pair_id": pair["pair_id"], "reason": "no_whole_or_too_short"})
            continue
        if built.get("skipped"):
            skipped.append(built)
            continue
        output.append(built["pair"])
    sha = write_jsonl(OUT, output)
    docs_out = {L.doc_key(p) for p in output}
    docs_val = {L.doc_key(p) for p in read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
                if p.get("split") == "val"}
    docs_held = {L.doc_key(p) for p in held}
    docs_mfava = {L.doc_key(p) for p in read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))}
    lengths = [len(p["model_output"]) for p in output]
    spans = [s["end"] - s["start"] + 1 for p in output for s in p["spans"]]
    manifest = {
        "rule": "whole-answer train пары -> минимальный по document support clause (v11_lab.support_features)",
        "separation_filter": f"min_support < {SEPARATION} * median_support",
        "sources": {
            "gold_v3.jsonl": hashlib.sha256((ROOT / "artifacts/gold_v3.jsonl").read_bytes()).hexdigest(),
            "gold1200_v3.jsonl": hashlib.sha256((ROOT / "artifacts/gold1200_v3.jsonl").read_bytes()).hexdigest(),
        },
        "outputs": {"clause_pseudo_v11.jsonl": sha},
        "counts": {
            "train_pairs_scanned": len(train),
            "pseudo_pairs": len(output),
            "skipped": len(skipped),
            "skip_reasons": {
                "no_separation": sum(1 for s in skipped if s.get("reason") == "no_separation"),
                "no_whole_or_too_short": sum(1 for s in skipped if s.get("reason") == "no_whole_or_too_short"),
            },
        },
        "stats": {
            "answer_chars_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "answer_chars_p50": statistics.median(lengths) if lengths else 0,
            "span_chars_mean": round(sum(spans) / len(spans), 1) if spans else 0,
            "span_chars_p50": statistics.median(spans) if spans else 0,
            "span_share_of_answer_mean": round(
                sum((s["end"] - s["start"] + 1) / len(p["model_output"]) for p in output for s in p["spans"])
                / max(1, len(spans)), 4),
        },
        "doc_overlap": {
            "pseudo_vs_val": len(docs_out & docs_val),
            "pseudo_vs_held1200": len(docs_out & docs_held),
            "pseudo_vs_mfava_train": len(docs_out & docs_mfava),
        },
        "old_versions_kept": True,
        "skipped_examples": skipped[:20],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(json.dumps(manifest["stats"], ensure_ascii=False))
    print(json.dumps(manifest["doc_overlap"], ensure_ascii=False))
    print("sha", sha, "manifest_sha",
          hashlib.sha256(MANIFEST.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
