"""Сборка договорного набора v3 с гармонизацией границ C1.

Источники: artifacts/gold.jsonl (train/val), artifacts/mfava_train_v2.jsonl,
artifacts/gold1200_v2.jsonl (1200 пар, test/extra).>Co rule C1 из convert_labels_v3.
Старые версии не удаляются. Проверяется doc_overlap с held-out = 0.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from convert_labels_v3 import convert_all
from spanver.evaluate import load_split, read_jsonl


def write(path, rows):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(target.read_bytes()).hexdigest(), target.stat().st_size


def main():
    base = load_split(str(ROOT / "artifacts/gold.jsonl"), "train") + load_split(str(ROOT / "artifacts/gold.jsonl"), "val")
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v2.jsonl"))
    v2 = read_jsonl(str(ROOT / "artifacts/gold1200_v2.jsonl"))
    held = [p for p in v2 if p.get("split") in ("test", "extra")]
    held_docs = {hashlib.sha256((p.get("document_text") or "").encode()).hexdigest() for p in held}
    excluded = [p["pair_id"] for p in mfava
                if hashlib.sha256((p.get("document_text") or "").encode()).hexdigest() in held_docs]
    mfava = [p for p in mfava if p["pair_id"] not in set(excluded)]
    shas = {}
    shas["gold_v3.jsonl"] = write("artifacts/gold_v3.jsonl", convert_all(base))
    shas["mfava_train_v3.jsonl"] = write("artifacts/mfava_train_v3.jsonl", convert_all(mfava))
    shas["gold1200_v3.jsonl"] = write("artifacts/gold1200_v3.jsonl", convert_all(v2))
    train_v3 = read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
    mfava_v3 = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    v3 = read_jsonl(str(ROOT / "artifacts/gold1200_v3.jsonl"))
    docs_train = {hashlib.sha256((p.get("document_text") or "").encode()).hexdigest()
                  for p in train_v3 + mfava_v3}
    docs_held_v3 = {hashlib.sha256((p.get("document_text") or "").encode()).hexdigest()
                    for p in v3 if p.get("split") in ("test", "extra")}
    manifest = {
        "rule": "C1_min_containing_clause from tools/convert_labels_v3.py",
        "sources": {
            "gold.jsonl_sha256": hashlib.sha256((ROOT / "artifacts/gold.jsonl").read_bytes()).hexdigest(),
            "gold1200_v2_sha256": hashlib.sha256((ROOT / "artifacts/gold1200_v2.jsonl").read_bytes()).hexdigest(),
            "mfava_train_v2_sha256": hashlib.sha256((ROOT / "artifacts/mfava_train_v2.jsonl").read_bytes()).hexdigest(),
        },
        "outputs": shas,
        "counts": {
            "gold_v3_pairs": len(train_v3),
            "mfava_train_v3_pairs": len(mfava_v3),
            "gold1200_v3_pairs": len(v3),
            "v3_held_pairs": sum(1 for p in v3 if p.get("split") in ("test", "extra")),
            "v3_train_spans": sum(len(p.get("spans") or []) for p in train_v3),
            "v3_mfava_spans": sum(len(p.get("spans") or []) for p in mfava_v3),
            "v3_held_spans": sum(len(p.get("spans") or []) for p in v3 if p.get("split") in ("test", "extra")),
        },
        "excluded_train_pairs_due_to_doc_overlap": excluded,
        "doc_overlap_train_vs_held": len(docs_train & docs_held_v3),
        "doc_overlap_held_v2_vs_v3": len(held_docs & docs_held_v3),
        "old_versions_kept": ["artifacts/gold.jsonl", "artifacts/gold1200_v2.jsonl", "artifacts/mfava_train_v2.jsonl"],
    }
    (ROOT / "artifacts/gold1200_v3_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
