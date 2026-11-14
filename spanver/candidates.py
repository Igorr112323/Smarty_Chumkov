from .detector import detect

PER_DOC_CAP = 2
DEFAULT_LIMIT = 200


def gold_documents(gold_pairs):
    documents = set()
    for pair in gold_pairs:
        for doc in pair.get("doc_ids") or []:
            documents.add(doc)
    return documents


def build_candidates(gold_pairs, unlabeled_pairs, limit=DEFAULT_LIMIT, cap=PER_DOC_CAP):
    reserved = gold_documents(gold_pairs)
    pool = [
        pair
        for pair in unlabeled_pairs
        if not (set(pair.get("doc_ids") or []) & reserved)
    ]
    scored: list = []
    used_per_doc: dict = {}
    for pair in sorted(pool, key=lambda item: item["pair_id"]):
        spans = detect(pair.get("document_text") or "", pair.get("model_output") or "")
        if not spans:
            continue
        documents = pair.get("doc_ids") or []
        if any(used_per_doc.get(doc, 0) >= cap for doc in documents):
            continue
        for doc in documents:
            used_per_doc[doc] = used_per_doc.get(doc, 0) + 1
        best = max(span["score"] for span in spans)
        scored.append((best, pair["pair_id"], pair, spans))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:limit]
    pairs = [item[2] for item in selected]
    max_per_doc = max(used_per_doc.values()) if used_per_doc else 0
    overlap = 0
    for pair in pairs:
        if set(pair.get("doc_ids") or []) & reserved:
            overlap += 1
    stats = {
        "pool": len(pool),
        "count": len(pairs),
        "per_doc_cap": cap,
        "doc_overlap_with_gold": overlap,
        "max_per_doc": max_per_doc,
        "scored": len(scored),
    }
    return [(item[2], item[3], item[0]) for item in selected], stats
