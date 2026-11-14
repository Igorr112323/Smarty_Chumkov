import hashlib
import re

HEADER_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.M)

FIELDS = (
    "pair_id",
    "source",
    "domain",
    "query_id",
    "query_text",
    "document_text",
    "doc_ids",
    "model_output",
    "spans",
    "clean",
    "label_source",
    "perturb",
    "split",
)

SPAN_FIELDS = ("text", "start", "end", "type_src", "type", "evidence_ids")


def doc_ids(document_text):
    text = document_text or ""
    found = []
    for match in HEADER_RE.finditer(text):
        ident = match.group(1).strip()
        if ident and ident not in found:
            found.append(ident)
    if not found:
        found.append("sha1:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])
    return found


def ordered(pair):
    return {name: pair.get(name) for name in FIELDS}
