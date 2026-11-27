"""V11: общая лаборатория — кэш кандидатов, document support, быстрые метрики.

Кандидаты и их 37 признаков считаются один раз на пару и переиспользуются всеми
экспериментами блока 2–4 (в v10 пересчёт на каждый конфиг стоил ~34 минуты).
"""
import hashlib
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver import ranker
from spanver.evaluate import iou, load_split, metrics, read_jsonl
from spanver.module import candidate_spans, document_context, tokens

SENT_RE = re.compile(r"[^.!?\n]+[.!?]*")
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)*")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
CLAUSE_CUT = re.compile(r"(?<=[,;:])\s+|(?<=[.!?])\s+|\s+(?=(?:а|но|однако|при этом|кроме того|также|зато|хотя|если|когда|который|которая|которые|что|чтобы|поскольку|поэтому)\b)", re.I)
CONJ_LEFT = ("и", "а", "но", "или", "либо", "что", "чтобы", "который", "которая", "которые",
             "когда", "если", "хотя", "поскольку", "поэтому", "также", "зато", "при", "для",
             "из", "от", "по", "на", "в", "с", "о", "об", "к", "до", "после", "под", "над", "без")


def weights(path="artifacts/ranker_weights_v3.json"):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def set_weights(data):
    ranker._WEIGHTS = data


def cell(result):
    return {
        "span": result["span_level"]["f1"],
        "token": result["token_level"]["f1"],
        "pair": result["pair_level"]["f1"],
        "fpr": result["fpr"],
        "span_p": result["span_level"]["precision"],
        "span_r": result["span_level"]["recall"],
    }


def doc_key(pair):
    return hashlib.sha256((pair.get("document_text") or "").encode()).hexdigest()


def load_all():
    gold = read_jsonl(str(ROOT / "artifacts/gold_v3.jsonl"))
    train = [p for p in gold if p.get("split") == "train"]
    val = [p for p in gold if p.get("split") == "val"]
    mfava = read_jsonl(str(ROOT / "artifacts/mfava_train_v3.jsonl"))
    return train, val, mfava


# ---------------------------------------------------------------- clause layer

def clause_spans(text):
    """Синтаксические куски ответа: по пунктуации и союзам, с точными offsets."""
    text = text or ""
    pieces = []
    for sentence in SENT_RE.finditer(text):
        body = sentence.group(0)
        base = sentence.start()
        positions = [0]
        for match in CLAUSE_CUT.finditer(body):
            positions.append(match.end())
        positions.append(len(body))
        for left, right in zip(positions, positions[1:]):
            chunk = body[left:right]
            stripped = chunk.strip()
            if len(stripped) < 8:
                continue
            offset = left + chunk.find(stripped)
            pieces.append((base + offset, base + offset + len(stripped) - 1, stripped))
    return pieces


def clause_boundaries(text, start, end):
    """Синтаксические границы выбранного фрагмента: 1.0 — граница на пунктуации/союзе."""
    text = text or ""
    left_ok = 0.0
    right_ok = 0.0
    if start == 0:
        left_ok = 1.0
    else:
        before = text[max(0, start - 3):start]
        if re.search(r"[,;:!?.\n]\s*$", before):
            left_ok = 1.0
        else:
            first = (WORD_RE.search(text[start:start + 40]) or [None])
            word = first.group(0).lower() if hasattr(first, "group") else ""
            left_ok = 0.5 * float(word in CONJ_LEFT)
    if end >= len(text) - 1:
        right_ok = 1.0
    else:
        after = text[end + 1:end + 4]
        right_ok = 1.0 if re.match(r"\s*[,;:!?.\n]", after) else 0.0
    return left_ok, right_ok


# ------------------------------------------------------------ document support

@lru_cache(maxsize=40000)
def _doc_index(document):
    sentences = []
    for match in SENT_RE.finditer(document or ""):
        piece = match.group(0).strip()
        if len(piece) < 12:
            continue
        counts = {}
        for word in WORD_RE.findall(piece.lower()):
            counts[word] = counts.get(word, 0) + 1
        sentences.append((piece, counts, set(NUM_RE.findall(piece))))
    return tuple(sentences)


def _cosine(counts, length, other, other_length):
    if not counts or not other:
        return 0.0
    common = 0.0
    for word, value in counts.items():
        if word in other:
            common += value * other[word]
    return common / math.sqrt(length * other_length) if length and other_length else 0.0


def support_features(text, document):
    """Признаки поддержки фрагмента документом — только document-side, без золота."""
    text = text or ""
    words = [w.lower() for w in WORD_RE.findall(text)]
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    length = sum(v * v for v in counts.values())
    sentences = _doc_index(document or "")
    best_cos = 0.0
    best_cov = 0.0
    best_sent = ""
    verbatim = 0.0
    lowered_doc = (document or "").lower()
    if lowered_doc and text.strip().lower() in lowered_doc:
        verbatim = 1.0
    for piece, other, numbers in sentences:
        other_length = sum(v * v for v in other.values())
        cos = _cosine(counts, length, other, other_length)
        covered = sum(1 for word in set(words) if word in other) / max(1, len(set(words)))
        if cos + covered > best_cos + best_cov:
            best_cos, best_cov, best_sent = cos, covered, piece
        del numbers
    doc_words = set(WORD_RE.findall(lowered_doc))
    coverage = sum(1 for word in set(words) if word in doc_words) / max(1, len(set(words)))
    text_numbers = set(NUM_RE.findall(text))
    doc_numbers = set(NUM_RE.findall(lowered_doc))
    numbers_missing = sum(1 for number in text_numbers if number not in doc_numbers) / max(1, len(text_numbers))
    capitals = [w for w in WORD_RE.findall(text) if w[:1].isupper()]
    names_missing = sum(1 for word in capitals if word.lower() not in doc_words) / max(1, len(capitals))
    return {
        "cos": best_cos,
        "sentence_cov": best_cov,
        "token_cov": coverage,
        "verbatim": verbatim,
        "numbers_missing": numbers_missing,
        "names_missing": names_missing,
        "has_numbers": float(bool(text_numbers)),
        "n_words": float(len(words)),
        "n_sentences": float(len(SENT_RE.findall(text))),
        "sentence": best_sent,
    }


# ------------------------------------------------------------- candidate cache

def candidates(pair):
    """Кандидаты пары с 37 признаками baseline + кэш-независимые поля."""
    answer = pair.get("model_output") or ""
    document = pair.get("document_text") or ""
    context = document_context(document)
    doc_tokens, doc_grams = context[0], context[2]
    rows = []
    for start, end, fragment in candidate_spans(answer):
        vector = ranker.features(fragment, start, end, answer, document, doc_tokens, doc_grams)
        rows.append({"start": start, "end": end, "text": fragment, "features": vector})
    return rows


def score(pair, rows, params):
    for row in rows:
        row["risk"] = ranker._logit(row["features"], params["cand"], ranker.FEATURE_COUNT)
    risks = [row["risk"] for row in rows]
    gate = ranker._logit(
        ranker.pair_features([{"features": row["features"]} for row in rows], risks,
                             pair.get("model_output") or ""),
        params["pair"], ranker.PAIR_FEATURE_COUNT,
    )
    return gate


class Cache:
    """Кэш кандидатов и их static-метрик; сохраняется на диск между прогонами."""

    def __init__(self, params=None, path="artifacts/v11_candidate_cache.json"):
        self.params = params or weights()
        self.path = ROOT / path
        self.data = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.dirty = False

    def get(self, pair):
        key = pair["pair_id"]
        entry = self.data.get(key)
        if entry is None:
            rows = candidates(pair)
            gate = score(pair, rows, self.params)
            entry = {
                "gate": gate,
                "rows": [[r["start"], r["end"], r["text"], r["features"], r["risk"]] for r in rows],
            }
            self.data[key] = entry
            self.dirty = True
        rows = [
            {"start": s, "end": e, "text": t, "features": f, "risk": risk}
            for s, e, t, f, risk in entry["rows"]
        ]
        return rows, entry["gate"]

    def save(self):
        if self.dirty:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            self.dirty = False


def nms_pick(rows, threshold, max_spans, nms_iou=0.10, order_key=None):
    order = sorted(
        range(len(rows)),
        key=order_key or (lambda index: (-rows[index]["risk"], rows[index]["start"])),
    )
    accepted = []
    for index in order:
        candidate = rows[index]
        if threshold is not None and candidate["risk"] < threshold:
            break
        if any(iou(candidate["start"], candidate["end"], other["start"], other["end"]) > nms_iou
               for other in accepted):
            continue
        accepted.append(candidate)
        if len(accepted) >= max_spans:
            break
    return accepted


def as_spans(rows):
    from spanver.module import predict_type, confidence
    return [
        {"start": row["start"], "end": row["end"], "risk": round(row.get("risk", 0.0), 4),
         "type": predict_type(row["text"], 0.0), "confidence": confidence(row.get("risk", 0.0))}
        for row in rows
    ]


def evaluate(pairs, cache, decode, ):
    """decode(pair, rows, gate) -> list выбранных rows."""
    picked = {}
    for pair in pairs:
        rows, gate = cache.get(pair)
        picked[pair["pair_id"]] = as_spans(decode(pair, rows, gate) or [])

    def predictor(pair):
        return picked[pair["pair_id"]]

    return metrics(pairs, predictor)


def gold_rank(rows, pair):
    """Ранг лучшего по IoU кандидата для каждого gold-спана (без учёта порогов)."""
    gold = pair.get("spans") or []
    order = sorted(range(len(rows)), key=lambda index: (-rows[index]["risk"], rows[index]["start"]))
    rank_of = {index: position + 1 for position, index in enumerate(order)}
    result = []
    for truth in gold:
        best_index, best_iou = None, 0.0
        for index, row in enumerate(rows):
            value = iou(row["start"], row["end"], truth["start"], truth["end"])
            if value > best_iou:
                best_index, best_iou = index, value
        result.append({
            "iou": round(best_iou, 4),
            "rank": rank_of.get(best_index) if best_index is not None else None,
            "oracle_ok": best_iou >= 0.5,
        })
    return result


def guard(cells):
    """Guard: все три F1 >= 0.50 при FPR <= 0.40 одновременно."""
    return all(
        item["span"] >= 0.5 and item["token"] >= 0.5 and item["pair"] >= 0.5 and item["fpr"] <= 0.4
        for item in cells
    )


def write_json(path, payload):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load(path):
    return read_jsonl(str(ROOT / path))
