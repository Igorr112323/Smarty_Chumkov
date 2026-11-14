import hashlib
import json
import math
from pathlib import Path

from .module import (
    LONG_ANSWER,
    SENTENCE_RE,
    candidate_spans,
    document_context,
    feature_vector,
    score,
    stem,
    tokens,
)

MODEL_NAME = "ai-forever/ruSciBERT"
MAX_MODEL_CANDIDATES = 50
TOP_SEGMENTS = 2
SEGMENT_TOKENS = 256
FRAGMENT_TOKENS = 128
ATTENTION_LAYERS = 4
MODEL_THREADS = 4
MODEL_SEED = 0
DOC_CHUNK_CHARS = 1000
STREAM_CHARS = 5000
CACHE_FILE = "artifacts/model_cache.jsonl"
WEIGHT_SUFFIXES = (".safetensors", ".bin")

ATTENTION_FEATURES = (
    "doc_attention_share",
    "attention_entropy",
    "cos_proximity",
    "doc_overlap",
    "uncertainty",
)
HEAD_FEATURES = (
    "bias",
    "doc_attention_share_mean",
    "doc_attention_share_max",
    "attention_entropy_mean",
    "attention_entropy_max",
    "cos_mean",
    "cos_max",
    "doc_overlap",
    "uncertainty",
    "relative_length",
)


class ModelUnavailable(RuntimeError):
    pass


def runtime():
    try:
        import torch
    except ImportError as error:
        raise ModelUnavailable("torch не установлен") from error
    torch.set_num_threads(MODEL_THREADS)
    torch.manual_seed(MODEL_SEED)
    return torch


def weights_files(directory):
    path = Path(directory)
    if not path.is_dir():
        raise ModelUnavailable(f"веса не найдены: {path}")
    files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix in WEIGHT_SUFFIXES)
    config = path / "config.json"
    if not config.is_file() or not files:
        raise ModelUnavailable(f"веса не найдены: {path}")
    return [config] + files


def weights_sha(directory):
    digest = hashlib.sha256()
    for path in weights_files(directory):
        digest.update(path.name.encode("utf-8"))
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_model(directory):
    torch = runtime()
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ModelUnavailable("transformers не установлен") from error
    weights_files(directory)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = AutoModel.from_pretrained(str(directory), local_files_only=True, attn_implementation="eager")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "torch": torch,
        "model": model,
        "tokenizer": tokenizer,
        "sha": weights_sha(directory),
        "name": MODEL_NAME,
        "directory": str(directory),
    }


def chunk_text(text, size=DOC_CHUNK_CHARS):
    pieces = []
    buffer = ""
    for match in SENTENCE_RE.finditer(text or ""):
        piece = match.group(0)
        if buffer and len(buffer) + len(piece) > size:
            pieces.append(buffer)
            buffer = piece
        else:
            buffer += piece
    if buffer.strip():
        pieces.append(buffer)
    return [piece for piece in pieces if piece.strip()]


def index_chunks(document, size=DOC_CHUNK_CHARS):
    pieces = chunk_text(document, size)
    stems = [{stem(token) for token in tokens(piece)} for piece in pieces]
    return pieces, stems


def retrieve(fragment, pieces, stems, top=TOP_SEGMENTS):
    wanted = {stem(token) for token in tokens(fragment)}
    ranked = []
    for index, chunk_stems in enumerate(stems):
        if not chunk_stems or not wanted:
            continue
        overlap = len(wanted & chunk_stems)
        if overlap == 0:
            continue
        ranked.append((-overlap / len(chunk_stems), index))
    ranked.sort()
    chosen = sorted(index for _, index in ranked[:top])
    if chosen:
        return chosen
    return [0] if pieces else []


def select_candidates(answer, document, limit=MAX_MODEL_CANDIDATES):
    text = answer or ""
    doc = document or ""
    document_tokens, document_stems, document_ngrams = document_context(doc)
    scored = []
    windows = []
    if len(text) <= LONG_ANSWER:
        windows.append((0, len(text)))
    else:
        start = 0
        while start < len(text):
            end = min(len(text), start + STREAM_CHARS)
            while end < len(text) and text[end] not in ".!?\n":
                end += 1
            if end <= start:
                end = min(len(text), start + STREAM_CHARS)
            windows.append((start, end))
            start = end
    seen = set()
    for window_start, window_end in windows:
        window = text[window_start:window_end]
        for start, end, fragment in candidate_spans(window):
            absolute_start = window_start + start
            absolute_end = window_start + end
            key = (absolute_start, absolute_end)
            if key in seen:
                continue
            seen.add(key)
            whole = absolute_start == 0 and absolute_end == len(text) - 1
            values, _ = feature_vector(
                fragment, len(text), document_tokens, whole, document_stems, document_ngrams, absolute_start, len(doc)
            )
            scored.append((score(values), absolute_start, absolute_end, fragment))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted(scored[:limit], key=lambda item: item[1])
    return [(start, end, fragment) for _, start, end, fragment in chosen]


def mean_max(values):
    if not values:
        return 0.0, 0.0
    return sum(values) / len(values), max(values)


def token_attention(bundle, segment_text, fragment_text):
    torch = bundle["torch"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    seg_ids = tokenizer(segment_text, truncation=True, max_length=SEGMENT_TOKENS, add_special_tokens=False)["input_ids"]
    frag_ids = tokenizer(fragment_text, truncation=True, max_length=FRAGMENT_TOKENS, add_special_tokens=False)["input_ids"]
    if not seg_ids or not frag_ids:
        return None
    sequence = [tokenizer.cls_token_id] + seg_ids + [tokenizer.sep_token_id] + frag_ids + [tokenizer.sep_token_id]
    limit = int(getattr(model.config, "max_position_embeddings", 512))
    sequence = sequence[:limit]
    doc_start = 1
    doc_stop = doc_start + len(seg_ids)
    answer_start = doc_stop + 1
    answer_stop = min(answer_start + len(frag_ids), len(sequence))
    if answer_stop <= answer_start or doc_stop <= doc_start:
        return None
    input_ids = torch.tensor([sequence], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
    layers = output.attentions[-ATTENTION_LAYERS:]
    stacked = torch.stack([layer[0].mean(dim=0) for layer in layers])
    doc_matrix = stacked[:, answer_start:answer_stop, doc_start:doc_stop].mean(dim=0)
    share = doc_matrix.sum(dim=1).clamp(0.0, 1.0)
    normalised = doc_matrix / doc_matrix.sum(dim=1, keepdim=True).clamp_min(1e-9)
    doc_tokens = doc_matrix.shape[1]
    if doc_tokens > 1:
        entropy = -(normalised * normalised.clamp_min(1e-9).log()).sum(dim=1) / math.log(doc_tokens)
    else:
        entropy = torch.zeros_like(share)
    entropy = entropy.clamp(0.0, 1.0)
    hidden = output.last_hidden_state[0]
    segment_vector = hidden[doc_start:doc_stop].mean(dim=0)
    answer_vectors = hidden[answer_start:answer_stop]
    cosines = torch.nn.functional.cosine_similarity(answer_vectors, segment_vector.unsqueeze(0), dim=1)
    cosines = cosines.clamp(-1.0, 1.0)
    return {
        "share": [float(value) for value in share],
        "entropy": [float(value) for value in entropy],
        "cos": [float(value) for value in cosines],
    }


def fragment_attention(bundle, fragment, pieces, stems):
    best_key = None
    best_values = None
    for index in retrieve(fragment, pieces, stems):
        values = token_attention(bundle, pieces[index], fragment)
        if values is None:
            continue
        key = sum(values["cos"]) / len(values["cos"])
        if best_key is None or key > best_key:
            best_key = key
            best_values = values
    return best_values


def head_vector(values, doc_overlap, uncertainty, relative_length):
    if values is None:
        share_mean = share_max = entropy_mean = entropy_max = cos_mean = cos_max = 0.0
    else:
        share_mean, share_max = mean_max(values["share"])
        entropy_mean, entropy_max = mean_max(values["entropy"])
        cos_mean, cos_max = mean_max(values["cos"])
    return [
        1.0,
        share_mean,
        share_max,
        entropy_mean,
        entropy_max,
        cos_mean,
        cos_max,
        doc_overlap,
        uncertainty,
        relative_length,
    ]


def pair_features(bundle, answer, document, limit=MAX_MODEL_CANDIDATES):
    text = answer or ""
    doc = document or ""
    pieces, stems = index_chunks(doc)
    document_tokens, document_stems, document_ngrams = document_context(doc)
    rows = []
    for start, end, fragment in select_candidates(text, doc, limit):
        values, support = feature_vector(
            fragment, len(text), document_tokens, False, document_stems, document_ngrams, start, len(doc)
        )
        attention = fragment_attention(bundle, fragment, pieces, stems)
        relative = (end - start + 1) / len(text) if text else 0.0
        rows.append(
            {
                "start": start,
                "end": end,
                "support": round(support, 6),
                "values": [round(value, 6) for value in head_vector(attention, values[4], values[2], relative)],
            }
        )
    return rows


def cache_key(pair_id, sha):
    return f"{pair_id}|{sha[:16]}"


def read_cache(path=CACHE_FILE):
    target = Path(path)
    rows = {}
    if not target.exists():
        return rows
    with open(target, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["key"]] = row
    return rows


def write_cache(rows, path=CACHE_FILE):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(rows):
            handle.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    return target
