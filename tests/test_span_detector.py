import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from spanver.span_detector import (
    CHUNK_TOKENS,
    answer_labels,
    choose_chunks,
    document_chunks,
    encode_pair,
    overlap_score,
    spans_from,
)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "rubert-tiny2"
DOCUMENT = "В языке Си тяжело избежать потенциально опасных элементов. " * 120
ANSWER = "Москва основана в 1147 году князем Юрием."


@pytest.fixture(scope="module")
def tokenizer():
    if not (WEIGHTS / "tokenizer.json").is_file():
        pytest.skip("нет токенизатора rubert-tiny2")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(WEIGHTS), local_files_only=True)


def test_chunks_cover_document_with_overlap(tokenizer):
    chunks = document_chunks(tokenizer, DOCUMENT, CHUNK_TOKENS)
    assert len(chunks) > 1
    assert chunks[0]["start"] == 0
    assert chunks[-1]["end"] >= len(DOCUMENT.strip()) - 15
    for chunk in chunks:
        assert chunk["tokens"] <= CHUNK_TOKENS
        assert chunk["text"] == DOCUMENT[chunk["start"] : chunk["end"]]
    for previous, current in zip(chunks, chunks[1:]):
        assert current["start"] < previous["end"]


def test_answer_offsets_match_manual_search(tokenizer):
    encoded = encode_pair(tokenizer, "Москва была основана в 1147 году.", ANSWER)
    offsets = [(start, end) for _, start, end in encoded["answer_tokens"]]
    for word in ("Москва", "основана", "1147", "князем"):
        manual = ANSWER.find(word)
        assert (manual, manual + len(word)) in offsets


def test_spans_from_uses_answer_coordinates(tokenizer):
    encoded = encode_pair(tokenizer, "Москва была основана в 1147 году.", ANSWER)
    cached = {"answer": ANSWER, "pieces": [(start, end, 0.9) for _, start, end in encoded["answer_tokens"]]}
    spans = spans_from(cached, 0.5)
    assert len(spans) == 1
    span = spans[0]
    assert span["start"] == ANSWER.find("Москва")
    assert ANSWER[span["start"] : span["end"] + 1] == ANSWER
    assert span["type"]
    assert 0.0 <= span["confidence"] <= 1.0


def test_answer_labels_mark_gold_overlap(tokenizer):
    encoded = encode_pair(tokenizer, "Москва была основана в 1147 году.", ANSWER)
    start = ANSWER.find("1147")
    gold = [{"start": start, "end": start + 3}]
    labels = answer_labels(encoded, gold)
    marked = sum(1 for label in labels if label == 1)
    assert marked >= 1
    zero_word = ANSWER.find("князем")
    for index, token_start, token_end in encoded["answer_tokens"]:
        if token_start >= zero_word:
            assert labels[index] == 0


def test_choose_chunks_prefers_overlapping_chunk(tokenizer):
    answer = "Тяжело избежать потенциально опасных элементов в языке Си."
    document = "Совершенно посторонний текст про погоду и урожай. " * 60 + DOCUMENT
    pair = {"pair_id": "t1", "model_output": answer, "document_text": document}
    best = choose_chunks(tokenizer, pair, top=1)
    assert len(best) == 1
    assert overlap_score(answer, best[0]["text"]) > 0.9
    four = choose_chunks(tokenizer, pair, top=4)
    assert len(four) == 4
    assert [chunk["index"] for chunk in four] == sorted({chunk["index"] for chunk in four})
