import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import build_tiny

from spanver.model_features import (
    ATTENTION_FEATURES,
    HEAD_FEATURES,
    MAX_MODEL_CANDIDATES,
    chunk_text,
    index_chunks,
    load_model,
    pair_features,
    retrieve,
    select_candidates,
    token_attention,
    weights_sha,
)

ANSWER = "Завод выпустил 1200 деталей в 1990 году. Москва основана в 1147 году князем."
DOCUMENT = "Завод выпускал детали в 1990 году. Москва — древний город, основанный князем."


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny")
    build_tiny(directory)
    return load_model(directory)


def test_shapes_and_ranges(bundle):
    pieces, stems = index_chunks(DOCUMENT)
    values = token_attention(bundle, pieces[0], "Москва основана в 1147 году князем.")
    assert values
    assert len(values["share"]) == len(values["entropy"]) == len(values["cos"])
    assert all(0.0 <= value <= 1.0 for value in values["share"])
    assert all(0.0 <= value <= 1.0 for value in values["entropy"])
    assert all(-1.0 <= value <= 1.0 for value in values["cos"])


def test_invalid_position_limit(bundle):
    assert token_attention(bundle, "", "текст") is None
    assert token_attention(bundle, "текст", "") is None


def test_features_deterministic(bundle):
    first = pair_features(bundle, ANSWER, DOCUMENT)
    second = pair_features(bundle, ANSWER, DOCUMENT)
    assert first == second


def test_retrieval_deterministic_and_ordered():
    pieces, stems = index_chunks(DOCUMENT, size=40)
    chosen = retrieve("Москва основана князем", pieces, stems, top=2)
    assert chosen == sorted(chosen)
    assert 0 <= len(chosen) <= 2
    assert retrieve("Москва основана князем", pieces, stems, top=2) == chosen


def test_chunks_cover_text():
    pieces = chunk_text(DOCUMENT, size=50)
    assert "".join(pieces) == DOCUMENT
    assert all(len(piece) <= 120 for piece in pieces)


def test_candidate_limit():
    long_answer = " ".join([ANSWER] * 40)
    chosen = select_candidates(long_answer, DOCUMENT, limit=7)
    assert len(chosen) == 7
    assert chosen == sorted(chosen, key=lambda item: item[0])
    everything = select_candidates(ANSWER, DOCUMENT)
    assert len(everything) <= MAX_MODEL_CANDIDATES


def test_head_vector_layout(bundle):
    rows = pair_features(bundle, ANSWER, DOCUMENT)
    assert rows
    for row in rows:
        assert len(row["values"]) == len(HEAD_FEATURES)
        assert 0 <= row["start"] <= row["end"] < len(ANSWER)
        assert row["values"][0] == 1.0


def test_feature_names_match_contract():
    assert ATTENTION_FEATURES == (
        "doc_attention_share",
        "attention_entropy",
        "cos_proximity",
        "doc_overlap",
        "uncertainty",
    )
    assert len(HEAD_FEATURES) == 10
