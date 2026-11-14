from spanver.module import FEATURE_NAMES, THRESHOLD, token_features, verify

TYPES = {"number", "temporal", "entity", "omission", "unsupported_addition", "contradiction", "scope"}
ANSWER = "В 1990 году завод выпустил 1200 деталей. Плотность составила 45%. Возможно, это неверно."
DOCUMENT = "Завод выпускает детали с 1990 года."


def test_schema_and_ranges():
    result = verify(ANSWER, DOCUMENT)
    assert set(result) == {"spans", "ai_share", "meta"}
    assert 0.0 <= result["ai_share"] <= 1.0
    assert result["meta"]["answer_len"] == len(ANSWER)
    assert result["meta"]["doc_len"] == len(DOCUMENT)
    assert result["meta"]["n_spans"] == len(result["spans"])
    for span in result["spans"]:
        assert 0 <= span["start"] <= span["end"] < len(ANSWER)
        assert span["type"] in TYPES
        assert 0.0 <= span["risk"] <= 1.0
        assert 0.0 <= span["confidence"] <= 1.0
        assert span["risk"] >= THRESHOLD


def test_deterministic():
    first = verify(ANSWER, DOCUMENT)
    second = verify(ANSWER, DOCUMENT)
    assert first["spans"] == second["spans"]
    assert first["ai_share"] == second["ai_share"]


def test_empty_answer():
    result = verify("", DOCUMENT)
    assert result["spans"] == []
    assert result["ai_share"] == 0.0
    assert result["meta"]["answer_len"] == 0


def test_five_features_per_token():
    features = token_features("Завод 45% в 1990 году", {"завод": True})
    assert len(FEATURE_NAMES) == 5
    for item in features:
        assert len(item) == len(FEATURE_NAMES) + 1
    assert features


def test_supported_answer_has_no_spans():
    result = verify("Завод выпускает детали.", "Завод выпускает детали.")
    assert result["spans"] == []
