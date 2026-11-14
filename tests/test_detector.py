from spanver.detector import THRESHOLD, detect, score_text

OUTPUT = (
    "Плотность населения составила 45% в 1990 году. "
    "Возможно, Иван Петров работал в институте. "
    "Он не пришёл вовремя, зато да.\n"
    "Совсем коротко. "
    "Завод выпустил 1200 деталей."
)


def test_repeatable():
    first = detect("", OUTPUT)
    second = detect("", OUTPUT)
    assert first == second
    assert first


def test_each_feature_contributes():
    number = score_text("рост составил 45% за период")
    plain = score_text("рост отмечен по показателям")
    assert number[0] > plain[0]
    assert "number" in number[1]
    temporal = score_text("это случилось в 1990 году")
    assert "temporal" in temporal[1]
    entity = score_text("он встретил Ивана Петрова в городе")
    assert "entity" in entity[1]
    uncertainty = score_text("возможно это правда и не подтверждено")
    assert any(item.startswith("uncertainty:") for item in uncertainty[1])


def test_short_fragment_dropped():
    found = detect("", "Да, нет.")
    assert found == []
    for span in found:
        assert len(span["text"]) >= 10


def test_year_gives_temporal():
    found = detect("", "Это произошло в 1945 году и стало известно всем.")
    assert found
    assert any("temporal" in span["features"] for span in found)
    assert all(span["score"] >= THRESHOLD for span in found)


def test_threshold_respected():
    found = detect("", "Обычное предложение без признаков вообще.")
    assert found == []
