from spanver.snap import boundaries, expand, nearest, shrink, snapped

ANSWER = "Завод выпускал детали в 1990 году, а Москва основана в 1147 году князем."


def test_boundaries_include_edges():
    edges = boundaries(ANSWER)
    assert edges[0] == 0
    assert edges[-1] == len(ANSWER)
    assert ANSWER.find(",") in edges or ANSWER.find(",") + 1 in edges


def test_expand_moves_outward():
    start = ANSWER.find("1990")
    span = {"start": start, "end": start + 3}
    left, right = expand(span, boundaries(ANSWER), len(ANSWER))
    assert left <= start
    assert right >= start + 3
    assert "," in ANSWER[left : right + 1] or "." in ANSWER[left : right + 1] or left == 0


def test_shrink_stays_inside():
    start = ANSWER.find("1990")
    span = {"start": start, "end": start + 3}
    left, right = shrink(span, boundaries(ANSWER), len(ANSWER))
    assert start <= left
    assert right <= start + 3


def test_nearest_returns_valid_span():
    start = ANSWER.find("1147")
    span = {"start": start, "end": start + 3}
    left, right = nearest(span, boundaries(ANSWER), len(ANSWER))
    assert 0 <= left <= right < len(ANSWER)


def test_snapped_keeps_count_and_bounds():
    spans = [{"start": 0, "end": 5, "risk": 0.9, "type": "number", "confidence": 0.5}]
    for mode in ("expand", "shrink", "nearest"):
        result = snapped(spans, ANSWER, mode)
        assert len(result) == 1
        assert 0 <= result[0]["start"] <= result[0]["end"] < len(ANSWER)
        assert result[0]["risk"] == 0.9
