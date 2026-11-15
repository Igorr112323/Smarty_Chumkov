from spanver.decode import flags_for, spans_from_flags, token_counts

ANSWER = "Завод выпустил 1200 деталей в 1990 году."


def pieces(*scores):
    return [(index, index + 1, score) for index, score in enumerate(scores)]


def test_isolated_token_is_dropped():
    flags = flags_for(pieces(0.9, 0.1, 0.1), 0.5)
    assert flags == [False, False, False]


def test_neighbor_keeps_pair():
    flags = flags_for(pieces(0.9, 0.7, 0.1), 0.5)
    assert flags == [True, True, False]


def test_threshold_below_scores():
    flags = flags_for(pieces(0.3, 0.3), 0.5)
    assert flags == [False, False]


def test_short_span_is_removed():
    answer = "ab"
    data = [(0, 1, 0.9), (1, 2, 0.9)]
    flags = flags_for(data, 0.5)
    spans = spans_from_flags(answer, data, flags, 3)
    assert spans == []
    spans = spans_from_flags(answer, data, flags, 2)
    assert len(spans) == 1
    assert (spans[0]["start"], spans[0]["end"]) == (0, 1)


def test_span_coordinates_are_inclusive():
    answer = ANSWER
    start = answer.find("1200")
    end = start + 4
    data = [(start, start + 2, 0.9), (start + 2, end, 0.9)]
    flags = flags_for(data, 0.5)
    spans = spans_from_flags(answer, data, flags, 2)
    assert answer[spans[0]["start"] : spans[0]["end"] + 1] == "1200"


def test_token_counts_match_formula():
    cache = {
        "p": {"answer": "abcd", "pieces": [(0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.1), (3, 4, 0.1)]},
        "n": {"answer": "abcd", "pieces": [(0, 1, 0.1), (1, 2, 0.1), (2, 3, 0.1), (3, 4, 0.1)]},
    }
    pairs = [
        {"pair_id": "p", "model_output": "abcd", "spans": [{"start": 0, "end": 1}]},
        {"pair_id": "n", "model_output": "abcd", "spans": []},
    ]
    result = token_counts(cache, pairs, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["fpr"] == 0.0
    expected = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
    assert round(expected, 4) == round(result["f1"], 4)
