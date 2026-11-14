import pytest

from spanver.locate import METHOD_EXACT, METHOD_NORMALIZED, LocateError, locate


def test_exact_match():
    result = locate([{"text": "тяжело избежать"}], "В языке Си тяжело избежать использования")
    assert result[0]["start"] == 11
    assert result[0]["end"] == 25
    assert result[0]["method"] == METHOD_EXACT


def test_two_identical_spans():
    result = locate([{"text": "нет"}, {"text": "нет"}], "нет нет")
    assert result[0]["start"] == 0
    assert result[1]["start"] == 4
    assert result[0]["start"] != result[1]["start"]


def test_double_space():
    result = locate([{"text": "Си  тяжело"}], "В языке Си тяжело избежать")
    assert result[0]["method"] == METHOD_NORMALIZED
    assert result[0]["steps"] == 2
    assert result[0]["start"] == 8


def test_quotes():
    result = locate([{"text": "он сказал \"да\""}], "он сказал \u00abда\u00bb")
    assert result[0]["method"] == METHOD_NORMALIZED
    assert result[0]["start"] == 0
    assert result[0]["end"] == 13


def test_dash():
    result = locate([{"text": "Москва-река"}], "Москва\u2014река")
    assert result[0]["method"] == METHOD_NORMALIZED
    assert result[0]["start"] == 0


def test_lower_case():
    result = locate([{"text": "москва"}], "МОСКВА")
    assert result[0]["method"] == METHOD_NORMALIZED
    assert result[0]["start"] == 0
    assert result[0]["end"] == 5


def test_out_of_bounds():
    with pytest.raises(LocateError):
        locate([{"text": "x", "start": 5, "end": 99}], "короткий")


def test_overlap_allowed():
    result = locate([{"text": "abc", "start": 0, "end": 3}, {"text": "cde", "start": 2, "end": 5}], "abcdefgh")
    assert result[0]["start"] == 0
    assert result[1]["start"] == 2
    assert result[1]["end"] == 5


def test_empty_spans():
    assert locate([], "любой текст") == []
